import pycolmap
import numpy as np

import torch
import torch.nn.functional as F
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from tensor_to_pycolmap import batch_matrix_to_pycolmap, pycolmap_to_batch_matrix

from lightglue import ALIKED, SuperPoint, SIFT


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def generate_rank_by_dino(
    images, query_frame_num, image_size=518, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=True
):
    """
    Generate a ranking of frames using DINO ViT features.

    Args:
        images: Tensor of shape (S, 3, H, W) with values in range [0, 1]
        query_frame_num: Number of frames to select
        image_size: Size to resize images to before processing
        model_name: Name of the DINO model to use
        device: Device to run the model on
        spatial_similarity: Whether to use spatial token similarity or CLS token similarity

    Returns:
        List of frame indices ranked by their representativeness
    """
    dino_v2_model = torch.hub.load('facebookresearch/dinov2', model_name)
    dino_v2_model.eval()
    dino_v2_model = dino_v2_model.to(device)

    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images - resnet_mean) / resnet_std

    with torch.no_grad():
        frame_feat = dino_v2_model(images_resnet_norm, is_training=True)

    if spatial_similarity:
        frame_feat = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

        # Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
        similarity_matrix = torch.bmm(
            frame_feat_norm, frame_feat_norm.transpose(-1, -2)
        )
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat = frame_feat["x_norm_clstoken"]
        frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)
        similarity_matrix = torch.mm(
            frame_feat_norm, frame_feat_norm.transpose(-1, -2)
        )

    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()

    # Conduct FPS sampling starting from the most common frame
    fps_idx = farthest_point_sampling(
        distance_matrix, query_frame_num, most_common_frame_index
    )

    return fps_idx


def farthest_point_sampling(
    distance_matrix, num_samples, most_common_frame_index=0
):
    """
    Farthest point sampling algorithm to select diverse frames.

    Args:
        distance_matrix: Matrix of distances between frames
        num_samples: Number of frames to select
        most_common_frame_index: Index of the first frame to select

    Returns:
        List of selected frame indices
    """
    distance_matrix = distance_matrix.clamp(min=0)
    N = distance_matrix.size(0)

    # Initialize with the most common frame
    selected_indices = [most_common_frame_index]
    check_distances = distance_matrix[selected_indices]

    while len(selected_indices) < num_samples:
        # Find the farthest point from the current set of selected points
        farthest_point = torch.argmax(check_distances)
        selected_indices.append(farthest_point.item())

        check_distances = distance_matrix[farthest_point]
        # Mark already selected points to avoid selecting them again
        check_distances[selected_indices] = 0

        # Break if all points have been selected
        if len(selected_indices) == N:
            break

    return selected_indices


def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that switches [query_index] and [0]
    so that the content of query_index would be placed at [0].

    Args:
        query_index: Index to swap with 0
        S: Total number of elements
        device: Device to place the tensor on

    Returns:
        Tensor of indices with the swapped order
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    """
    Reorder tensors along a specific dimension according to the given order.

    Args:
        tensors: List of tensors to reorder
        order: Tensor of indices specifying the new order
        dim: Dimension along which to reorder

    Returns:
        List of reordered tensors
    """
    return [
        torch.index_select(tensor, dim, order) if tensor is not None else None
        for tensor in tensors
    ]


def predict_track(model, images, query_points, dtype=torch.bfloat16, use_tf32_for_track=True, iters=4):
    """
    Predict tracks for query points across frames.

    Args:
        model: VGGT model
        images: Tensor of images of shape (S, 3, H, W)
        query_points: Query points to track
        dtype: Data type for computation
        use_tf32_for_track: Whether to use TF32 precision for tracking
        iters: Number of iterations for tracking

    Returns:
        Predicted tracks, visibility scores, and confidence scores
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            if not use_tf32_for_track:
                track_list, vis_score, conf_score = model.track_head(
                    aggregated_tokens_list, images, ps_idx, query_points=query_points, iters=iters
                )

        if use_tf32_for_track:
            with torch.cuda.amp.autocast(enabled=False):
                track_list, vis_score, conf_score = model.track_head(
                    aggregated_tokens_list, images, ps_idx, query_points=query_points, iters=iters
                )

    pred_track = track_list[-1]

    return pred_track.squeeze(0), vis_score.squeeze(0), conf_score.squeeze(0)


def initialize_feature_extractors(max_query_num, det_thres, extractor_method="aliked", device="cuda"):
    """
    Initialize feature extractors that can be reused based on a method string.

    Args:
        max_query_num: Maximum number of keypoints to extract
        det_thres: Detection threshold for keypoint extraction
        extractor_method: String specifying which extractors to use (e.g., "aliked", "sp+sift", "aliked+sp+sift")
        device: Device to run extraction on

    Returns:
        Dictionary of initialized extractors
    """
    extractors = {}
    methods = extractor_method.lower().split('+')

    for method in methods:
        method = method.strip()
        if method == "aliked":
            aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors['aliked'] = aliked_extractor.to(device).eval()
        elif method == "sp":
            sp_extractor = SuperPoint(max_num_keypoints=max_query_num, detection_threshold=det_thres)
            extractors['sp'] = sp_extractor.to(device).eval()
        elif method == "sift":
            sift_extractor = SIFT(max_num_keypoints=max_query_num)
            extractors['sift'] = sift_extractor.to(device).eval()
        else:
            print(f"Warning: Unknown feature extractor '{method}', ignoring.")

    if not extractors:
        print(f"Warning: No valid extractors found in '{extractor_method}'. Using ALIKED by default.")
        aliked_extractor = ALIKED(max_num_keypoints=max_query_num, detection_threshold=det_thres)
        extractors['aliked'] = aliked_extractor.to(device).eval()

    return extractors


def extract_keypoints(query_image, extractors, max_query_num):
    """
    Extract keypoints using pre-initialized feature extractors.

    Args:
        query_image: Input image tensor (3xHxW, range [0, 1])
        extractors: Dictionary of initialized extractors

    Returns:
        Tensor of keypoint coordinates (1xNx2)
    """
    query_points_round = None

    with torch.no_grad():
        for extractor_name, extractor in extractors.items():
            query_points_data = extractor.extract(query_image)
            extractor_points = query_points_data["keypoints"].round()

            if query_points_round is not None:
                query_points_round = torch.cat([query_points_round, extractor_points], dim=1)
            else:
                query_points_round = extractor_points

    if query_points_round.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points_round.shape[1])[
            :max_query_num
        ]
        query_points_round = query_points_round[:, random_point_indices, :]

    return query_points_round


def run_vggt_with_ba(
    model, 
    images, 
    image_names=None, 
    dtype=torch.bfloat16,
    max_query_num=2048, 
    det_thres=0.005, 
    query_frame_num=3,
    extractor_method="aliked+sp+sift",
    max_reproj_error=12,
    shared_camera=True,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Run VGGT with bundle adjustment for pose estimation.

    Args:
        model: VGGT model for feature extraction and tracking
        images: Tensor of images of shape (S, 3, H, W)
        image_names: Optional list of image names
        dtype: Data type for computation (default: torch.bfloat16)
        max_query_num: Maximum number of query points to track (default: 2048)
        det_thres: Detection threshold for keypoint extraction (default: 0.005)
        query_frame_num: Number of frames to select for feature extraction (default: 3)
        extractor_method: Feature extraction method (default: "aliked+sp+sift")
        max_reproj_error: Maximum reprojection error for bundle adjustment (default: 12)
        shared_camera: Whether to use shared camera parameters (default: True)
        camera_type: Camera model type (default: "SIMPLE_PINHOLE")

    Returns:
        Predicted extrinsic camera parameters

    TODO:
        - [ ] Use VGGT's vit instead of dinov2 for rank generation
    """

    assert "RADIAL" not in camera_type, "RADIAL camera is not supported yet"

    device = images.device
    frame_num = images.shape[0]

    # Select representative frames for feature extraction
    query_frame_indexes = generate_rank_by_dino(
        images, query_frame_num, image_size=518,
        model_name="dinov2_vitb14_reg", device=device,
        spatial_similarity=False
    )

    # Add the first image to the front if not already present
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # Get initial pose and depth predictions
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images) # TODO: point map head is redundant here, remove it

    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        pred_extrinsic = extrinsic[0]
        pred_intrinsic = intrinsic[0]

        # Get 3D points from depth map
        # You can also directly use the point map head to get 3D points, but its performance is slightly worse than the depth map
        depth_map, depth_conf = predictions["depth"][0], predictions["depth_conf"][0]
        world_points = unproject_depth_map_to_point_map(depth_map, pred_extrinsic, pred_intrinsic)
        world_points = torch.from_numpy(world_points).to(device)
        world_points_conf = depth_conf.to(device)

    torch.cuda.empty_cache()

    # Lists to store predictions
    pred_tracks = []
    pred_vis_scores = []
    pred_conf_scores = []
    pred_world_points = []
    pred_world_points_conf = []

    # Initialize feature extractors
    extractors = initialize_feature_extractors(max_query_num, det_thres, extractor_method, device)

    # Process each query frame
    for query_index in query_frame_indexes:
        query_image = images[query_index]
        query_points_round = extract_keypoints(query_image, extractors, max_query_num)

        # Reorder images to put query image first
        reorder_index = calculate_index_mappings(query_index, frame_num, device=device)
        reorder_images = switch_tensor_order([images], reorder_index, dim=0)[0]

        # Track points across frames
        reorder_tracks, reorder_vis_score, reorder_conf_score = predict_track(
            model, reorder_images, query_points_round, dtype=dtype, use_tf32_for_track=True, iters=4,
        )

        # Restore original order
        pred_track, pred_vis, pred_score = switch_tensor_order(
            [reorder_tracks, reorder_vis_score, reorder_conf_score], reorder_index, dim=0
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_conf_scores.append(pred_score)

        # Get corresponding 3D points
        query_points_round_long = query_points_round.squeeze(0).long()
        query_world_points = world_points[query_index][
            query_points_round_long[:, 1], query_points_round_long[:, 0]
        ]
        query_world_points_conf = world_points_conf[query_index][
            query_points_round_long[:, 1], query_points_round_long[:, 0]
        ]

        pred_world_points.append(query_world_points)
        pred_world_points_conf.append(query_world_points_conf)

    # Concatenate prediction lists
    pred_tracks = torch.cat(pred_tracks, dim=1)
    pred_vis_scores = torch.cat(pred_vis_scores, dim=1)
    pred_conf_scores = torch.cat(pred_conf_scores, dim=1)
    pred_world_points = torch.cat(pred_world_points, dim=0)
    pred_world_points_conf = torch.cat(pred_world_points_conf, dim=0)

    # Filter points by confidence
    filtered_flag = pred_world_points_conf > 1.5

    if filtered_flag.sum() > max_query_num // 2:
        # Only filter if we have enough high-confidence points
        pred_world_points = pred_world_points[filtered_flag]
        pred_world_points_conf = pred_world_points_conf[filtered_flag]
        pred_tracks = pred_tracks[:, filtered_flag]
        pred_vis_scores = pred_vis_scores[:, filtered_flag]
        pred_conf_scores = pred_conf_scores[:, filtered_flag]

    torch.cuda.empty_cache()

    # Bundle adjustment parameters
    S, _, H, W = images.shape
    image_size = torch.tensor([W, H], dtype=pred_tracks.dtype, device=device)
    masks = torch.logical_and(pred_vis_scores > 0.05, pred_conf_scores > 0.05)

    # Convert to pycolmap format and run bundle adjustment
    reconstruction, valid_track_mask = batch_matrix_to_pycolmap(
        pred_world_points,
        pred_extrinsic,
        pred_intrinsic,
        pred_tracks,
        image_size,
        masks=masks,
        max_reproj_error=max_reproj_error,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    
    if reconstruction is None:
        return pred_extrinsic

    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    _, pred_extrinsic, _, _ = pycolmap_to_batch_matrix(
        reconstruction, device=device, camera_type=camera_type
    )

    return pred_extrinsic
