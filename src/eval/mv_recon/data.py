import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
import random
from eval.mv_recon.base import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import eval.mv_recon.dataset_utils.cropping as cropping
import imageio.v3 as iio
from tifffile import tifffile
from einops import rearrange


def shuffle_deque(dq, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Convert deque to list, shuffle, and convert back
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = os.listdir(base_dir)

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = img_idxs[:: self.kf_every]

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            rgb_image = imread_cv2(impath)

            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views

class ETH3D(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=10,
        full_video=False,
        kf_every=50,
        shuffle_seed=-1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq     = num_seq
        self.num_frames  = num_frames
        self.full_video  = full_video
        self.kf_every    = kf_every
        self.shuffle_seed = shuffle_seed
        self._load_all_scenes(ROOT)

    @staticmethod
    def _read_cameras(path):
        cams = {}
        with open(path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                toks = line.strip().split()
                cam_id = int(toks[0])
                w, h   = map(int, toks[2:4])
                params = list(map(float, toks[4:]))
                # PINHOLE: fx fy cx cy
                fx, fy, cx, cy = params[:4]
                K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], np.float32)
                cams[cam_id] = dict(K=K, size=(w, h))
        return cams

    @staticmethod
    def _qvec2rotmat(q):
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
            [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
        ], np.float32)

    @staticmethod
    def _read_images(path):
        extrinsics = {}      # image_name -> (T_world2cam, cam_id)
        with open(path) as f:
            lines = [l for l in f.readlines() if l and not l.startswith('#')]
        for i in range(0, len(lines), 2):
            toks = lines[i].strip().split()
            q   = list(map(float, toks[1:5]))
            t   = list(map(float, toks[5:8]))
            cam_id = int(toks[8])
            name   = toks[9]
            R = ETH3D._qvec2rotmat(q)
            t = np.asarray(t, np.float32).reshape(3, 1)
            Tw2c = np.eye(4, dtype=np.float32)
            Tw2c[:3,:3] = R
            Tw2c[:3, 3] = t[:,0]
            extrinsics[name] = (Tw2c, cam_id)
        return extrinsics

    @staticmethod
    def _read_depth_raw(path, shape):
        h, w = shape
        depth = np.fromfile(path, "<f4")
        depth = rearrange(depth, "(h w) -> h w", h=h, w=w)
        depth[np.isinf(depth)] = 0
        depth[np.isnan(depth)] = 0
        return depth

    def _load_all_scenes(self, root):
        self.scene_list = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        print(f"[ETH3D] Found {len(self.scene_list)} scenes.")
        self.scenes = self.scene_list

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]        # e.g. courtyard
        seq_idx  = idx % self.num_seq

        scene_dir  = os.path.join(self.ROOT, scene_id)
        calib_dir  = os.path.join(scene_dir, 'dslr_calibration_jpg')

        cams = self._read_cameras(os.path.join(calib_dir, 'cameras.txt'))
        exts = self._read_images (os.path.join(calib_dir, 'images.txt'))

        img_names = sorted(exts.keys())
        if not self.full_video:
            if self.shuffle_seed >= 0:
                rng_local = np.random.default_rng(self.shuffle_seed + seq_idx)
                rng_local.shuffle(img_names)
            else:
                rng.shuffle(img_names)
            img_names = img_names[: self.num_frames]
        else:
            img_names = img_names[:: self.kf_every]

        views = []
        for name in img_names:
            img_path   = os.path.join(scene_dir, 'images', name)
            depth_fname = os.path.basename(name)
            depth_path = os.path.join(scene_dir,
                                      'ground_truth_depth',
                                      'dslr_images',
                                      depth_fname)

            # intrinsics & extrinsics
            Tw2c, cam_id = exts[name]
            Tc2w = np.linalg.inv(Tw2c)
            K = cams[cam_id]['K']
            w, h = cams[cam_id]['size']

            # load RGB
            rgb = imread_cv2(img_path)                           # BGR uint8
            if rgb.shape[0]!=h or rgb.shape[1]!=w:
                rgb = cv2.resize(rgb, (w,h), interpolation=cv2.INTER_AREA)

            # load depth
            depth = self._read_depth_raw(depth_path, (h, w))     # float32, metres
            if resolution != (224, 224):
                rgb, depth, K_ = self._crop_resize_if_necessary(
                    rgb, depth, K, resolution, rng=rng, info=img_path
                )
            else:
                rgb, depth, K_ = self._crop_resize_if_necessary(
                    rgb, depth, K, resolution=(512, 384), rng=rng, info=img_path
                )
                W, H = rgb.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb, depth, K_ = cropping.crop_image_depthmap(
                    rgb, depth, K_, crop_bbox
                )

            views.append(dict(
                img=rgb,
                depthmap=depth,
                camera_pose=Tc2w,
                camera_intrinsics=K_,
                dataset='eth3d',
                label=f"{scene_id}/{name}",
                instance=img_path,
            ))
        return views


class DTU(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=49,
        num_frames=10,
        min_thresh=10,
        max_thresh=30,
        test_id=None,
        full_video=False,
        sample_pairs=False,
        kf_every=1,
        random_seed=0,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sample_pairs = sample_pairs
        self.random_seed = random_seed 

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        if self.test_id is None:
            self.scene_list = os.listdir(osp.join(base_dir))
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")

        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]

            print(f"Test_id: {self.test_id}")

    def load_cam_mvsnet(self, file, interval_scale=1):
        """read camera txt file"""
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic

    def _sample_pairs_random(self, pairs_path: str, seq_idx: int, num_frames: int = 10):
        scene_dir = osp.dirname(pairs_path)
        image_dir = osp.join(scene_dir, "images")
        img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])

        if not img_files:
            raise RuntimeError(f"No images found in {image_dir}")

        rng = random.Random(self.random_seed if self.random_seed is not None else seq_idx)
        return rng.sample(img_files, min(num_frames, len(img_files)))

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq

        print("Scene ID:", scene_id)

        image_path = osp.join(self.ROOT, scene_id, "images")
        depth_path = osp.join(self.ROOT, scene_id, "depths")
        mask_path = osp.join(self.ROOT, scene_id, "binary_masks")
        cam_path = osp.join(self.ROOT, scene_id, "cams")
        pairs_path = osp.join(self.ROOT, scene_id, "pair.txt")

        if not self.full_video:
            img_idxs = self._sample_pairs_random(pairs_path, seq_id, self.num_frames)
        else:
            img_idxs = sorted([f for f in os.listdir(image_path) if f.lower().endswith(".jpg")])
            img_idxs = img_idxs[:: self.kf_every]

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace(".jpg", ".npy"))
            campath = osp.join(cam_path, im_idx.replace(".jpg", "_cam.txt"))
            maskpath = osp.join(mask_path, im_idx.replace(".jpg", ".png"))

            rgb_image = imread_cv2(impath)
            depthmap = np.load(depthpath)
            depthmap = np.nan_to_num(depthmap, posinf=0., neginf=0., nan=0.)

            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, "r"))
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="dtu",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, "images")))
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = img_idxs[:: min(self.kf_every, len(img_idxs) // 2)]

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0
            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views
