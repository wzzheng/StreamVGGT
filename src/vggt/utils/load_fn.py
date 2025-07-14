# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 224

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calcu late height maintaining aspect ratio, divisible by 14
            # new_height = round(height * (new_width / width) / 14) * 14
            new_height = target_size

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


# """
# Image Pre-processing Utility
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Meta Platforms, Inc. and affiliates.

# * 修改要点 *
# -----------
# 1. 当 ``mode="crop"`` 时：  
#    - 先按 **等比例** 将宽度缩放到 518 px；  
#    - 若缩放后高度 ≥ 392，则中心裁剪到 392；  
#    - 若缩放后高度 < 392，则对称填充至 392（白色 = 1.0）；  
#    - 最终得到 (3, 392, 518)，满足 14×k 对齐。  

# 2. ``mode="pad"`` 仍保持 “最长边 518、另一边填充成正方形” 的行为。

# Author: Meta AI (with later modifications)
# """

# import torch
# from PIL import Image
# from torchvision import transforms as TF


# def load_and_preprocess_images(image_path_list, mode: str = "crop") -> torch.Tensor:
#     """
#     Load a list of images and preprocess them for model input.

#     Parameters
#     ----------
#     image_path_list : list[str]
#         文件路径列表。
#     mode : {"crop", "pad"}, default="crop"
#         * "crop": 生成固定大小 (3, 392, 518)；宽度 518、保持比例，过高→裁剪，不足→填充。
#         * "pad" : 最大边缩放至 518，再将短边对称填充为正方形 (518×518)。

#     Returns
#     -------
#     torch.Tensor
#         形状为 (N, 3, H, W) 的批量张量。  
#         - crop 模式：H=392, W=518  
#         - pad 模式：H=W=518

#     Raises
#     ------
#     ValueError
#         若列表为空或 mode 参数不在 {"crop", "pad"}。

#     Notes
#     -----
#     * 所有透明区域都会被融合到白底后转为 RGB。
#     * 若批次内不同形状，会再进行一次对齐填充（白色）。
#     """
#     if not image_path_list:
#         raise ValueError("At least one image path must be provided.")
#     if mode not in {"crop", "pad"}:
#         raise ValueError("Mode must be either 'crop' or 'pad'.")

#     TARGET_W = 224
#     TARGET_H = 224  # 仅 crop 模式使用
#     to_tensor = TF.ToTensor()

#     images = []
#     shapes  = set()

#     for img_path in image_path_list:
#         # -------- 读取并消除透明度 --------
#         img = Image.open(img_path)
#         if img.mode == "RGBA":
#             bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
#             img = Image.alpha_composite(bg, img)
#         img = img.convert("RGB")

#         orig_w, orig_h = img.size

#         # -------- pad 模式 --------
#         if mode == "pad":
#             # 1) 等比例把最长边缩放到 518
#             if orig_w >= orig_h:
#                 new_w = TARGET_W
#                 new_h = round(orig_h * (new_w / orig_w) / 14) * 14
#             else:
#                 new_h = TARGET_W
#                 new_w = round(orig_w * (new_h / orig_h) / 14) * 14

#             img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
#             img = to_tensor(img)

#             # 2) 对称填充至正方形 518×518
#             pad_h = TARGET_W - new_h
#             pad_w = TARGET_W - new_w
#             if pad_h or pad_w:
#                 pad_top    = pad_h // 2
#                 pad_bottom = pad_h - pad_top
#                 pad_left   = pad_w // 2
#                 pad_right  = pad_w - pad_left
#                 img = torch.nn.functional.pad(
#                     img, (pad_left, pad_right, pad_top, pad_bottom),
#                     mode="constant", value=1.0
#                 )

#         # -------- crop 模式 --------
#         else:  # mode == "crop"
#             # 1) 按宽度缩放到 518，保持比例
#             new_w = TARGET_W
#             new_h = round(orig_h * (new_w / orig_w))
#             # 强制 14×k 对齐，避免下游 patch 不整除
#             new_h = max(14, round(new_h / 14) * 14)

#             img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

#             # 2A) 如果高度 ≥ 392 → 中心裁剪
#             if new_h >= TARGET_H:
#                 top = (new_h - TARGET_H) // 2
#                 img = img.crop((0, top, new_w, top + TARGET_H))
#                 img = to_tensor(img)
#             # 2B) 如果高度 < 392 → 转张量后上下填充
#             else:
#                 img = to_tensor(img)
#                 pad_total = TARGET_H - new_h
#                 pad_top   = pad_total // 2
#                 pad_bottom = pad_total - pad_top
#                 img = torch.nn.functional.pad(
#                     img, (0, 0, pad_top, pad_bottom),
#                     mode="constant", value=1.0
#                 )

#         # -------- 收集形状 --------
#         shapes.add((img.shape[1], img.shape[2]))
#         images.append(img)

#     # -------- 若批次内形状不一，再统一填充 --------
#     if len(shapes) > 1:
#         max_h = max(h for h, _ in shapes)
#         max_w = max(w for _, w in shapes)
#         padded_batch = []
#         for img in images:
#             pad_h = max_h - img.shape[1]
#             pad_w = max_w - img.shape[2]
#             if pad_h or pad_w:
#                 pad_top    = pad_h // 2
#                 pad_bottom = pad_h - pad_top
#                 pad_left   = pad_w // 2
#                 pad_right  = pad_w - pad_left
#                 img = torch.nn.functional.pad(
#                     img, (pad_left, pad_right, pad_top, pad_bottom),
#                     mode="constant", value=1.0
#                 )
#             padded_batch.append(img)
#         images = padded_batch

#     batch = torch.stack(images)  # → (N, 3, H, W)
#     return batch


# # ------------------ Quick test ------------------
# if __name__ == "__main__":
#     # 替换为您自己的图片路径做单元测试
#     paths = ["example1.jpg", "example2.jpg"]
#     out_crop = load_and_preprocess_images(paths, mode="crop")
#     out_pad  = load_and_preprocess_images(paths, mode="pad")
#     print("crop shape:", out_crop.shape)  # → (N, 3, 392, 518)
#     print("pad  shape:", out_pad.shape)   # → (N, 3, 518, 518)
