# spynet/augmentations.py
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import numpy as np
import cv2  # For homography and perspective transforms

# --- Parameters ---
IMG_HEIGHT = 192  # Target height for training crops
IMG_WIDTH = 320  # Target width for training crops

# --- Augmentations for Flow Dataset (as per Appendix A.1) ---


# Background Augmentation for Flow (applied to B_i, B_j)
def augment_background_pair(img1_pil, img2_pil):
    """Augments a pair of PIL background images identically."""
    w_orig, h_orig = img1_pil.size

    # 1. Random homography transformation (applied identically to both)
    # We'll use cv2.getPerspectiveTransform and cv2.warpPerspective for precise control.
    img1_np = np.array(img1_pil)
    img2_np = np.array(img2_pil)

    # Define source points (corners of the original image)
    src_points = np.float32(
        [[0, 0], [w_orig - 1, 0], [0, h_orig - 1], [w_orig - 1, h_orig - 1]]
    )

    # Define 4 destination points with random perturbations (e.g., up to 10-15% of width/height)
    # This creates a perspective distortion.
    max_pert_w = int(w_orig * 0.15)
    max_pert_h = int(h_orig * 0.15)

    dst_points = np.float32(
        [
            [
                random.uniform(-max_pert_w, max_pert_w),
                random.uniform(-max_pert_h, max_pert_h),
            ],
            [
                w_orig - 1 + random.uniform(-max_pert_w, max_pert_w),
                random.uniform(-max_pert_h, max_pert_h),
            ],
            [
                random.uniform(-max_pert_w, max_pert_w),
                h_orig - 1 + random.uniform(-max_pert_h, max_pert_h),
            ],
            [
                w_orig - 1 + random.uniform(-max_pert_w, max_pert_w),
                h_orig - 1 + random.uniform(-max_pert_h, max_pert_h),
            ],
        ]
    )
    M_homography = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the same homography to both images
    # Using BORDER_REFLECT_101 to minimize black borders from warping
    img1_np_homog = cv2.warpPerspective(
        img1_np, M_homography, (w_orig, h_orig), borderMode=cv2.BORDER_REFLECT_101
    )
    img2_np_homog = cv2.warpPerspective(
        img2_np, M_homography, (w_orig, h_orig), borderMode=cv2.BORDER_REFLECT_101
    )

    # Convert back to PIL for subsequent torchvision transforms
    img1_transformed_pil = TF.to_pil_image(img1_np_homog)
    img2_transformed_pil = TF.to_pil_image(img2_np_homog)

    # 2. Random cropping of a 320x192 window (or IMG_WIDTH x IMG_HEIGHT)
    # This crop is applied *after* the homography.
    # The paper mentions "center cropping to avoid black borders" THEN "random cropping".
    # If homography + BORDER_REFLECT might still leave some unusable edges, a center crop first could be beneficial.
    # However, for simplicity and to match common practice, a single random crop on the warped image is often done.
    # If black borders are an issue, they should be handled, perhaps by cropping more from the center.

    # Get shared crop parameters
    i, j, th, tw = T.RandomCrop.get_params(
        img1_transformed_pil, output_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img1_cropped = TF.crop(img1_transformed_pil, i, j, th, tw)
    img2_cropped = TF.crop(img2_transformed_pil, i, j, th, tw)

    # 3. Random horizontal flip (applied identically)
    if random.random() > 0.5:
        img1_cropped = TF.hflip(img1_cropped)
        img2_cropped = TF.hflip(img2_cropped)

    return img1_cropped, img2_cropped


# Foreground (Fence) Augmentation for Flow
def augment_fence_structure(fence_img_pil, fence_mask_pil):
    """
    Augments a single fence structure (image and its mask).
    The output fence and mask are resized to (IMG_HEIGHT, IMG_WIDTH).
    """

    # 1. Random downsample (of fence image and its mask) for size/thickness variation.
    if random.random() < 0.5:
        scale_factor = random.uniform(0.5, 1.0)
        if fence_img_pil.width > 0 and fence_img_pil.height > 0:
            new_w = int(fence_img_pil.width * scale_factor)
            new_h = int(fence_img_pil.height * scale_factor)
            if new_w > 0 and new_h > 0:
                fence_img_pil = TF.resize(
                    fence_img_pil,
                    (new_h, new_w),
                    interpolation=T.InterpolationMode.BILINEAR,
                )
                fence_mask_pil = TF.resize(
                    fence_mask_pil,
                    (new_h, new_w),
                    interpolation=T.InterpolationMode.NEAREST,
                )

    # 2. Random "outer" window crop (focus on a subregion of the original fence image)
    if fence_img_pil.width > 0 and fence_img_pil.height > 0 and random.random() < 0.7:
        crop_scale_w = random.uniform(0.7, 1.0)
        crop_scale_h = random.uniform(0.7, 1.0)
        crop_w = int(fence_img_pil.width * crop_scale_w)
        crop_h = int(fence_img_pil.height * crop_scale_h)
        if (
            crop_w > 0
            and crop_h > 0
            and fence_img_pil.width >= crop_w
            and fence_img_pil.height >= crop_h
        ):
            i, j, th, tw = T.RandomCrop.get_params(
                fence_img_pil, output_size=(crop_h, crop_w)
            )
            fence_img_pil = TF.crop(fence_img_pil, i, j, th, tw)
            fence_mask_pil = TF.crop(fence_mask_pil, i, j, th, tw)

    # 3. Color jitter (on fence image only)
    if random.random() < 0.7:
        fence_img_pil = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        )(fence_img_pil)

    # 4. Random perspective distortion (single distortion for the fence structure)
    # This is different from creating a K-frame *burst* of distorted fences.
    # This step is to add variability to the *single* fence structure used for overlay.
    if random.random() < 0.3:
        if (
            fence_img_pil.width > 10 and fence_img_pil.height > 10
        ):  # Ensure reasonable size
            fence_np = np.array(fence_img_pil)
            mask_np = np.array(fence_mask_pil)
            w_f, h_f = fence_img_pil.size

            src_pts_f = np.float32(
                [[0, 0], [w_f - 1, 0], [0, h_f - 1], [w_f - 1, h_f - 1]]
            )
            max_pert_f = int(min(w_f, h_f) * 0.15)  # Max perturbation for perspective
            dst_pts_f = np.float32(
                [
                    [
                        random.uniform(-max_pert_f, max_pert_f),
                        random.uniform(-max_pert_f, max_pert_f),
                    ],
                    [
                        w_f - 1 + random.uniform(-max_pert_f, max_pert_f),
                        random.uniform(-max_pert_f, max_pert_f),
                    ],
                    [
                        random.uniform(-max_pert_f, max_pert_f),
                        h_f - 1 + random.uniform(-max_pert_f, max_pert_f),
                    ],
                    [
                        w_f - 1 + random.uniform(-max_pert_f, max_pert_f),
                        h_f - 1 + random.uniform(-max_pert_f, max_pert_f),
                    ],
                ]
            )
            M_persp_f = cv2.getPerspectiveTransform(src_pts_f, dst_pts_f)

            # Pad with black for fence image, 0 for mask if warped outside original bounds
            fence_np_persp = cv2.warpPerspective(
                fence_np,
                M_persp_f,
                (w_f, h_f),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            mask_np_persp = cv2.warpPerspective(
                mask_np,
                M_persp_f,
                (w_f, h_f),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            fence_img_pil = TF.to_pil_image(fence_np_persp)
            fence_mask_pil = TF.to_pil_image(mask_np_persp)

    # 5. Random Gaussian blur (on fence image only)
    if random.random() < 0.3:
        if (
            fence_img_pil.width > 0 and fence_img_pil.height > 0
        ):  # Ensure it's not an empty image
            # kernel_size must be odd and positive
            kernel_s = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 1.5)
            fence_img_pil = TF.gaussian_blur(
                fence_img_pil, kernel_size=kernel_s, sigma=sigma
            )

    # Ensure final fence_img and fence_mask are resized to the target background crop size
    if fence_img_pil.width > 0 and fence_img_pil.height > 0:
        fence_img_pil = TF.resize(
            fence_img_pil,
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=T.InterpolationMode.BILINEAR,
        )
        fence_mask_pil = TF.resize(
            fence_mask_pil,
            (IMG_HEIGHT, IMG_WIDTH),
            interpolation=T.InterpolationMode.NEAREST,
        )
    else:  # Handle cases where augmentations might lead to an empty image (e.g. extreme crop on small image)
        fence_img_pil = TF.to_pil_image(
            np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        )
        fence_mask_pil = TF.to_pil_image(
            np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        )

    return fence_img_pil, fence_mask_pil


# Combined Augmentation for Flow Dataset Item
def pv_augmentation_flow(
    background_img1_pil, background_img2_pil, fence_img_pil, fence_mask_pil
):
    """
    Applies augmentations to background pair and a single fence structure.
    Returns:
        bg1_aug_pil, bg2_aug_pil (augmented backgrounds, PIL Images)
        fence_aug_pil, fence_mask_aug_pil (single augmented fence structure, resized to match bg crop, PIL Images)
    """
    bg1_aug, bg2_aug = augment_background_pair(background_img1_pil, background_img2_pil)
    fence_struct_aug, fence_mask_struct_aug = augment_fence_structure(
        fence_img_pil, fence_mask_pil
    )

    return bg1_aug, bg2_aug, fence_struct_aug, fence_mask_struct_aug


if __name__ == "__main__":
    from PIL import Image

    print("Testing REVISED augmentations...")

    dummy_bg1 = Image.new("RGB", (640, 360), color="blue")
    dummy_bg2 = Image.new("RGB", (640, 360), color="lightblue")
    dummy_fence_img = Image.new("RGB", (300, 300), color="red")
    dummy_fence_mask = Image.new("L", (300, 300), color="white")

    print(f"Target crop size: {IMG_WIDTH}x{IMG_HEIGHT}")

    # Test background augmentation multiple times
    print("\nTesting Background Augmentation:")
    for i in range(3):
        bg1_aug_test, bg2_aug_test = augment_background_pair(dummy_bg1, dummy_bg2)
        print(
            f"Iter {i + 1}: Augmented background 1 size: {bg1_aug_test.size}, Augmented background 2 size: {bg2_aug_test.size}"
        )
        assert bg1_aug_test.size == (IMG_WIDTH, IMG_HEIGHT)
        assert bg2_aug_test.size == (IMG_WIDTH, IMG_HEIGHT)
        # bg1_aug_test.save(f"test_bg1_aug_iter{i}.png")
        # bg2_aug_test.save(f"test_bg2_aug_iter{i}.png")

    # Test fence augmentation multiple times
    print("\nTesting Fence Augmentation:")
    for i in range(3):
        fence_aug_img_test, fence_aug_mask_test = augment_fence_structure(
            dummy_fence_img.copy(), dummy_fence_mask.copy()
        )
        print(
            f"Iter {i + 1}: Augmented fence image size: {fence_aug_img_test.size}, Augmented fence mask size: {fence_aug_mask_test.size}"
        )
        assert fence_aug_img_test.size == (IMG_WIDTH, IMG_HEIGHT)
        assert fence_aug_mask_test.size == (IMG_WIDTH, IMG_HEIGHT)
        # fence_aug_img_test.save(f"test_fence_img_aug_iter{i}.png")
        # fence_aug_mask_test.save(f"test_fence_mask_aug_iter{i}.png")

    print("\nTesting Combined Augmentation:")
    bg1_c, bg2_c, fence_c, mask_c = pv_augmentation_flow(
        dummy_bg1, dummy_bg2, dummy_fence_img, dummy_fence_mask
    )
    print(f"Combined bg1 size: {bg1_c.size}, Combined fence size: {fence_c.size}")
    assert bg1_c.size == (IMG_WIDTH, IMG_HEIGHT)
    assert fence_c.size == (IMG_WIDTH, IMG_HEIGHT)

    print("\nRevised augmentation test finished successfully.")
