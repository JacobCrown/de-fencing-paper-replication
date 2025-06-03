import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image

# Note: IMG_HEIGHT and IMG_WIDTH would typically be passed as arguments or part of a config


def augment_background_burst(bg_frames_pil, img_height, img_width):
    # bg_frames_pil: list of K PIL Images (clean backgrounds)
    # Apply consistent augmentation (homography, crop, flip) across the burst.

    if not bg_frames_pil:
        return []

    # 1. Shared Random Perspective (Homography)
    distortion_scale = 0.2
    perspective_prob = 1.0

    transformed_frames = bg_frames_pil
    if random.random() < perspective_prob:
        first_frame_w, first_frame_h = bg_frames_pil[0].size
        startpoints, endpoints = T.RandomPerspective.get_params(
            first_frame_w, first_frame_h, distortion_scale
        )
        transformed_frames = [
            TF.perspective(
                frame,
                startpoints,
                endpoints,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=[0, 0, 0],  # Corrected: List for RGB fill
            )
            for frame in bg_frames_pil
        ]

    # 2. Shared Random Crop to target size
    i, j, th, tw = T.RandomCrop.get_params(
        transformed_frames[0], output_size=(img_height, img_width)
    )
    cropped_frames = [TF.crop(frame, i, j, th, tw) for frame in transformed_frames]

    # 3. Shared Random Horizontal Flip
    if random.random() > 0.5:
        cropped_frames = [TF.hflip(frame) for frame in cropped_frames]

    return cropped_frames


def augment_fence_for_burst(
    fence_img_pil,
    fence_mask_pil,
    k_frames,
    img_height,
    img_width,
):
    # fence_img_pil, fence_mask_pil: PIL images for a single fence structure
    # Returns: list of K pairs: [(F'_1, M'_1_fence), ..., (F'_K, M'_K_fence)], each PIL

    augmented_fences = []
    base_fence_img_pil = fence_img_pil
    base_fence_mask_pil = fence_mask_pil

    # 1. Random Downsample
    downsample_factor = random.uniform(0.5, 1.0)
    if base_fence_img_pil.width > 0 and base_fence_img_pil.height > 0:
        new_w = int(base_fence_img_pil.width * downsample_factor)
        new_h = int(base_fence_img_pil.height * downsample_factor)
        if new_w > 0 and new_h > 0:
            base_fence_img_pil = TF.resize(
                base_fence_img_pil,
                [new_h, new_w],
                interpolation=T.InterpolationMode.BILINEAR,
            )
            base_fence_mask_pil = TF.resize(
                base_fence_mask_pil,
                [new_h, new_w],
                interpolation=T.InterpolationMode.NEAREST,
            )

    # 2. Random "Outer" Window Crop
    perspective_input_h = int(img_height * 1.5)
    perspective_input_w = int(img_width * 1.5)
    rrc_transform_img = T.RandomResizedCrop(
        size=[perspective_input_h, perspective_input_w],
        scale=(0.6, 1.0),
        ratio=(0.75, 1.33),
        interpolation=T.InterpolationMode.BILINEAR,
    )
    rrc_transform_mask = T.RandomResizedCrop(
        size=[perspective_input_h, perspective_input_w],
        scale=(0.6, 1.0),
        ratio=(0.75, 1.33),
        interpolation=T.InterpolationMode.NEAREST,
    )
    if base_fence_img_pil.width > 0 and base_fence_img_pil.height > 0:
        try:
            base_fence_img_pil = rrc_transform_img(base_fence_img_pil)
            base_fence_mask_pil = rrc_transform_mask(base_fence_mask_pil)
        except (
            ValueError
        ):  # Handles cases where image might be too small for RRC scales
            base_fence_img_pil = TF.resize(
                base_fence_img_pil,
                [perspective_input_h, perspective_input_w],
                interpolation=T.InterpolationMode.BILINEAR,
            )
            base_fence_mask_pil = TF.resize(
                base_fence_mask_pil,
                [perspective_input_h, perspective_input_w],
                interpolation=T.InterpolationMode.NEAREST,
            )

    # 3. Color Jitter
    if random.random() < 0.5:
        base_fence_img_pil = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        )(base_fence_img_pil)

    # 4. Random Blur (Gaussian kernel)
    if random.random() < 0.3:
        kernel_s = random.choice([3, 5])
        sigma_val = random.uniform(0.1, 1.0)
        base_fence_img_pil = TF.gaussian_blur(
            base_fence_img_pil,
            kernel_size=[kernel_s, kernel_s],
            sigma=[sigma_val, sigma_val],
        )

    # Per-frame augmentations for K frames:
    # 5. Random Perspective Distortion (per frame)
    # 6. Center Cropping (per frame)
    perspective_transformer_img = T.RandomPerspective(
        distortion_scale=0.4,
        p=1.0,
        interpolation=T.InterpolationMode.BILINEAR,
        fill=0,  # Corrected: fill=0 for T.RandomPerspective, interpreted as (0,0,0) for RGB PIL
    )
    perspective_transformer_mask = T.RandomPerspective(
        distortion_scale=0.4,
        p=1.0,
        interpolation=T.InterpolationMode.NEAREST,
        fill=0,
    )

    for _ in range(k_frames):
        f_perspective_pil = perspective_transformer_img(base_fence_img_pil)
        m_perspective_pil = perspective_transformer_mask(base_fence_mask_pil)

        f_prime_pil_j = TF.center_crop(f_perspective_pil, [img_height, img_width])
        m_prime_pil_j = TF.center_crop(m_perspective_pil, [img_height, img_width])

        augmented_fences.append((f_prime_pil_j, m_prime_pil_j))

    return augmented_fences


def run_basic_augmentations_test():
    print("--- Running Basic Augmentations Test ---")
    img_h, img_w = 192, 320
    k_frames_test = 3

    # Test augment_background_burst
    print("\nTesting augment_background_burst...")
    try:
        dummy_bg_pils = [
            Image.new("RGB", (img_w + 20, img_h + 20)) for _ in range(k_frames_test)
        ]  # Larger for crop
        augmented_bg_pils = augment_background_burst(dummy_bg_pils, img_h, img_w)
        assert len(augmented_bg_pils) == k_frames_test, (
            "BG augmentation length mismatch"
        )
        for aug_bg in augmented_bg_pils:
            assert aug_bg.size == (img_w, img_h), (
                f"BG augmented size mismatch: {aug_bg.size}"
            )
        print("augment_background_burst successful.")
    except Exception as e:
        print(f"Error in augment_background_burst test: {e}")
        import traceback

        traceback.print_exc()

    # Test augment_fence_for_burst
    print("\nTesting augment_fence_for_burst...")
    try:
        dummy_fence_pil = Image.new(
            "RGB", (img_w - 50, img_h - 50)
        )  # Smaller to test resizing
        dummy_mask_pil = Image.new("L", (img_w - 50, img_h - 50))

        augmented_fences = augment_fence_for_burst(
            dummy_fence_pil,
            dummy_mask_pil,
            k_frames_test,
            img_h,
            img_w,
        )
        assert len(augmented_fences) == k_frames_test, (
            "Fence augmentation length mismatch"
        )
        for f_img, f_mask in augmented_fences:
            assert f_img.size == (img_w, img_h), (
                f"Fence augmented image size mismatch: {f_img.size}"
            )
            assert f_mask.size == (img_w, img_h), (
                f"Fence augmented mask size mismatch: {f_mask.size}"
            )
            assert f_img.mode == "RGB", "Fence image mode error"
            assert f_mask.mode == "L", "Fence mask mode error"
        print("augment_fence_for_burst successful.")
    except Exception as e:
        print(f"Error in augment_fence_for_burst test: {e}")
        import traceback

        traceback.print_exc()

    print("--- Augmentations Test Finished ---")


if __name__ == "__main__":
    run_basic_augmentations_test()
