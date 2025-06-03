import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image

# Note: IMG_HEIGHT and IMG_WIDTH would typically be passed as arguments or part of a config


def augment_background_burst(bg_frames_pil, img_height, img_width):
    # bg_frames_pil: list of K PIL Images (clean backgrounds)
    # Apply consistent augmentation (crop, flip) across the burst.

    if not bg_frames_pil:
        return []

    # 1. Shared Random Crop to target size
    i, j, th, tw = T.RandomCrop.get_params(
        bg_frames_pil[0], output_size=(img_height, img_width)
    )
    cropped_frames = [TF.crop(frame, i, j, th, tw) for frame in bg_frames_pil]

    # 2. Shared Random Horizontal Flip
    if random.random() > 0.5:
        cropped_frames = [TF.hflip(frame) for frame in cropped_frames]

    # TODO: Add random homography if needed (consistent across burst or evolving)
    # TODO: Add color jitter etc. if specified by paper for background.
    return cropped_frames  # List of K augmented PIL images


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

    # 1. Base Augmentations (once on the raw fence before K transformations)
    if random.random() < 0.5:  # Color jitter
        base_fence_img_pil = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        )(base_fence_img_pil)
    if random.random() < 0.3:  # Gaussian blur
        kernel_s = random.choice([3, 5])
        base_fence_img_pil = TF.gaussian_blur(
            base_fence_img_pil,
            kernel_size=[kernel_s, kernel_s],
            sigma=[random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)],
        )

    # 2. K Random Shifted Crops after initial scaling
    # Scale the base fence to be larger than the target frame size
    scale_factor = 1.50  # e.g., 25% larger
    work_img_h = int(img_height * scale_factor)
    work_img_w = int(img_width * scale_factor)

    # Resize the (potentially augmented) base fence and mask
    # Ensure interpolation methods are appropriate
    f_work_pil = TF.resize(
        base_fence_img_pil,
        [work_img_h, work_img_w],
        interpolation=T.InterpolationMode.BILINEAR,
    )
    m_work_pil = TF.resize(
        base_fence_mask_pil,
        [work_img_h, work_img_w],
        interpolation=T.InterpolationMode.NEAREST,
    )

    # Calculate maximum possible shift based on size difference
    # This is half the difference, as shift can be positive or negative
    max_allowable_shift_x = (work_img_w - img_width) / 2.0
    max_allowable_shift_y = (work_img_h - img_height) / 2.0

    # User-defined shift range (e.g., up to 20 pixels)
    user_max_shift_pixels = 20.0

    # Effective shift limit is the minimum of allowable and user-defined
    actual_max_dx = min(user_max_shift_pixels, max_allowable_shift_x)
    actual_max_dy = min(user_max_shift_pixels, max_allowable_shift_y)

    if actual_max_dx < 0:
        actual_max_dx = 0  # Ensure non-negative if work_img_w is smaller (should not happen with scale_factor > 1)
    if actual_max_dy < 0:
        actual_max_dy = 0

    for _ in range(k_frames):
        # Generate small random 2D shift for the current frame
        delta_x_j = random.uniform(-actual_max_dx, actual_max_dx)
        delta_y_j = random.uniform(-actual_max_dy, actual_max_dy)

        # Calculate top-left corner for cropping from the work_image
        # The crop window is of size (img_height, img_width)
        # Origin for crop is top-left of work_image.
        # A positive delta_x_j means the content shifts left (crop window moves right on work_image)
        # A positive delta_y_j means the content shifts up (crop window moves down on work_image)
        crop_start_x = int(max_allowable_shift_x - delta_x_j)
        crop_start_y = int(max_allowable_shift_y - delta_y_j)

        # Ensure crop coordinates are within bounds of f_work_pil / m_work_pil
        # This should be guaranteed if actual_max_dx/dy are calculated correctly
        # and delta_x_j/delta_y_j are within [-actual_max_d*, actual_max_d*]
        crop_start_x = max(0, min(crop_start_x, work_img_w - img_width))
        crop_start_y = max(0, min(crop_start_y, work_img_h - img_height))

        f_prime_pil_j = TF.crop(
            f_work_pil, crop_start_y, crop_start_x, img_height, img_width
        )
        m_prime_pil_j = TF.crop(
            m_work_pil, crop_start_y, crop_start_x, img_height, img_width
        )

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
        # perspective_distorter_test = T.RandomPerspective(distortion_scale=0.2, p=1.0) # Removed

        augmented_fences = augment_fence_for_burst(
            dummy_fence_pil,
            dummy_mask_pil,
            k_frames_test,
            img_h,
            img_w,
            # perspective_distorter_test, # Argument removed
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
