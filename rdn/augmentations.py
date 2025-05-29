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
    perspective_distorter,
):
    # fence_img_pil, fence_mask_pil: PIL images for a single fence structure
    # perspective_distorter: an instance of T.RandomPerspective
    # Returns: list of K pairs: [(F'_1, M'_1_fence), ..., (F'_K, M'_K_fence)], each PIL

    augmented_fences = []

    # 1. Base Augmentations (once on the raw fence before K perspectives)
    if random.random() < 0.5:  # Color jitter
        fence_img_pil = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        )(fence_img_pil)
    if random.random() < 0.3:  # Gaussian blur
        kernel_s = random.choice([3, 5])
        fence_img_pil = TF.gaussian_blur(
            fence_img_pil,
            kernel_size=[kernel_s, kernel_s],
            sigma=[random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)],
        )

    # 2. K Random Perspective Distortions
    for _ in range(k_frames):
        distorted_fence_img = perspective_distorter(fence_img_pil)
        distorted_fence_mask = perspective_distorter(fence_mask_pil)

        distorted_fence_img_resized = TF.resize(
            distorted_fence_img,
            [img_height, img_width],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        distorted_fence_mask_resized = TF.resize(
            distorted_fence_mask,
            [img_height, img_width],
            interpolation=T.InterpolationMode.NEAREST,
        )
        augmented_fences.append(
            (distorted_fence_img_resized, distorted_fence_mask_resized)
        )
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
        perspective_distorter_test = T.RandomPerspective(distortion_scale=0.2, p=1.0)

        augmented_fences = augment_fence_for_burst(
            dummy_fence_pil,
            dummy_mask_pil,
            k_frames_test,
            img_h,
            img_w,
            perspective_distorter_test,
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
