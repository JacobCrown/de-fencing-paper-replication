import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# import matplotlib.pyplot as plt # Matplotlib currently unused for saving images directly
import numpy as np
from PIL import Image
import sys
from typing import Optional, List

# Try to import OpenCV, but make it optional
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    # print("WARNING: OpenCV (cv2) not found. Flow visualization will be basic (U/V channels).")
    # print("To get color-coded flow images, please install OpenCV: pip install opencv-python")

# Setup sys.path for sibling imports
current_dir_rdn_viz = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn_viz = os.path.dirname(current_dir_rdn_viz)
if parent_dir_rdn_viz not in sys.path:
    sys.path.append(parent_dir_rdn_viz)

from rdn.datasets import InpaintingDataset  # Assuming this can be used
from rdn.train_rdn import TrainConfig  # Use TrainConfig for dataset parameters


# Helper function to convert flow to RGB image
def flow_to_rgb(flow_uv: torch.Tensor) -> torch.Tensor:
    """
    Visualizes optical flow (2 channels) as an RGB image.
    Args:
        flow_uv (torch.Tensor): Flow tensor of shape (2, H, W)
    Returns:
        torch.Tensor: RGB image tensor of shape (3, H, W)
    """
    if not OPENCV_AVAILABLE:
        # Fallback: show U and V as grayscale normalized, then combine into RGB
        u_comp = flow_uv[0, :, :]
        v_comp = flow_uv[1, :, :]

        u_norm = (u_comp - u_comp.min()) / (u_comp.max() - u_comp.min() + 1e-6)
        v_norm = (v_comp - v_comp.min()) / (v_comp.max() - v_comp.min() + 1e-6)
        # Create a 3-channel image: (U_norm, V_norm, zeros)
        return torch.stack([u_norm, v_norm, torch.zeros_like(u_norm)], dim=0)

    if flow_uv.dim() != 3 or flow_uv.shape[0] != 2:
        raise ValueError(
            f"Input flow_uv must be a 2-channel tensor (2, H, W). Got {flow_uv.shape}"
        )

    u = flow_uv[0, :, :].cpu().numpy()  # Ensure numpy array for cv2
    v = flow_uv[1, :, :].cpu().numpy()  # Ensure numpy array for cv2

    # Create HSV image
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255  # Max Value

    magnitude, angle_rad = cv2.cartToPolar(
        u, v, angleInDegrees=False
    )  # angle in radians

    # Calculate hue: angle_rad is in radians, map to 0-179 for OpenCV HSV hue
    hue_conversion_factor = np.float32(180.0 / (np.pi * 2.0))  # Map 0-2pi to 0-180
    # Ensure angle_rad is a numpy array for the multiplication
    angle_rad_np = np.asarray(angle_rad, dtype=np.float32)
    calculated_hue = angle_rad_np * hue_conversion_factor
    hsv[..., 0] = calculated_hue.astype(np.uint8)

    if magnitude.max() > magnitude.min():
        normalized_magnitude = (
            (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255.0
        ).astype(np.uint8)
    else:
        normalized_magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    hsv[..., 1] = normalized_magnitude  # Saturation from magnitude

    # Convert HSV to RGB
    rgb_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0


class VisualizeConfig(TrainConfig):  # Inherit from TrainConfig for shared params
    # Override or add specific visualization params
    num_bursts_to_visualize: int = 2
    output_viz_dir: str = "./rdn_burst_visualizations"
    spynet_m_weights_path: Optional[str] = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth"  # CRITICAL: Must be valid
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize TrainConfig part
        # Update any specific viz params
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Ensure num_samples is at least num_bursts_to_visualize for dataset
        if self.num_samples is None or self.num_samples < self.num_bursts_to_visualize:
            self.num_samples = self.num_bursts_to_visualize
        self.batch_size = 1  # Process one burst at a time for visualization


def visualize_bursts(config: VisualizeConfig):
    print(f"Starting burst visualization with config:\n{config}")
    os.makedirs(config.output_viz_dir, exist_ok=True)
    # current_device = torch.device(config.device) # Not strictly needed for CPU-based viz

    if not config.spynet_m_weights_path or not os.path.exists(
        config.spynet_m_weights_path
    ):
        print(
            f"ERROR: SPyNetModified weights path not provided or invalid: {config.spynet_m_weights_path}"
        )
        print("This is required for InpaintingDataset to generate flows.")
        return

    # 1. Dataset
    print("Preparing InpaintingDataset for visualization...")
    # Use a subset of data for visualization
    dataset_config_for_viz = VisualizeConfig(**vars(config))  # Make a copy
    dataset_config_for_viz.subset_fraction = None  # Ensure we use num_samples
    dataset_config_for_viz.num_samples = config.num_bursts_to_visualize

    viz_dataset = InpaintingDataset(
        config=dataset_config_for_viz,
        spynet_model_path=config.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name,
        is_train=False,  # Use is_train=False for consistency if eval data is different
        spynet_device=config.spynet_device,  # CPU is fine for this
    )

    if len(viz_dataset) == 0:
        print("ERROR: Visualization dataset is empty. Check paths and config.")
        return

    viz_loader = DataLoader(
        dataset=viz_dataset,
        batch_size=1,  # One burst at a time
        shuffle=False,  # Get specific samples if desired, or just first N
        num_workers=0,  # Simpler for debugging
    )

    print(
        f"Dataset loaded. Will visualize {min(config.num_bursts_to_visualize, len(viz_dataset))} bursts."
    )

    # 2. Visualization Loop
    for i, (f_in_tensor, i_k_m_tensor, b_k_tensor) in enumerate(viz_loader):
        if i >= config.num_bursts_to_visualize:
            break

        print(f"\nProcessing burst {i + 1} for visualization...")

        f_in_tensor = f_in_tensor.squeeze(0)  # Remove batch dim
        i_k_m_tensor = i_k_m_tensor.squeeze(0)
        b_k_tensor = b_k_tensor.squeeze(0)

        k_actual = config.k_frames
        num_non_keyframes = k_actual - 1

        s_k_tensor = f_in_tensor[3:4, :, :]  # Single channel mask

        images_to_display: List[torch.Tensor] = []
        # titles: List[str] = [] # Titles via matplotlib is disabled for now

        images_to_display.append(b_k_tensor)
        # titles.append(f"B_k (Clean BG)")
        images_to_display.append(i_k_m_tensor)
        # titles.append(f"I_k^m (Masked BG)")
        images_to_display.append(s_k_tensor.repeat(3, 1, 1))
        # titles.append(f"S_k (Fence Mask)")

        current_channel_idx = 4

        for non_key_idx in range(num_non_keyframes):
            warped_frame_m = f_in_tensor[
                current_channel_idx : current_channel_idx + 3, :, :
            ]
            images_to_display.append(warped_frame_m)
            # titles.append(f"Î_j^m_{non_key_idx + 1}")
            current_channel_idx += 3

            warped_mask = f_in_tensor[
                current_channel_idx : current_channel_idx + 1, :, :
            ]
            images_to_display.append(warped_mask.repeat(3, 1, 1))
            # titles.append(f"Š_j_{non_key_idx + 1}")
            current_channel_idx += 1

            validity_mask = f_in_tensor[
                current_channel_idx : current_channel_idx + 1, :, :
            ]
            images_to_display.append(validity_mask.repeat(3, 1, 1))
            # titles.append(f"V_j_{non_key_idx + 1}")
            current_channel_idx += 1

            flow_tensor_comp = f_in_tensor[
                current_channel_idx : current_channel_idx + 2, :, :
            ]
            if OPENCV_AVAILABLE:
                images_to_display.append(flow_to_rgb(flow_tensor_comp))
                # titles.append(f"Flow RGB_{non_key_idx+1}")
            else:  # Fallback if no OpenCV
                images_to_display.append(
                    flow_tensor_comp[0:1, :, :].repeat(3, 1, 1)
                )  # Flow U
                # titles.append(f"Flow U_{non_key_idx + 1}")
                images_to_display.append(
                    flow_tensor_comp[1:2, :, :].repeat(3, 1, 1)
                )  # Flow V
                # titles.append(f"Flow V_{non_key_idx + 1}")
            current_channel_idx += 2

        num_images_per_burst = len(images_to_display)
        grid_cols = min(num_images_per_burst, 8)

        grid_img = make_grid(
            images_to_display,
            nrow=grid_cols,
            padding=5,
            normalize=False,
            scale_each=False,  # Important for masks and flows not to be individually scaled if not desired
        )

        output_filename = os.path.join(
            config.output_viz_dir, f"burst_visualization_{i + 1}.png"
        )
        save_image(grid_img, output_filename)
        print(f"Saved burst visualization to {output_filename}")

    print(f"\n--- Burst visualization finished. Outputs in {config.output_viz_dir} ---")


if __name__ == "__main__":
    viz_config = VisualizeConfig(
        vimeo_dir="data_raw/vimeo_test_clean/sequences",
        defencing_dir="data_raw/De-fencing-master/dataset",
        spynet_m_weights_path="spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth",
        num_bursts_to_visualize=3,
        img_width=320,
        img_height=192,
        k_frames=5,
        spynet_device="cpu",
        num_samples=3,
    )

    if not OPENCV_AVAILABLE:
        print(
            "\n********************************************************************************"
        )
        print(
            "WARNING: OpenCV (cv2) is not installed. Optical flow will be visualized "
        )
        print(
            "         as separate U and V channels instead of a combined color image."
        )
        print("         To get color-coded flow images, please install OpenCV:")
        print("         pip install opencv-python")
        print(
            "********************************************************************************\n"
        )

    if (
        not viz_config.spynet_m_weights_path
        or viz_config.spynet_m_weights_path == "path/to/your/spynet_m_weights.pth"
        or not os.path.exists(viz_config.spynet_m_weights_path)
    ):
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! CRITICAL: `spynet_m_weights_path` in VisualizeConfig is not set      !!!"
        )
        print(
            "!!! to a valid path or the file does not exist.                        !!!"
        )
        print(
            f"!!! Current path: {viz_config.spynet_m_weights_path}                      !!!"
        )
        print(
            "!!! Please update it in the script (near __main__) for the script to run.!!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
    else:
        visualize_bursts(viz_config)
