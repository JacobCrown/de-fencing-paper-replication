import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# import matplotlib.pyplot as plt # Matplotlib currently unused for saving images directly
import numpy as np
from PIL import Image
import sys
from typing import Optional, List
import datetime

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
# from rdn.train_rdn import TrainConfig # Removed problematic import


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


class VisualizeConfig:
    # Paths for input data
    vimeo_dir: str = "data_raw/vimeo_test_clean/sequences"
    defencing_dir: str = "data_raw/De-fencing-master/dataset"
    spynet_m_weights_path: Optional[str] = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth"
    )

    # Visualization parameters
    num_bursts_to_visualize: int = 2

    # New output structure
    base_output_dir: str = "output"
    module_name: str = "rdn_inpainting"
    experiment_name: str = "burst_visualization"
    run_timestamp: str  # To be set at runtime

    # Dataset and model parameters (required by InpaintingDataset and the logic here)
    img_width: int = 320
    img_height: int = 192
    k_frames: int = 5
    spynet_base_model_name: str = "sintel-final"
    spynet_device: str = "cpu"
    device: str = "cuda"  # General device, can be cuda if available

    # For InpaintingDataset compatibility
    subset_fraction: Optional[float] = None
    num_samples: Optional[int] = None  # Will be derived from num_bursts_to_visualize
    batch_size: int = 1  # Fixed for visualization

    # Derived RDN input channels, kept for consistency with original structure, though RDN model not used here
    num_input_channels: int

    def __init__(self, **kwargs):
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Set default values for all attributes first
        # This makes all attributes defined before applying kwargs
        default_attrs = {
            attr_name: getattr(self.__class__, attr_name)
            for attr_name in dir(self.__class__)
            if not attr_name.startswith("__")
            and not callable(getattr(self.__class__, attr_name))
            and attr_name
            not in [
                "run_timestamp",
                "num_input_channels",
                "num_samples",
                "batch_size",
            ]  # Already handled or derived
        }
        for key, value in default_attrs.items():
            setattr(self, key, value)

        # Apply kwargs to override defaults
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Derive num_samples
        if self.num_samples is None or self.num_samples < self.num_bursts_to_visualize:
            self.num_samples = self.num_bursts_to_visualize

        # Derive num_input_channels (consistent with InpaintingDataset's expectation for f_in structure)
        self.num_input_channels = 4 + (self.k_frames - 1) * 7

        # Set fixed batch_size for visualization
        self.batch_size = 1

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def __str__(self):
        # More robust string representation
        attrs = self.to_dict()  # Use to_dict for consistency
        try:
            return f"{self.__class__.__name__}({attrs})"
        except:  # Fallback if any attribute causes issues with f-string formatting
            return str(attrs)


def visualize_bursts(config: VisualizeConfig):
    print(f"Starting burst visualization with config:\n{config}")

    # Construct the specific output directory for this visualization run
    output_viz_dir_for_run = os.path.join(
        config.base_output_dir,
        config.module_name,
        config.experiment_name,
        config.run_timestamp,
        "visualizations",  # Explicitly save into a 'visualizations' subfolder
    )
    os.makedirs(output_viz_dir_for_run, exist_ok=True)
    print(f"Burst visualizations will be saved in: {output_viz_dir_for_run}")

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

    # The config object itself is already configured for the dataset
    # No need to create dataset_config_for_viz separately if VisualizeConfig is self-contained

    viz_dataset = InpaintingDataset(
        config=config,  # Pass the main config object
        spynet_model_path=config.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name,
        is_train=False,
        spynet_device=config.spynet_device,
    )

    if len(viz_dataset) == 0:
        print("ERROR: Visualization dataset is empty. Check paths and config.")
        return

    viz_loader = DataLoader(
        dataset=viz_dataset,
        batch_size=config.batch_size,  # Should be 1 as set in VisualizeConfig
        shuffle=False,
        num_workers=0,
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

        # Ensure k_frames from config matches the f_in_tensor structure implicitly
        # The parsing logic below relies on config.k_frames
        expected_channels = 4 + (config.k_frames - 1) * 7
        if f_in_tensor.shape[0] != expected_channels:
            print(f"ERROR: f_in_tensor channel mismatch for burst {i + 1}!")
            print(
                f"  Expected {expected_channels} channels (based on k_frames={config.k_frames}), got {f_in_tensor.shape[0]}."
            )
            print(f"  Skipping this burst visualization.")
            continue

        s_k_tensor = f_in_tensor[3:4, :, :]  # Single channel mask

        images_to_display: List[torch.Tensor] = []
        # titles: List[str] = [] # Titles via matplotlib is disabled for now

        images_to_display.append(b_k_tensor)
        # titles.append(f"B_k (Clean BG)")
        images_to_display.append(i_k_m_tensor)
        # titles.append(f"I_k^m (Masked BG)")
        images_to_display.append(s_k_tensor.repeat(3, 1, 1))
        # titles.append(f"S_k (Fence Mask)")

        current_channel_idx = 4  # Starting index for non-keyframe data in f_in

        num_non_keyframes = config.k_frames - 1
        for non_key_idx in range(num_non_keyframes):
            # Check if there are enough channels left for this non-keyframe
            if current_channel_idx + 7 > f_in_tensor.shape[0]:
                print(
                    f"Warning: Not enough channels in f_in_tensor for non-keyframe {non_key_idx + 1} (k_frames={config.k_frames})."
                )
                print(
                    f"  Current index: {current_channel_idx}, needed up to {current_channel_idx + 7 - 1}, total: {f_in_tensor.shape[0]}"
                )
                break

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
        grid_cols = min(num_images_per_burst, 8)  # Ensure at least 1 column

        if not images_to_display:
            print(f"No images to display for burst {i + 1}. Skipping grid creation.")
            continue

        grid_img = make_grid(
            images_to_display,
            nrow=grid_cols if grid_cols > 0 else 1,  # Ensure nrow > 0
            padding=5,
            normalize=False,  # Values should be in [0,1] mostly, but flow_to_rgb output is. Masks are 0 or 1.
            scale_each=False,
        )

        output_filename = os.path.join(
            output_viz_dir_for_run,
            f"burst_visualization_{i + 1}.png",  # Use new path
        )
        save_image(grid_img, output_filename)
        print(f"Saved burst visualization to {output_filename}")

    print(
        f"\n--- Burst visualization finished. Outputs in {output_viz_dir_for_run} ---"
    )  # Log new path


if __name__ == "__main__":
    viz_config = VisualizeConfig(
        vimeo_dir="data_raw/vimeo_test_clean/sequences",
        defencing_dir="data_raw/De-fencing-master/dataset",
        spynet_m_weights_path="spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth",
        num_bursts_to_visualize=5,
        img_width=320,  # Example: Make sure these match your data/expectations
        img_height=192,  # Example
        k_frames=5,  # Example
        spynet_device="cpu",
        # num_samples is derived, batch_size is fixed to 1
        # spynet_base_model_name will use its default "sintel-final"
        # device will use its default "cpu"
    )

    visualize_bursts(viz_config)
