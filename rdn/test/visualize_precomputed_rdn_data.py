import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import sys
import json
from typing import Optional, List, Dict, Any
import datetime

# Try to import OpenCV, but make it optional
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Setup sys.path for sibling imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
rdn_dir = os.path.dirname(current_script_dir)  # rdn/
project_root_dir = os.path.dirname(rdn_dir)  # project root

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
if rdn_dir not in sys.path:
    sys.path.append(rdn_dir)
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)  # To find precomputed_rdn_dataset

from precomputed_rdn_dataset import PrecomputedRDNDataset


# Helper function to convert flow to RGB image (adapted from rdn/visualize_burst_data.py)
def flow_to_rgb(flow_uv: torch.Tensor) -> torch.Tensor:
    """
    Visualizes optical flow (2 channels) as an RGB image.
    Args:
        flow_uv (torch.Tensor): Flow tensor of shape (2, H, W)
    Returns:
        torch.Tensor: RGB image tensor of shape (3, H, W)
    """
    if not OPENCV_AVAILABLE:
        u_comp = flow_uv[0, :, :]
        v_comp = flow_uv[1, :, :]
        # Normalize U and V components to [0,1] for grayscale visualization
        u_norm = (u_comp - u_comp.min()) / (u_comp.max() - u_comp.min() + 1e-6)
        v_norm = (v_comp - v_comp.min()) / (v_comp.max() - v_comp.min() + 1e-6)
        # Stack into a 3-channel image (R=U, G=V, B=0)
        return torch.stack([u_norm, v_norm, torch.zeros_like(u_norm)], dim=0)

    if flow_uv.dim() != 3 or flow_uv.shape[0] != 2:
        raise ValueError(
            f"Input flow_uv must be a 2-channel tensor (2, H, W). Got {flow_uv.shape}"
        )

    u = flow_uv[0, :, :].cpu().numpy()
    v = flow_uv[1, :, :].cpu().numpy()

    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255

    magnitude, angle_rad = cv2.cartToPolar(u, v, angleInDegrees=False)

    hue_conversion_factor = np.float32(180.0 / (np.pi * 2.0))
    angle_rad_np = np.asarray(angle_rad, dtype=np.float32)
    calculated_hue = angle_rad_np * hue_conversion_factor
    hsv[..., 0] = calculated_hue.astype(np.uint8)

    if magnitude.max() > magnitude.min():  # Avoid division by zero if max == min
        normalized_magnitude = (
            (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255.0
        ).astype(np.uint8)
    else:
        normalized_magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    hsv[..., 1] = normalized_magnitude

    rgb_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0


class VisualizePrecomputedConfig:
    precomputed_data_dir: str = "data_precomputed/rdn_data/val"  # Default to val set
    # Optional: Path to generation_config.json used for data creation.
    # Helps in automatically determining k_frames and num_input_channels.
    generation_config_path: Optional[str] = (
        "data_precomputed/rdn_data/generation_config.json"
    )

    num_samples_to_visualize: int = 5
    # output_viz_dir: str = "./rdn_precomputed_visualizations" # Old path

    # New output structure
    base_output_dir: str = "output"
    module_name: str = "rdn_inpainting"  # Or could be "data_visualization"
    experiment_name: str = "precomputed_data_visualization"
    run_timestamp: str  # To be set at runtime

    # Fallbacks if generation_config.json is not found or doesn't contain these keys
    k_frames_fallback: int = 5
    num_input_channels_fallback: Optional[int] = (
        None  # Will be derived from k_frames_fallback
    )

    # Determined from generation_config or fallback
    k_frames: int
    num_input_channels: int
    img_height: int
    img_width: int

    def __init__(self, **kwargs):
        # Set defaults
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.base_output_dir = "output"
        self.module_name = "rdn_inpainting"
        self.experiment_name = "precomputed_data_visualization"

        self.k_frames = self.k_frames_fallback
        self.num_input_channels = (
            self.num_input_channels_fallback
            if self.num_input_channels_fallback
            else (4 + (self.k_frames_fallback - 1) * 7)
        )
        self.img_height = 192  # Default, will be overridden by gen_config if present
        self.img_width = 320  # Default, will be overridden by gen_config if present

        # Apply kwargs first, so they can override defaults including paths
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Load generation_config.json if path is valid
        gen_cfg_data = self._load_generation_config()
        if gen_cfg_data:
            print(
                f"Loading parameters from generation_config: {self.generation_config_path}"
            )
            self.k_frames = gen_cfg_data.get("k_frames", self.k_frames)
            self.num_input_channels = gen_cfg_data.get(
                "num_input_channels", self.num_input_channels
            )
            # If num_input_channels is missing from gen_cfg but k_frames is there, re-derive
            if "k_frames" in gen_cfg_data and "num_input_channels" not in gen_cfg_data:
                self.num_input_channels = 4 + (self.k_frames - 1) * 7
                print(
                    f"Derived num_input_channels from gen_config's k_frames: {self.num_input_channels}"
                )
            self.img_height = gen_cfg_data.get("img_height", self.img_height)
            self.img_width = gen_cfg_data.get("img_width", self.img_width)
            print(
                f"  Effective k_frames: {self.k_frames}, num_input_channels: {self.num_input_channels}"
            )
            print(
                f"  Effective img_height: {self.img_height}, img_width: {self.img_width}"
            )

        # Final check for num_input_channels if it's still None
        if self.num_input_channels is None:
            if self.k_frames is not None:
                self.num_input_channels = 4 + (self.k_frames - 1) * 7
            else:
                raise ValueError(
                    "k_frames must be set to derive num_input_channels if not provided in generation_config."
                )

    def _load_generation_config(self):
        if self.generation_config_path and os.path.exists(self.generation_config_path):
            try:
                with open(self.generation_config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load or parse {self.generation_config_path}: {e}"
                )
        elif self.generation_config_path:
            print(
                f"Warning: generation_config_path specified but not found: {self.generation_config_path}"
            )
        return None

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)


def visualize_precomputed_samples(config: VisualizePrecomputedConfig):
    print(f"Starting precomputed data visualization with config:\n{config}")

    # Construct the specific output directory for this visualization run
    output_viz_dir_for_run = os.path.join(
        config.base_output_dir,
        config.module_name,
        config.experiment_name,
        config.run_timestamp,
        "visualizations",  # Explicitly save into a 'visualizations' subfolder
    )
    os.makedirs(output_viz_dir_for_run, exist_ok=True)
    print(f"Precomputed data visualizations will be saved in: {output_viz_dir_for_run}")

    if not OPENCV_AVAILABLE:
        print(
            "\nWARNING: OpenCV (cv2) not installed. Optical flow will be shown as normalized U/V channels.\n"
        )

    try:
        dataset = PrecomputedRDNDataset(data_dir=config.precomputed_data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: Could not load precomputed dataset: {e}")
        print(f"Please ensure data exists at: {config.precomputed_data_dir}")
        return

    if len(dataset) == 0:
        print(f"ERROR: Precomputed dataset at {config.precomputed_data_dir} is empty.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(
        f"Dataset loaded. Will visualize {min(config.num_samples_to_visualize, len(dataset))} samples."
    )

    for i, (f_in_tensor_batch, i_k_m_tensor_batch, b_k_tensor_batch) in enumerate(
        dataloader
    ):
        if i >= config.num_samples_to_visualize:
            break

        print(f"\nProcessing sample {i + 1} for visualization...")

        f_in_tensor = f_in_tensor_batch.squeeze(0)  # Remove batch dim (C, H, W)
        i_k_m_tensor = i_k_m_tensor_batch.squeeze(0)  # (C_img, H, W)
        b_k_tensor = b_k_tensor_batch.squeeze(0)  # (C_img, H, W)

        if f_in_tensor.shape[0] != config.num_input_channels:
            print(
                f"ERROR: f_in_tensor channel mismatch! Expected {config.num_input_channels}, got {f_in_tensor.shape[0]}."
            )
            print(
                "Check if generation_config.json matches the data or if k_frames_fallback is correct."
            )
            continue

        if (
            i_k_m_tensor.shape[1:] != (config.img_height, config.img_width)
            or b_k_tensor.shape[1:] != (config.img_height, config.img_width)
            or f_in_tensor.shape[1:] != (config.img_height, config.img_width)
        ):
            print(f"ERROR: Image dimension mismatch for sample {i + 1}.")
            print(f"  Expected HxW: {config.img_height}x{config.img_width}")
            print(
                f"  Got f_in: {f_in_tensor.shape[1:]}, i_k_m: {i_k_m_tensor.shape[1:]}, b_k: {b_k_tensor.shape[1:]}"
            )
            print("  Please check generation_config.json or config image dimensions.")
            continue

        images_to_display: List[torch.Tensor] = []

        # --- Parse f_in_tensor ---
        current_channel_idx = 0

        # I_k^m (from f_in, should be identical to i_k_m_tensor)
        # images_to_display.append(f_in_tensor[current_channel_idx : current_channel_idx+3, :, :]) # Already have i_k_m_tensor
        current_channel_idx += 3

        # S_k (keyframe mask)
        s_k_tensor = f_in_tensor[current_channel_idx : current_channel_idx + 1, :, :]
        images_to_display.append(s_k_tensor.repeat(3, 1, 1))  # Repeat for RGB vis
        current_channel_idx += 1

        # Ground truth and Masked Input for RDN
        images_to_display.insert(0, b_k_tensor)  # GT Clean background
        images_to_display.insert(1, i_k_m_tensor)  # Masked input I_k^m

        num_non_keyframes = config.k_frames - 1

        for nf_idx in range(num_non_keyframes):
            # Î_j^m (warped non-keyframe masked background, 3 channels)
            warped_frame_m = f_in_tensor[
                current_channel_idx : current_channel_idx + 3, :, :
            ]
            images_to_display.append(warped_frame_m)
            current_channel_idx += 3

            # Š_j (warped non-keyframe mask, 1 channel)
            warped_mask_s = f_in_tensor[
                current_channel_idx : current_channel_idx + 1, :, :
            ]
            images_to_display.append(warped_mask_s.repeat(3, 1, 1))
            current_channel_idx += 1

            # V_j (validity mask for warp, 1 channel)
            validity_mask_v = f_in_tensor[
                current_channel_idx : current_channel_idx + 1, :, :
            ]
            images_to_display.append(validity_mask_v.repeat(3, 1, 1))
            current_channel_idx += 1

            # f_kj^m (flow from keyframe to non-keyframe, 2 channels)
            flow_f = f_in_tensor[current_channel_idx : current_channel_idx + 2, :, :]
            images_to_display.append(flow_to_rgb(flow_f))
            current_channel_idx += 2

        # Make grid
        # Layout:
        # Row 1: B_k, I_k^m, S_k
        # Row 2 onwards: For each non-keyframe j: Î_j^m, Š_j, V_j, Flow_kj
        # We have 3 initial images. Then 4 images per non-keyframe.
        # Total images = 3 + num_non_keyframes * 4
        # Let's try a fixed number of columns for better readability, e.g., 4 or 5.
        # If k_frames = 5, num_non_keyframes = 4. Total images = 3 + 4*4 = 19.
        # If k_frames = 3, num_non_keyframes = 2. Total images = 3 + 2*4 = 11.

        grid_cols = 4  # Number of columns in the output image grid
        # For k_frames=5, we have B_k, I_k^m, S_k | Î1,Š1,V1,F1 | Î2,Š2,V2,F2 | ...
        # To make it more structured:
        # B_k, I_k^m, S_k (empty slot or flow placeholder if k_frames = 1?)
        # Î_j1^m, Š_j1, V_j1, Flow_j1
        # Î_j2^m, Š_j2, V_j2, Flow_j2
        # ...
        # The current images_to_display order is: B_k, I_k^m, S_k, Î_1, Š_1, V_1, F_1, Î_2, ...
        # This seems fine for make_grid with nrow.

        try:
            grid_img = make_grid(
                images_to_display,
                nrow=grid_cols,
                padding=5,
                normalize=True,  # Normalize to [0,1] if not already (flows might be out)
                scale_each=True,  # Scale each image in the grid individually for better viz
            )
            output_filename = os.path.join(
                output_viz_dir_for_run,
                f"sample_viz_{i + 1:04d}.png",  # Use new path
            )
            save_image(grid_img, output_filename)
            print(f"Saved visualization to {output_filename}")
        except Exception as e_grid:
            print(f"Error creating or saving image grid for sample {i + 1}: {e_grid}")
            import traceback

            traceback.print_exc()

    print(
        f"\n--- Precomputed data visualization finished. Outputs in {output_viz_dir_for_run} ---"  # Log new path
    )


if __name__ == "__main__":
    # --- Configuration for the visualization ---
    # Ensure these paths are correct for your setup
    vis_config = VisualizePrecomputedConfig(
        precomputed_data_dir="data_precomputed/rdn_data/train",  # Or your train data
        generation_config_path="data_precomputed/rdn_data/generation_config.json",
        num_samples_to_visualize=6,
        # output_viz_dir is now constructed internally based on base_output_dir etc.
        # Example: Override base_output_dir if needed:
        # base_output_dir="my_custom_outputs"
    )

    # Example: If you want to visualize training data and its gen_config is elsewhere:
    # vis_config = VisualizePrecomputedConfig(
    #     precomputed_data_dir="data_precomputed/rdn_data/train",
    #     generation_config_path="data_precomputed/rdn_data/generation_config.json", # Assuming same config for train/val
    #     num_samples_to_visualize=2,
    #     output_viz_dir="./rdn_precomputed_train_viz"
    # )

    visualize_precomputed_samples(vis_config)
