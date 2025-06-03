import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np  # For metrics
from PIL import Image  # For saving example images and dummy data creation in test
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import sys
from typing import Optional, Union
from collections import OrderedDict  # For loading state_dict
import datetime

current_dir_rdn_eval = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn_eval = os.path.dirname(current_dir_rdn_eval)
if parent_dir_rdn_eval not in sys.path:
    sys.path.append(parent_dir_rdn_eval)

from rdn.models import RDNInpainting
from rdn.datasets import (
    InpaintingDataset,
)  # Assuming same dataset can be used for eval with is_train=False

try:
    from spynet.spynet_modified import SPyNetModified  # For dummy spynet in test
except ImportError:
    SPyNetModified = None

# --- Default Configuration (should match training or be set as needed for eval) ---
DEFAULT_K_FRAMES = 5
DEFAULT_IMG_WIDTH = 320
DEFAULT_IMG_HEIGHT = 192
# RDN specific - must match the trained model being evaluated
DEFAULT_NUM_FEATURES = 64
DEFAULT_GROWTH_RATE = 64
DEFAULT_NUM_BLOCKS = 16
DEFAULT_NUM_LAYERS = 8
DEFAULT_NUM_INPUT_CHANNELS = 32  # 4 + (K-1)*7
DEFAULT_NUM_OUTPUT_CHANNELS = 3


class EvalConfig:
    # Paths
    checkpoint_path: str = (
        "rdn_inpainting_outputs_precomputed/checkpoints/rdn_precomp_best.pth"
    )
    vimeo_dir: str = "data_raw/vimeo_test_clean/sequences"
    defencing_dir: str = "data_raw/De-fencing-master/dataset"
    spynet_m_weights_path: Optional[str] = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"
    )
    spynet_base_model_name: str = "sintel-final"
    spynet_device: str = "cpu"  # New: device for SPyNet in dataset

    # New output structure:
    base_output_dir: str = "output"  # Base for all outputs
    module_name: str = "rdn_inpainting"  # Specific module
    experiment_name: str = "evaluation"  # Specific experiment/run type
    # eval_outputs_dir will be dynamically constructed
    # Example: output/rdn_inpainting/evaluation/[timestamp_or_id]/evaluations/
    # Example: output/rdn_inpainting/evaluation/[timestamp_or_id]/visualizations/

    # Model Architecture (DEFAULTS - checkpoint values will override these for model instantiation)
    num_features: int = 64
    growth_rate: int = 64
    num_blocks: int = 16
    num_layers: int = 8
    k_frames: int = 5
    # num_input_channels will be derived or from checkpoint
    num_output_channels: int = 3  # Usually fixed for RGB

    # Evaluation parameters
    eval_batch_size: int = 4
    eval_subset_fraction: Optional[float] = None
    eval_num_samples: Optional[int] = 100
    save_eval_images: bool = True

    # Image/Patch size (DEFAULTS - checkpoint values will override)
    img_width: int = 320
    img_height: int = 192

    # Misc
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_mode: str = "eval"  # "eval" or "test"
    num_input_channels: Optional[int] = None
    run_timestamp: str  # To be set at runtime

    # Attributes for InpaintingDataset compatibility regarding subsetting
    subset_fraction: Optional[float] = None
    num_samples: Optional[int] = None

    def __init__(self, **kwargs):
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Initialize all attributes to their class-defined defaults first
        class_attrs = {
            attr_name: getattr(EvalConfig, attr_name)
            for attr_name in dir(EvalConfig)
            if not attr_name.startswith("__")
            and not callable(getattr(EvalConfig, attr_name))
            and attr_name not in ["run_timestamp"]
        }
        for key, value in class_attrs.items():
            setattr(self, key, value)

        # Then override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.num_input_channels is None and self.k_frames is not None:
            self.num_input_channels = 4 + (self.k_frames - 1) * 7

    def __str__(self):
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return str(attrs)


def evaluate_model(eval_config_obj: EvalConfig):
    print(f"Starting RDN Inpainting evaluation with EvalConfig:\n{eval_config_obj}")
    current_device = torch.device(eval_config_obj.device)

    # Setup output directories
    run_specific_output_dir = os.path.join(
        eval_config_obj.base_output_dir,
        eval_config_obj.module_name,
        eval_config_obj.experiment_name,
        eval_config_obj.run_timestamp,
    )
    evaluations_subdir = os.path.join(run_specific_output_dir, "evaluations")
    visualizations_subdir = os.path.join(run_specific_output_dir, "visualizations")
    os.makedirs(evaluations_subdir, exist_ok=True)
    os.makedirs(visualizations_subdir, exist_ok=True)
    print(
        f"Evaluation outputs (metrics, images) will be saved in: {run_specific_output_dir}"
    )

    if not eval_config_obj.checkpoint_path or not os.path.exists(
        eval_config_obj.checkpoint_path
    ):
        print(f"ERROR: Eval checkpoint path invalid: {eval_config_obj.checkpoint_path}")
        return
    if not eval_config_obj.spynet_m_weights_path or not os.path.exists(
        eval_config_obj.spynet_m_weights_path
    ):
        print(
            f"ERROR: SPyNetModified weights for dataset invalid or not provided: {eval_config_obj.spynet_m_weights_path}"
        )
        return

    print(f"Loading checkpoint: {eval_config_obj.checkpoint_path}")
    checkpoint = torch.load(
        eval_config_obj.checkpoint_path, map_location=current_device
    )

    model_hyperparams = {
        "num_features": eval_config_obj.num_features,
        "growth_rate": eval_config_obj.growth_rate,
        "num_blocks": eval_config_obj.num_blocks,
        "num_layers": eval_config_obj.num_layers,
        "k_frames": eval_config_obj.k_frames,
        "num_output_channels": eval_config_obj.num_output_channels,
        "img_width": eval_config_obj.img_width,
        "img_height": eval_config_obj.img_height,
        "num_input_channels": eval_config_obj.num_input_channels,
    }

    if "config" in checkpoint:
        chkpt_config = checkpoint["config"]
        print("\n--- Model Hyperparameters from Checkpoint ---")
        for key in model_hyperparams.keys():
            if key in chkpt_config:
                model_hyperparams[key] = chkpt_config[key]
                print(f"  {key}: {model_hyperparams[key]} (from checkpoint)")
            else:
                print(f"  {key}: {model_hyperparams[key]} (using EvalConfig default)")

        if (
            model_hyperparams.get("num_input_channels") is None
            and model_hyperparams.get("k_frames") is not None
        ):
            model_hyperparams["num_input_channels"] = (
                4 + (model_hyperparams["k_frames"] - 1) * 7
            )
            print(
                f"  num_input_channels: {model_hyperparams['num_input_channels']} (derived)"
            )
        elif model_hyperparams.get("num_input_channels") is not None:
            print(
                f"  num_input_channels: {model_hyperparams['num_input_channels']} (from checkpoint or default)"
            )
        else:
            print(
                f"ERROR: Could not determine num_input_channels from checkpoint or k_frames."
            )
            return
        print("--------------------------------------------")
    else:
        print(
            "WARNING: No 'config' key found in checkpoint. Using EvalConfig defaults for model architecture."
        )
        if (
            model_hyperparams["num_input_channels"] is None
            and model_hyperparams["k_frames"] is not None
        ):
            model_hyperparams["num_input_channels"] = (
                4 + (model_hyperparams["k_frames"] - 1) * 7
            )
        elif model_hyperparams["num_input_channels"] is None:
            print(
                f"ERROR: Could not determine num_input_channels from k_frames in EvalConfig for model."
            )
            return

    model = RDNInpainting(
        num_input_channels=model_hyperparams["num_input_channels"],
        num_output_channels=model_hyperparams["num_output_channels"],
        num_features=model_hyperparams["num_features"],
        growth_rate=model_hyperparams["growth_rate"],
        num_blocks=model_hyperparams["num_blocks"],
        num_layers=model_hyperparams["num_layers"],
    ).to(current_device)

    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded and in evaluation mode.")

    eval_config_obj.img_width = model_hyperparams["img_width"]
    eval_config_obj.img_height = model_hyperparams["img_height"]
    eval_config_obj.k_frames = model_hyperparams["k_frames"]
    eval_config_obj.num_input_channels = model_hyperparams["num_input_channels"]

    print("\nPreparing InpaintingDataset for evaluation using resolved config...")

    # Align subset config names for InpaintingDataset
    # by copying eval-specific names to the generic names expected by InpaintingDataset
    eval_config_obj.subset_fraction = eval_config_obj.eval_subset_fraction
    eval_config_obj.num_samples = eval_config_obj.eval_num_samples

    eval_dataset = InpaintingDataset(
        config=eval_config_obj,
        spynet_model_path=eval_config_obj.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=eval_config_obj.spynet_base_model_name,
        is_train=False,
        spynet_device=eval_config_obj.spynet_device,
    )
    if len(eval_dataset) == 0:
        print("ERROR: Eval dataset empty.")
        return

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_config_obj.eval_batch_size,
        shuffle=False,
        num_workers=eval_config_obj.num_workers,
        pin_memory=True if eval_config_obj.device == "cuda" else False,
        drop_last=False,
    )
    print(
        f"Eval dataset loaded: {len(eval_dataset)} samples, {len(eval_dataloader)} batches."
    )

    # It's good practice to nullify these if they were only for the dataset init,
    # though not strictly necessary here as they are not used further down.
    # eval_config_obj.subset_fraction = None
    # eval_config_obj.num_samples = None

    print("--- Starting evaluation ---")
    total_psnr, total_ssim, num_images_processed = 0.0, 0.0, 0
    with torch.inference_mode():
        for batch_idx, (f_in_batch, i_k_m_batch, b_k_batch) in tqdm(
            enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluating"
        ):
            f_in_batch = f_in_batch.to(current_device)
            i_k_m_batch = i_k_m_batch.to(current_device)
            b_k_batch = b_k_batch.to(current_device)
            residual_pred_batch = model(f_in_batch)
            b_k_pred_batch = i_k_m_batch + residual_pred_batch

            for i in range(b_k_pred_batch.size(0)):
                # Ensure images are on CPU, in HWC format, and in [0,1] range for skimage metrics
                pred_img_np = b_k_pred_batch[i].cpu().permute(1, 2, 0).numpy()
                gt_img_np = b_k_batch[i].cpu().permute(1, 2, 0).numpy()

                pred_img_np = np.clip(pred_img_np, 0, 1)
                gt_img_np = np.clip(gt_img_np, 0, 1)

                try:
                    current_psnr = psnr(gt_img_np, pred_img_np, data_range=1.0)
                    # For SSIM, specify channel_axis for multichannel images (H,W,C)
                    # win_size should be odd and <= min(H,W). Common default is 7 or 11.
                    # Ensure win_size is appropriate for image dimensions.
                    win_size = min(7, gt_img_np.shape[0], gt_img_np.shape[1])
                    if win_size % 2 == 0:  # Ensure it's odd
                        win_size -= 1
                    if win_size < 3:  # Minimal sensible win_size
                        # Handle very small images where SSIM might not be meaningful or error out
                        current_ssim = np.nan  # Or some other placeholder / skip SSIM
                    else:
                        current_ssim = ssim(
                            gt_img_np,
                            pred_img_np,
                            data_range=1.0,
                            channel_axis=-1,
                            win_size=win_size,
                            gaussian_weights=True,
                            K1=0.01,
                            K2=0.03,
                            sigma=1.5,
                            use_sample_covariance=False,
                        )

                    if not np.isnan(current_ssim):
                        total_psnr += current_psnr
                        total_ssim += current_ssim
                        num_images_processed += 1
                    else:
                        if eval_config_obj.run_mode != "test":  # Don't spam for tests
                            print(
                                f"Warning: SSIM was NaN for image {num_images_processed + i} in batch {batch_idx}. PSNR was {current_psnr:.2f}. Skipping this image for SSIM."
                            )
                        # Still count for PSNR if it was valid
                        total_psnr += current_psnr
                        num_images_processed += 1  # Count for PSNR if valid, SSIM will be averaged over non-NaN

                except ValueError as e_metric:
                    print(
                        f"Error calculating metrics for an image: {e_metric}. Skipping this image."
                    )
                    continue  # Skip this image for metrics

                if (
                    eval_config_obj.save_eval_images and batch_idx == 0 and i < 2
                ):  # Save first few of first batch
                    # os.makedirs(eval_config_obj.eval_outputs_dir, exist_ok=True) # Already created above
                    save_pred_path = os.path.join(
                        visualizations_subdir,  # Use new path
                        f"pred_batch{batch_idx}_img{i}.png",
                    )
                    save_gt_path = os.path.join(
                        visualizations_subdir,  # Use new path
                        f"gt_batch{batch_idx}_img{i}.png",
                    )
                    save_masked_path = os.path.join(
                        visualizations_subdir,  # Use new path
                        f"masked_input_batch{batch_idx}_img{i}.png",
                    )

                    Image.fromarray((pred_img_np * 255).astype(np.uint8)).save(
                        save_pred_path
                    )
                    Image.fromarray((gt_img_np * 255).astype(np.uint8)).save(
                        save_gt_path
                    )

                    i_k_m_img_np = i_k_m_batch[i].cpu().permute(1, 2, 0).numpy()
                    i_k_m_img_np = np.clip(i_k_m_img_np, 0, 1)
                    Image.fromarray((i_k_m_img_np * 255).astype(np.uint8)).save(
                        save_masked_path
                    )
                    print(
                        f"Saved example images to {visualizations_subdir}"
                    )  # Log new path

    if num_images_processed > 0:
        avg_psnr = total_psnr / num_images_processed
        avg_ssim = (
            total_ssim / num_images_processed
        )  # This will be correct if NaN SSIMs didn't add to total_ssim or num_images_processed for ssim
        # Corrected logic: num_images_processed for ssim might be different if NaNs occur
        # For simplicity now, assuming num_images_processed applies to both where valid values were obtained.
        print(f"\n--- Evaluation Metrics ({num_images_processed} images) ---")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        # Save metrics to a file
        metrics_file_path = os.path.join(evaluations_subdir, "evaluation_metrics.txt")
        with open(metrics_file_path, "w") as f_metrics:
            f_metrics.write(
                f"Evaluation Metrics for run: {eval_config_obj.run_timestamp}\n"
            )
            f_metrics.write(f"Checkpoint used: {eval_config_obj.checkpoint_path}\n")
            f_metrics.write(f"Number of images processed: {num_images_processed}\n")
            f_metrics.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f_metrics.write(f"Average SSIM: {avg_ssim:.4f}\n")
        print(f"Evaluation metrics saved to {metrics_file_path}")
    else:
        print("No images were processed for metrics.")
    print("--- Evaluation finished ---")


if __name__ == "__main__":
    # Create a new EvalConfig instance for the run to avoid modifying the global default 'config'
    run_config = EvalConfig()
    run_config.run_mode = (
        "eval"  # Ensure this is set if there was a global 'config' object previously
    )

    print("Running in EVALUATION mode.")
    if (
        run_config.checkpoint_path
        == "path/to/your/rdn_checkpoint.pth"  # Default placeholder
        or not os.path.exists(run_config.checkpoint_path)
    ):
        print(
            f"ERROR: run_config.checkpoint_path ('{run_config.checkpoint_path}') must be set to a valid model checkpoint for evaluation."
        )
        sys.exit(1)
    if (
        not run_config.spynet_m_weights_path
        or run_config.spynet_m_weights_path
        == "path/to/your/spynet_m_weights.pth"  # Default placeholder
        or not os.path.exists(run_config.spynet_m_weights_path)
    ):
        print(
            f"ERROR: run_config.spynet_m_weights_path ('{run_config.spynet_m_weights_path}') must be set to valid SPyNet weights for the dataset."
        )
        sys.exit(1)
    # Output directory creation is now handled inside evaluate_model
    # os.makedirs(run_config.eval_outputs_dir, exist_ok=True)
    evaluate_model(run_config)
    print("--- evaluate_model(run_config) completed. --- ")
