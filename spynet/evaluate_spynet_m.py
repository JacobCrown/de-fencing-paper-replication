# spynet/evaluate_spynet_m.py
import os
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time

# Ensure the package can be imported correctly if run as a script
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # spynet/
    project_root = os.path.dirname(current_dir)  # Defencing_paper_replication/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from spynet.spynet_modified import SPyNetModified
from spynet.dataset_flow import (
    FlowDataset,
    IMG_HEIGHT,
    IMG_WIDTH,
)  # Assuming IMG_HEIGHT, IMG_WIDTH are available


class EvalConfig:
    # --- User MUST update these paths ---
    MODEL_PATH = "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"  # UPDATE THIS
    VIMEO_DIR = "data_raw/vimeo_test_clean"  # UPDATE THIS if different
    DEFENCING_DIR = "data_raw/De-fencing-master/dataset"  # UPDATE THIS if different
    # --- End of paths to update ---

    VANILLA_SPYNET_NAME = "sintel-final"  # For GT flow generation in FlowDataset

    BATCH_SIZE = 16
    NUM_SAMPLES_TO_EVALUATE = 100
    NUM_SAMPLES_TO_VISUALIZE = (
        4  # How many samples to include in the output image (must be <= BATCH_SIZE)
    )
    OUTPUT_IMAGE_PATH = "test/evaluation_spynet_m_visualization_first.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0  # 0 for main process, helps with debugging on Windows

    # Default image dimensions (should match training)
    IMG_H = IMG_HEIGHT
    IMG_W = IMG_WIDTH


def calculate_aepe(pred_flow, gt_flow):
    """Calculates Average End-Point Error."""
    # pred_flow, gt_flow: [B, 2, H, W]
    error = torch.norm(
        pred_flow - gt_flow, p=2, dim=1
    )  # Calculate L2 norm along the channel dimension (dim 1)
    return error.mean().item()


def normalize_flow_component_for_viz(flow_comp_batch):
    """Normalizes a batch of single flow components (U or V) for visualization."""
    # flow_comp_batch: [B, 1, H, W]
    normalized_list = []
    for i in range(flow_comp_batch.size(0)):
        comp = flow_comp_batch[i]  # [1, H, W]
        min_val = comp.min()
        max_val = comp.max()
        if (max_val - min_val).abs() > 1e-6:
            normalized = (comp - min_val) / (max_val - min_val)
        else:
            normalized = torch.zeros_like(comp)
        # Repeat to 3 channels to display as RGB image
        normalized_list.append(normalized.repeat(3, 1, 1))
    return torch.stack(normalized_list) if normalized_list else torch.empty(0)


def visualize_evaluation_batch(batch_data, pred_flow, config, filename_suffix=""):
    """Visualizes and saves a batch of evaluation data and predictions."""
    input1_rgbm = batch_data["input1_rgbm"].cpu()  # [B, 4, H, W]
    input2_rgbm = batch_data["input2_rgbm"].cpu()
    gt_flow = batch_data["gt_flow"].cpu()  # [B, 2, H, W]
    pred_flow_cpu = pred_flow.cpu()  # [B, 2, H, W]

    num_to_show = min(config.NUM_SAMPLES_TO_VISUALIZE, input1_rgbm.size(0))
    if num_to_show == 0:
        return

    # Extract components
    input1_rgb = input1_rgbm[:num_to_show, 0:3, :, :]
    input1_mask = input1_rgbm[:num_to_show, 3:4, :, :]
    input2_rgb = input2_rgbm[:num_to_show, 0:3, :, :]
    input2_mask = input2_rgbm[:num_to_show, 3:4, :, :]

    gt_flow_u = gt_flow[:num_to_show, 0:1, :, :]
    gt_flow_v = gt_flow[:num_to_show, 1:2, :, :]
    pred_flow_u = pred_flow_cpu[:num_to_show, 0:1, :, :]
    pred_flow_v = pred_flow_cpu[:num_to_show, 1:2, :, :]

    # Normalize flow components for visualization
    gt_flow_u_viz = normalize_flow_component_for_viz(gt_flow_u)
    gt_flow_v_viz = normalize_flow_component_for_viz(gt_flow_v)
    pred_flow_u_viz = normalize_flow_component_for_viz(pred_flow_u)
    pred_flow_v_viz = normalize_flow_component_for_viz(pred_flow_v)

    display_tensors = []
    for i in range(num_to_show):
        display_tensors.append(input1_rgb[i])
        display_tensors.append(input1_mask[i].repeat(3, 1, 1))
        display_tensors.append(input2_rgb[i])
        display_tensors.append(input2_mask[i].repeat(3, 1, 1))
        if gt_flow_u_viz.numel() > 0:
            display_tensors.append(gt_flow_u_viz[i])
        if gt_flow_v_viz.numel() > 0:
            display_tensors.append(gt_flow_v_viz[i])
        if pred_flow_u_viz.numel() > 0:
            display_tensors.append(pred_flow_u_viz[i])
        if pred_flow_v_viz.numel() > 0:
            display_tensors.append(pred_flow_v_viz[i])

    if not display_tensors:
        return

    images_per_sample = 8  # input1_rgb, input1_mask, input2_rgb, input2_mask, gt_u, gt_v, pred_u, pred_v
    grid = make_grid(
        display_tensors, nrow=images_per_sample, padding=2, normalize=False
    )

    plt.figure(figsize=(images_per_sample * 2.5, num_to_show * 2.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")

    base, ext = os.path.splitext(config.OUTPUT_IMAGE_PATH)
    output_path = f"{base}{filename_suffix}{ext}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close()


def evaluate_model(config):
    print(f"Evaluating SPyNetModified on device: {config.DEVICE}")
    print(f"Loading model from: {config.MODEL_PATH}")

    # 1. Model
    # The SPyNetModified class loads Sintel-final by default if pretrained=True and no specific model_name is given for its own weights.
    # Here, we want to load OUR trained weights, so pretrained for base SPyNet should ideally be False
    # or ensure the model_name for SPyNetModified's constructor is not set to a value that triggers base weight loading if we only want to load a full checkpoint.
    model = SPyNetModified(
        model_name=config.VANILLA_SPYNET_NAME, pretrained=False
    )  # Initialize with a model name, but don't load base weights.

    if not os.path.exists(config.MODEL_PATH):
        print(f"ERROR: Model path does not exist: {config.MODEL_PATH}")
        print("Please update EvalConfig.MODEL_PATH in the script.")
        return

    try:
        # Load the state dict. This should be the weights of your *trained* SPyNetModified.
        checkpoint = torch.load(
            config.MODEL_PATH, map_location=config.DEVICE, weights_only=True
        )
        # Check if the checkpoint is a state_dict itself or nested
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model weights from 'model_state_dict' key in checkpoint.")
        elif (
            isinstance(checkpoint, dict) and "state_dict" in checkpoint
        ):  # Another common pattern
            model.load_state_dict(checkpoint["state_dict"])
            print("Loaded model weights from 'state_dict' key in checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights directly from checkpoint.")

    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the SPyNetModified architecture matches the saved weights.")
        print(
            "If loading official SPyNet weights, ensure model_name is correct and pretrained=True."
        )
        print(
            "If loading your fine-tuned model, ensure the path and checkpoint format are correct."
        )
        return

    model.to(config.DEVICE)
    model.eval()

    # 2. Dataset and DataLoader
    print("Preparing dataset and DataLoader...")
    try:
        eval_dataset = FlowDataset(
            vimeo_clean_test_dir=config.VIMEO_DIR,
            defencing_dir=config.DEFENCING_DIR,
            vanilla_spynet_model_name=config.VANILLA_SPYNET_NAME,
        )
    except Exception as e:
        print(f"CRITICAL ERROR creating dataset: {e}")
        import traceback

        traceback.print_exc()
        print("Please check VIMEO_DIR and DEFENCING_DIR in EvalConfig.")
        return

    if len(eval_dataset) == 0:
        print("CRITICAL ERROR: Evaluation dataset is empty.")
        return

    # Determine number of samples to evaluate
    num_samples = len(eval_dataset)
    if config.NUM_SAMPLES_TO_EVALUATE is not None:
        num_samples = min(config.NUM_SAMPLES_TO_EVALUATE, num_samples)

    # Create a subset of the dataset if NUM_SAMPLES_TO_EVALUATE is set
    if config.NUM_SAMPLES_TO_EVALUATE is not None:
        subset_indices = list(range(num_samples))
        eval_dataset_subset = torch.utils.data.Subset(eval_dataset, subset_indices)
        eval_loader = DataLoader(
            eval_dataset_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,  # Important for consistent visualization if run multiple times
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE == "cuda" else False,
        )
    else:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE == "cuda" else False,
        )

    print(f"Dataset and DataLoader ready. Evaluating on {num_samples} samples.")
    print(f"Number of batches: {len(eval_loader)}")

    # 3. Evaluation Loop
    total_aepe = 0.0
    num_batches_processed = 0
    first_batch_visualized = False

    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            if (
                i * config.BATCH_SIZE >= num_samples
            ):  # Ensure we don't exceed num_samples if it's not perfectly divisible
                break

            input1_rgbm = batch_data["input1_rgbm"].to(config.DEVICE)
            input2_rgbm = batch_data["input2_rgbm"].to(config.DEVICE)
            gt_flow = batch_data["gt_flow"].to(config.DEVICE)

            start_time = time.time()
            pred_flow = model(input1_rgbm, input2_rgbm)
            inference_time = time.time() - start_time

            aepe_batch = calculate_aepe(pred_flow, gt_flow)
            total_aepe += aepe_batch * input1_rgbm.size(
                0
            )  # Weight by batch size for accurate overall mean
            num_batches_processed += 1

            print(
                f"Batch [{i + 1}/{len(eval_loader)}]: AEPE = {aepe_batch:.4f}, Inference Time = {inference_time:.4f}s/batch"
            )

            # Visualize the first processed batch that has enough samples
            if (
                not first_batch_visualized
                and input1_rgbm.size(0) >= config.NUM_SAMPLES_TO_VISUALIZE
            ):
                print(
                    f"Visualizing first batch (or first with >= {config.NUM_SAMPLES_TO_VISUALIZE} samples)..."
                )
                visualize_evaluation_batch(
                    batch_data, pred_flow, config, filename_suffix=f"_batch{i + 1}"
                )
                first_batch_visualized = True

    if num_batches_processed > 0:
        mean_aepe = total_aepe / (
            num_batches_processed * config.BATCH_SIZE
        )  # Careful if last batch is smaller
        # More robust calculation for mean_aepe, considering total samples evaluated:
        evaluated_sample_count = min(
            num_samples, len(eval_loader) * config.BATCH_SIZE
        )  # Actual number of samples processed
        if evaluated_sample_count > 0:
            mean_aepe_overall = total_aepe / evaluated_sample_count
            print(f"\n--- Evaluation Complete ---")
            print(f"Evaluated on {evaluated_sample_count} samples.")
            print(f"Overall Mean AEPE: {mean_aepe_overall:.4f}")
        else:
            print("\n--- Evaluation Complete --- No samples processed.")

    else:
        print("No batches were processed. Check dataset or NUM_SAMPLES_TO_EVALUATE.")

    print(
        f"Visualization saved to directory: {os.path.dirname(config.OUTPUT_IMAGE_PATH)}"
    )


if __name__ == "__main__":
    config = EvalConfig()

    # Basic check for essential paths
    if (
        config.MODEL_PATH == "path/to/your/trained/spynet_modified_model.pth"
        or not os.path.exists(config.MODEL_PATH)
    ):
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! CRITICAL: MODEL_PATH is not set or is invalid.                         !!!"
        )
        print(
            "!!! Please update EvalConfig.MODEL_PATH in the script or use --model_path argument. !!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        sys.exit(1)

    if not os.path.isdir(config.VIMEO_DIR) or not os.path.isdir(config.DEFENCING_DIR):
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! WARNING: VIMEO_DIR or DEFENCING_DIR might be invalid.                 !!!"
        )
        print(
            "!!! Please check EvalConfig paths or use --vimeo_dir/--defencing_dir arguments. !!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        # Not exiting, as FlowDataset will raise a more specific error if paths are truly unusable.

    evaluate_model(config)
