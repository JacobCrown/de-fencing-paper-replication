# spynet/evaluate_spynet_m.py
import os
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time
import glob
from collections import OrderedDict

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
    CHECKPOINT_DIR = "spynet_checkpoints"


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


def evaluate_single_model(config, model_path_to_evaluate, results_summary):
    """Evaluates a single model specified by model_path_to_evaluate."""
    print(f"\\n--- Evaluating Model: {os.path.basename(model_path_to_evaluate)} ---")
    print(f"Evaluating SPyNetModified on device: {config.DEVICE}")
    print(f"Loading model from: {model_path_to_evaluate}")

    # 1. Model
    model = SPyNetModified(
        model_name=config.VANILLA_SPYNET_NAME, pretrained=False
    )

    if not os.path.exists(model_path_to_evaluate):
        print(f"ERROR: Model path does not exist: {model_path_to_evaluate}")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": "Model file not found"}
        return

    try:
        checkpoint = torch.load(
            model_path_to_evaluate, map_location=config.DEVICE, weights_only=True
        )
        
        # Handle checkpoints saved with DDP (keys might be prefixed with 'module.')
        # or other common patterns like 'model_state_dict' or 'state_dict'
        model_state_dict_to_load = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model_state_dict_to_load = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                model_state_dict_to_load = checkpoint["state_dict"]
            else: # Assume the checkpoint itself is the state_dict
                model_state_dict_to_load = checkpoint
        else: # If checkpoint is not a dict, assume it's the state_dict directly (less common)
            model_state_dict_to_load = checkpoint

        if model_state_dict_to_load:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict_to_load.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(f"Loaded model weights from {os.path.basename(model_path_to_evaluate)}.")
        else:
            print(f"Could not identify model state_dict in {os.path.basename(model_path_to_evaluate)}")
            results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": "Could not load state_dict"}
            return


    except Exception as e:
        print(f"Error loading model weights from {model_path_to_evaluate}: {e}")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": str(e)}
        return

    model.to(config.DEVICE)
    model.eval()

    # 2. Dataset and DataLoader
    # print("Preparing dataset and DataLoader...") # Reduced verbosity for multiple runs
    try:
        eval_dataset = FlowDataset(
            vimeo_clean_test_dir=config.VIMEO_DIR,
            defencing_dir=config.DEFENCING_DIR,
            vanilla_spynet_model_name=config.VANILLA_SPYNET_NAME,
        )
    except Exception as e:
        print(f"CRITICAL ERROR creating dataset: {e}")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": f"Dataset creation error: {e}"}
        return

    if len(eval_dataset) == 0:
        print("CRITICAL ERROR: Evaluation dataset is empty.")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": "Dataset empty"}
        return

    num_samples = len(eval_dataset)
    if config.NUM_SAMPLES_TO_EVALUATE is not None:
        num_samples = min(config.NUM_SAMPLES_TO_EVALUATE, num_samples)

    eval_dataset_subset = torch.utils.data.Subset(eval_dataset, list(range(num_samples)))
    eval_loader = DataLoader(
        eval_dataset_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )
    
    # print(f"Dataset and DataLoader ready. Evaluating on {num_samples} samples.")
    # print(f"Number of batches: {len(eval_loader)}")

    total_aepe = 0.0
    num_samples_processed_count = 0 # To correctly average AEPE if last batch is smaller or NUM_SAMPLES_TO_EVALUATE is used
    first_batch_visualized_this_model = False

    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            if config.NUM_SAMPLES_TO_EVALUATE is not None and i * config.BATCH_SIZE >= config.NUM_SAMPLES_TO_EVALUATE:
                break
            
            input1_rgbm = batch_data["input1_rgbm"].to(config.DEVICE)
            input2_rgbm = batch_data["input2_rgbm"].to(config.DEVICE)
            gt_flow = batch_data["gt_flow"].to(config.DEVICE)

            # start_time = time.time() # Less verbose timing for multiple runs
            pred_flow = model(input1_rgbm, input2_rgbm)
            # inference_time = time.time() - start_time

            aepe_batch = calculate_aepe(pred_flow, gt_flow)
            total_aepe += aepe_batch * input1_rgbm.size(0) 
            num_samples_processed_count += input1_rgbm.size(0)


            # print(f"Batch [{i + 1}/{len(eval_loader)}]: AEPE = {aepe_batch:.4f}") # Reduced verbosity

            if (
                not first_batch_visualized_this_model
                and config.NUM_SAMPLES_TO_VISUALIZE > 0
                and input1_rgbm.size(0) >= config.NUM_SAMPLES_TO_VISUALIZE # Ensure batch is large enough
            ):
                # print(f"Visualizing first batch for model {os.path.basename(model_path_to_evaluate)}...")
                model_name_suffix = "_" + os.path.splitext(os.path.basename(model_path_to_evaluate))[0]
                visualize_evaluation_batch(
                    batch_data, pred_flow, config, filename_suffix=model_name_suffix
                )
                first_batch_visualized_this_model = True
    
    mean_aepe_overall = float('inf')
    if num_samples_processed_count > 0:
        mean_aepe_overall = total_aepe / num_samples_processed_count
        print(f"Model: {os.path.basename(model_path_to_evaluate)} - Evaluated on {num_samples_processed_count} samples. Overall Mean AEPE: {mean_aepe_overall:.4f}")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": mean_aepe_overall, "samples_evaluated": num_samples_processed_count}
    else:
        print(f"Model: {os.path.basename(model_path_to_evaluate)} - No samples processed.")
        results_summary[os.path.basename(model_path_to_evaluate)] = {"aepe": float('inf'), "error": "No samples processed", "samples_evaluated": 0}


def evaluate_all_checkpoints(config):
    """Evaluates all .pth checkpoints in the config.CHECKPOINT_DIR."""
    print(f"Starting evaluation for all checkpoints in: {config.CHECKPOINT_DIR}")
    
    checkpoint_files = sorted(glob.glob(os.path.join(config.CHECKPOINT_DIR, "*.pth")))
    
    if not checkpoint_files:
        print(f"No checkpoint files (.pth) found in {config.CHECKPOINT_DIR}.")
        return

    print(f"Found {len(checkpoint_files)} checkpoints to evaluate.")
    
    all_results = {}

    for model_file in checkpoint_files:
        evaluate_single_model(config, model_file, all_results)
        # Small delay or clear CUDA cache if memory issues occur, though typically not needed per model
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()


    print("\n\n--- Overall Evaluation Summary ---")
    if all_results:
        # Sort results by AEPE (ascending)
        sorted_results = sorted(all_results.items(), key=lambda item: item[1].get('aepe', float('inf')))
        
        for model_name, result in sorted_results:
            if "error" in result:
                print(f"Model: {model_name} - Error: {result['error']}")
            else:
                print(f"Model: {model_name} - Mean AEPE: {result['aepe']:.4f} (on {result['samples_evaluated']} samples)")
        
        best_model = sorted_results[0]
        if "error" not in best_model[1]:
             print(f"\nBest performing model: {best_model[0]} with AEPE: {best_model[1]['aepe']:.4f}")
        else:
            print("\nCould not determine best model due to errors during evaluation.")

    else:
        print("No results to summarize.")
    
    if config.NUM_SAMPLES_TO_VISUALIZE > 0:
        print(f"Visualizations (if generated) saved to directory: {os.path.dirname(config.OUTPUT_IMAGE_PATH)}")


if __name__ == "__main__":
    config = EvalConfig()
    
    # Adjustments for multi-model evaluation
    config.BATCH_SIZE = 1 # For safety with num_samples_to_evaluate and partial batches
    config.NUM_SAMPLES_TO_EVALUATE = 20 # Reduce for faster testing of multiple models
    config.NUM_SAMPLES_TO_VISUALIZE = 1 # Reduce to 1 or 0 to speed up / save disk space

    # Basic check for essential paths (CHECKPOINT_DIR will be checked by glob)
    if not os.path.isdir(config.VIMEO_DIR) or not os.path.isdir(config.DEFENCING_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: VIMEO_DIR or DEFENCING_DIR might be invalid.                 !!!")
        print("!!! Please check EvalConfig paths.                                       !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Not exiting, FlowDataset will raise a more specific error.

    evaluate_all_checkpoints(config)
