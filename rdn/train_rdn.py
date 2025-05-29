import os
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import sys
from typing import Optional
from collections import OrderedDict  # For loading state_dict

# Setup sys.path for sibling imports
current_dir_rdn = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn = os.path.dirname(current_dir_rdn)
if parent_dir_rdn not in sys.path:
    sys.path.append(parent_dir_rdn)

try:
    from spynet.spynet_modified import SPyNetModified
except ImportError as e:
    print(f"Could not import SPyNetModified. Ensure spynet module is accessible: {e}")
    SPyNetModified = None

from rdn.models import RDNInpainting
from rdn.datasets import InpaintingDataset


class TrainConfig:
    # Paths
    vimeo_dir: str = (
        "data_raw/vimeo_test_clean/sequences"  # Changed default as per user edit
    )
    defencing_dir: str = "data_raw/De-fencing-master/dataset"
    outputs_dir: str = "./rdn_inpainting_checkpoints"
    spynet_m_weights_path: Optional[str] = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"
    )
    spynet_base_model_name: str = "sintel-final"
    spynet_device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Changed default
    resume_checkpoint: Optional[str] = (
        None  # "rdn_inpainting_checkpoints/rdn_inpainting_epoch10_20250528-174819.pth"
    )

    # RDN Architecture
    num_features: int = 2
    growth_rate: int = 2
    num_blocks: int = 2
    num_layers: int = 2
    k_frames: int = 5  # Used to calculate num_input_channels
    # num_input_channels is derived from k_frames
    num_output_channels: int = 3

    # Training Params
    learning_rate: float = 1e-4
    weight_decay: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_epochs: int = 10
    batch_size: int = 1
    start_epoch: int = 0

    # Image/Patch size
    img_width: int = 320
    img_height: int = 192

    # Checkpointing
    save_every_n_epochs: int = 1

    # Data Subset for faster testing/debugging
    subset_fraction: Optional[float] = 0.05
    num_samples: Optional[int] = None

    # Misc
    num_workers: int = 2  # Changed default to 2, can be tuned (e.g., 0, 2, 4)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_mode: str = "train"  # "train" or "test"
    num_input_channels: Optional[int] = None  # Derived

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.num_input_channels is None:  # Derived property
            self.num_input_channels = 4 + (self.k_frames - 1) * 7

    def to_dict(self):
        """Serializes the config object to a dictionary."""
        d = {}
        for key in dir(self):
            if not key.startswith("__") and not callable(getattr(self, key)):
                d[key] = getattr(self, key)
        return d

    def __str__(self):
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return str(attrs)


def main_train(config: TrainConfig):
    print(f"Starting RDN Inpainting training with config:\n{config}")
    os.makedirs(config.outputs_dir, exist_ok=True)
    current_device = torch.device(config.device)

    # Create a mutable dictionary for model hyperparameters, initialized from current config
    model_hyperparams = {
        "num_input_channels": config.num_input_channels,
        "num_output_channels": config.num_output_channels,
        "num_features": config.num_features,
        "growth_rate": config.growth_rate,
        "num_blocks": config.num_blocks,
        "num_layers": config.num_layers,
        "k_frames": config.k_frames,  # Keep for dataset config consistency
        "img_width": config.img_width,  # Keep for dataset config consistency
        "img_height": config.img_height,  # Keep for dataset config consistency
    }

    optimizer_state_dict = None
    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
        checkpoint = torch.load(config.resume_checkpoint, map_location=current_device)

        loaded_ckpt_config = checkpoint.get("config", {})
        print("\n--- Model Hyperparameters from Checkpoint (for train_rdn.py) ---")
        for key in model_hyperparams.keys():
            if key in loaded_ckpt_config:
                model_hyperparams[key] = loaded_ckpt_config[key]
                print(f"  {key}: {model_hyperparams[key]} (from checkpoint)")
            else:
                print(
                    f"  {key}: {model_hyperparams[key]} (using current TrainConfig default)"
                )

        # Ensure num_input_channels is correctly set after loading
        if (
            model_hyperparams.get("num_input_channels") is None
            and model_hyperparams.get("k_frames") is not None
        ):
            model_hyperparams["num_input_channels"] = (
                4 + (model_hyperparams["k_frames"] - 1) * 7
            )
            print(
                f"  num_input_channels: {model_hyperparams['num_input_channels']} (derived post-load)"
            )
        elif model_hyperparams.get("num_input_channels") is not None:
            print(
                f"  num_input_channels: {model_hyperparams['num_input_channels']} (loaded/default)"
            )
        else:
            print(
                "ERROR: num_input_channels could not be determined for model instantiation after checkpoint load."
            )
            return
        print("---------------------------------------------------------------")

        # Update the main config object's architecture-specific fields for dataset consistency
        config.num_input_channels = model_hyperparams["num_input_channels"]
        config.num_output_channels = model_hyperparams["num_output_channels"]
        config.num_features = model_hyperparams["num_features"]
        config.growth_rate = model_hyperparams["growth_rate"]
        config.num_blocks = model_hyperparams["num_blocks"]
        config.num_layers = model_hyperparams["num_layers"]
        config.k_frames = model_hyperparams["k_frames"]
        config.img_width = model_hyperparams["img_width"]
        config.img_height = model_hyperparams["img_height"]

        model_state_to_load = checkpoint["model_state_dict"]
        if "optimizer_state_dict" in checkpoint:
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            print("Optimizer state will be loaded.")
        config.start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resuming from epoch {config.start_epoch}")
    else:
        model_state_to_load = None
        print("Starting new training run (no valid resume_checkpoint found).")

    # 1. Model (instantiated with resolved hyperparameters)
    print("Initializing RDNInpainting model...")
    model = RDNInpainting(
        num_input_channels=model_hyperparams["num_input_channels"],
        num_output_channels=model_hyperparams["num_output_channels"],
        num_features=model_hyperparams["num_features"],
        growth_rate=model_hyperparams["growth_rate"],
        num_blocks=model_hyperparams["num_blocks"],
        num_layers=model_hyperparams["num_layers"],
    ).to(current_device)

    if model_state_to_load:
        # Adjust for DDP 'module.' prefix if present from a DDP checkpoint
        new_state_dict = OrderedDict()
        for k, v in model_state_to_load.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("Model weights loaded from checkpoint.")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
        print("Optimizer state loaded from checkpoint.")

    criterion = nn.L1Loss()

    # 3. Dataset and DataLoader
    if not config.spynet_m_weights_path or not os.path.exists(
        config.spynet_m_weights_path
    ):
        print(
            f"ERROR: SPyNetModified weights path not provided or invalid: {config.spynet_m_weights_path}"
        )
        if (
            config.run_mode != "test"
        ):  # Allow test to proceed if it handles dummy weights
            return
        elif SPyNetModified is None:
            print(
                "SPyNetModified not available, cannot create dummy for test mode without valid path."
            )
            return
        # Test mode might attempt to create/use a dummy path later if spynet_m_weights_path is placeholder

    print("Preparing InpaintingDataset and DataLoader...")
    train_dataset = InpaintingDataset(
        config=config,  # Pass the whole config object
        spynet_model_path=config.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name,
        is_train=True,
        spynet_device=config.spynet_device,  # Pass the spynet_device config
    )

    if len(train_dataset) == 0:
        print(
            "ERROR: Training dataset is empty. Please check data paths and subset parameters."
        )
        return

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        # num_workers=config.num_workers,
        pin_memory=True
        if config.device == "cuda"
        else False,  # Enable pin_memory if on CUDA
        drop_last=True,
    )

    if hasattr(train_dataset, "spynet_m") and train_dataset.spynet_m is not None:
        print(
            f"SPyNet-m in dataset is on device: {next(train_dataset.spynet_m.parameters()).device}"
        )

    print(f"--- Starting training for {config.num_epochs} epochs ---")
    model.train()

    for epoch in range(config.start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"
        )
        for batch_idx, (f_in_batch, i_k_m_batch, b_k_batch) in enumerate(progress_bar):
            f_in_batch = f_in_batch.to(current_device)
            i_k_m_batch = i_k_m_batch.to(current_device)
            b_k_batch = b_k_batch.to(current_device)

            optimizer.zero_grad()
            residual_pred_batch = model(f_in_batch)
            b_k_pred_batch = i_k_m_batch + residual_pred_batch
            loss = criterion(b_k_pred_batch, b_k_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

        epoch_loss = running_loss / len(train_dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(
            f"End of Epoch {epoch + 1}. Average Loss: {epoch_loss:.4f}. Duration: {epoch_duration:.2f}s"
        )

        if (epoch + 1) % config.save_every_n_epochs == 0 or (
            epoch + 1
        ) == config.num_epochs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_name = f"rdn_inpainting_epoch{epoch + 1}_{timestamp}.pth"
            checkpoint_path = os.path.join(config.outputs_dir, checkpoint_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")
    print("--- Training finished ---")


if __name__ == "__main__":
    # Create a default config
    config = TrainConfig()

    print("Running in TRAIN mode.")
    if (
        not config.spynet_m_weights_path
        or config.spynet_m_weights_path == "path/to/your/spynet_m_weights.pth"
        or not os.path.exists(config.spynet_m_weights_path)
    ):
        print(
            f"ERROR: config.spynet_m_weights_path ('{config.spynet_m_weights_path}') is required for actual training and was not found or is a placeholder."
        )
        print(
            "Please set it in the TrainConfig near the top of the script or provide it via override."
        )
        sys.exit(1)
    main_train(config)
    print("--- main_train(config) completed. --- ")
