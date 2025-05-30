import os
import time
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import sys
from typing import Optional, List, Dict, Any
from collections import OrderedDict  # For loading state_dict
import matplotlib.pyplot as plt

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
    vimeo_dir: str = "data_raw/vimeo_test_clean/sequences"
    defencing_dir: str = "data_raw/De-fencing-master/dataset"
    outputs_dir: str = "./rdn_inpainting_outputs"  # Changed to a more general name
    checkpoint_dir: str = os.path.join(outputs_dir, "checkpoints")

    spynet_m_weights_path: Optional[str] = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"
    )
    spynet_base_model_name: str = "sintel-final"
    spynet_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_checkpoint: Optional[str] = None

    # RDN Architecture
    num_features: int = 1
    growth_rate: int = 1
    num_blocks: int = 1
    num_layers: int = 1
    k_frames: int = 5
    num_output_channels: int = 3

    # Training Params
    learning_rate: float = 1e-4
    weight_decay: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_epochs: int = 100
    batch_size: int = 4
    start_epoch: int = 0

    # Image/Patch size
    img_width: int = 320
    img_height: int = 192

    # Checkpointing & Saving
    save_every_n_epochs: int = 1
    save_best_model_only_on_val: bool = True

    # Data Subset for faster testing/debugging
    train_subset_fraction: Optional[float] = 0.01
    val_subset_fraction: Optional[float] = 0.01
    subset_fraction: Optional[float] = (
        None  # Generic field for InpaintingDataset compatibility
    )

    # Enhancements
    validation_enabled: bool = True
    lr_scheduler_enabled: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    gradient_clipping_norm: Optional[float] = None

    # Misc
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_input_channels: Optional[int] = None  # Derived

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Derive num_input_channels after all potential k_frames updates
        if self.num_input_channels is None and self.k_frames is not None:
            self.num_input_channels = 4 + (self.k_frames - 1) * 7
        elif self.num_input_channels is None and self.k_frames is None:
            raise ValueError(
                "k_frames must be set to derive num_input_channels if not provided directly."
            )

    def to_dict(self):
        d = {}
        for key in dir(self):
            if not key.startswith("__") and not callable(getattr(self, key)):
                d[key] = getattr(self, key)
        return d

    def __str__(self):
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return str(attrs)


def main_train(config: TrainConfig):
    print(f"Starting RDN Inpainting training with enhanced config:\n{config}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    current_device = torch.device(config.device)

    # Define which keys from config correspond to RDNInpainting constructor arguments
    rdn_constructor_arg_keys = [
        "num_input_channels",
        "num_output_channels",
        "num_features",
        "growth_rate",
        "num_blocks",
        "num_layers",
    ]
    # Prepare model_hyperparams by filtering config, this will be used for model instantiation
    # and potentially updated from checkpoint
    model_hyperparams_for_rdn = {
        k: getattr(config, k) for k in rdn_constructor_arg_keys
    }
    # Ensure all required arch keys are present, using config as the source of truth for NEW runs
    for key in rdn_constructor_arg_keys:
        if key not in model_hyperparams_for_rdn:
            # This case should ideally not happen if rdn_constructor_arg_keys and TrainConfig are aligned
            if hasattr(config, key):
                model_hyperparams_for_rdn[key] = getattr(config, key)
            else:
                raise ValueError(
                    f"RDN architecture key '{key}' missing in TrainConfig and not set."
                )

    optimizer_state_dict = None
    scheduler_state_dict = None
    train_loss_history: List[Dict[str, Any]] = []
    val_loss_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")

    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
        checkpoint = torch.load(config.resume_checkpoint, map_location=current_device)

        loaded_ckpt_config_dict = checkpoint.get("config", {})
        if loaded_ckpt_config_dict:
            print(
                "\nResuming training. RDN model architecture WILL BE LOADED FROM CHECKPOINT, overriding script defaults."
            )
            # Update model_hyperparams_for_rdn from checkpoint if they exist
            for key in (
                model_hyperparams_for_rdn.keys()
            ):  # Iterate over keys expected by RDN constructor
                if key in loaded_ckpt_config_dict:
                    model_hyperparams_for_rdn[key] = loaded_ckpt_config_dict[key]
                    print(
                        f"  Loading from checkpoint for RDN arch: {key} = {model_hyperparams_for_rdn[key]}"
                    )
                else:
                    print(
                        f"  Using current TrainConfig for RDN arch (not in ckpt): {key} = {model_hyperparams_for_rdn[key]}"
                    )

            # Update relevant parts of the main `config` object from checkpoint for dataset compatibility, etc.
            config.k_frames = loaded_ckpt_config_dict.get("k_frames", config.k_frames)
            config.img_width = loaded_ckpt_config_dict.get(
                "img_width", config.img_width
            )
            config.img_height = loaded_ckpt_config_dict.get(
                "img_height", config.img_height
            )
            # Ensure num_input_channels in model_hyperparams_for_rdn is consistent with k_frames from checkpoint
            if "k_frames" in loaded_ckpt_config_dict and model_hyperparams_for_rdn.get(
                "num_input_channels"
            ) != (4 + (config.k_frames - 1) * 7):
                model_hyperparams_for_rdn["num_input_channels"] = (
                    4 + (config.k_frames - 1) * 7
                )
                print(
                    f"  Re-derived num_input_channels for RDN arch based on checkpoint k_frames: {model_hyperparams_for_rdn['num_input_channels']}"
                )

            # Update the main config's num_input_channels as well
            if "num_input_channels" in model_hyperparams_for_rdn:
                config.num_input_channels = model_hyperparams_for_rdn[
                    "num_input_channels"
                ]
            print("--- End of checkpoint config loading ---")
        else:
            print(
                "WARNING: Checkpoint loaded, but no 'config' dictionary found within. Using script's current config for model architecture."
            )

        model_state_to_load = checkpoint["model_state_dict"]
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        config.start_epoch = checkpoint.get("epoch", -1) + 1
        train_loss_history = checkpoint.get("train_loss_history", [])
        val_loss_history = checkpoint.get("val_loss_history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(
            f"Resuming from epoch {config.start_epoch}. Best val loss so far: {best_val_loss:.4f}"
        )
    else:
        model_state_to_load = None
        print("Starting new training run.")

    model = RDNInpainting(**model_hyperparams_for_rdn).to(current_device)

    if model_state_to_load:
        new_state_dict = OrderedDict()
        for k, v_model in model_state_to_load.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v_model
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

    scheduler = None
    if config.lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            patience=config.lr_scheduler_patience,
            factor=config.lr_scheduler_factor,
        )
        if scheduler_state_dict:
            scheduler.load_state_dict(scheduler_state_dict)
            print("LR Scheduler state loaded from checkpoint.")

    criterion = nn.L1Loss().to(current_device)

    if not config.spynet_m_weights_path or not os.path.exists(
        config.spynet_m_weights_path
    ):
        print(
            f"ERROR: SPyNetModified weights path invalid: {config.spynet_m_weights_path}"
        )
        return

    train_dataset_config = TrainConfig(**config.to_dict())
    train_dataset_config.subset_fraction = config.train_subset_fraction
    train_dataset = InpaintingDataset(
        config=train_dataset_config,
        spynet_model_path=config.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name,
        is_train=True,
        spynet_device=config.spynet_device,
    )
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty.")
        return
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=True,
    )

    val_dataloader = None
    if config.validation_enabled:
        val_dataset_config = TrainConfig(**config.to_dict())
        val_dataset_config.subset_fraction = config.val_subset_fraction
        val_dataset = InpaintingDataset(
            config=val_dataset_config,
            spynet_model_path=config.spynet_m_weights_path,
            spynet_model_name_for_gt_flow_in_spynet_m=config.spynet_base_model_name,
            is_train=False,
            spynet_device=config.spynet_device,
        )
        if len(val_dataset) == 0:
            print(
                "WARNING: Validation dataset is empty. Proceeding without validation."
            )
            config.validation_enabled = False
        else:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=(config.device == "cuda"),
                drop_last=False,
            )
            print(
                f"Training dataset: {len(train_dataset)} samples. Validation dataset: {len(val_dataset)} samples."
            )
    else:
        print(f"Training dataset: {len(train_dataset)} samples. Validation disabled.")

    print(
        f"--- Starting training from epoch {config.start_epoch + 1} for {config.num_epochs} total epochs ---"
    )

    for epoch in range(config.start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]"
        )
        for batch_idx, (f_in_batch, i_k_m_batch, b_k_batch) in enumerate(progress_bar):
            f_in_batch = f_in_batch.to(current_device, non_blocking=True)
            i_k_m_batch = i_k_m_batch.to(current_device, non_blocking=True)
            b_k_batch = b_k_batch.to(current_device, non_blocking=True)

            optimizer.zero_grad()
            residual_pred_batch = model(f_in_batch)
            b_k_pred_batch = i_k_m_batch + residual_pred_batch
            loss = criterion(b_k_pred_batch, b_k_batch)
            loss.backward()

            if config.gradient_clipping_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )

            optimizer.step()
            running_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{running_train_loss / (batch_idx + 1):.4f}")

        avg_epoch_train_loss = running_train_loss / len(train_dataloader)
        train_loss_history.append({"epoch": epoch + 1, "loss": avg_epoch_train_loss})

        avg_epoch_val_loss = -1.0

        if config.validation_enabled and val_dataloader:
            model.eval()
            running_val_loss = 0.0
            val_progress_bar = tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [Val]"
            )
            with torch.inference_mode():
                for f_in_val, i_k_m_val, b_k_val in val_progress_bar:
                    f_in_val = f_in_val.to(current_device, non_blocking=True)
                    i_k_m_val = i_k_m_val.to(current_device, non_blocking=True)
                    b_k_val = b_k_val.to(current_device, non_blocking=True)

                    residual_pred_val = model(f_in_val)
                    b_k_pred_val = i_k_m_val + residual_pred_val
                    val_loss = criterion(b_k_pred_val, b_k_val)
                    running_val_loss += val_loss.item()
                    val_progress_bar.set_postfix(
                        loss=f"{running_val_loss / (len(val_progress_bar) + 1e-6):.4f}"
                    )  # Add epsilon to avoid div by zero if val_dataloader is empty somehow

            avg_epoch_val_loss = running_val_loss / (
                len(val_dataloader) + 1e-6
            )  # Add epsilon here too
            val_loss_history.append({"epoch": epoch + 1, "loss": avg_epoch_val_loss})

            if config.lr_scheduler_enabled and scheduler:
                scheduler.step(avg_epoch_val_loss)

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                best_model_path = os.path.join(
                    config.checkpoint_dir, "rdn_inpainting_best.pth"
                )
                # Create a dictionary representing the config to save
                config_to_save = config.to_dict()
                config_to_save.update(
                    model_hyperparams_for_rdn
                )  # Ensure RDN arch params are from model_hyperparams_for_rdn

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                        if scheduler
                        else None,
                        "train_loss_history": train_loss_history,
                        "val_loss_history": val_loss_history,
                        "best_val_loss": best_val_loss,
                        "config": config_to_save,  # Save the updated config dict
                    },
                    best_model_path,
                )
                print(
                    f"New best model saved with val_loss: {best_val_loss:.4f} at {best_model_path}"
                )

        epoch_duration = time.time() - epoch_start_time
        val_loss_str = (
            f"Val Loss: {avg_epoch_val_loss:.4f}. "
            if config.validation_enabled and avg_epoch_val_loss != -1.0
            else ""
        )
        print(
            f"End of Epoch {epoch + 1}. Train Loss: {avg_epoch_train_loss:.4f}. "
            f"{val_loss_str}Duration: {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if (epoch + 1) % config.save_every_n_epochs == 0 or (
            epoch + 1
        ) == config.num_epochs:
            chkpt_name = f"rdn_inpainting_epoch{epoch + 1}_{timestamp}.pth"
            checkpoint_path = os.path.join(config.checkpoint_dir, chkpt_name)

            # Create a dictionary representing the config to save
            config_to_save_epoch = config.to_dict()
            config_to_save_epoch.update(
                model_hyperparams_for_rdn
            )  # Ensure RDN arch params are from model_hyperparams_for_rdn

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler
                    else None,
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                    "best_val_loss": best_val_loss,
                    "config": config_to_save_epoch,  # Save the updated config dict
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("--- Training finished ---")

    if train_loss_history:
        train_epochs = [item["epoch"] for item in train_loss_history]
        train_losses = [item["loss"] for item in train_loss_history]
        plt.figure(figsize=(12, 7))
        plt.plot(train_epochs, train_losses, label="Training Loss", marker="o")

        if val_loss_history and config.validation_enabled:
            val_epochs = [item["epoch"] for item in val_loss_history]
            val_losses = [item["loss"] for item in val_loss_history]
            plt.plot(val_epochs, val_losses, label="Validation Loss", marker="x")

        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.title("RDN Inpainting Training & Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.minorticks_on()
        plot_save_path = os.path.join(config.outputs_dir, f"loss_plot_{timestamp}.png")
        try:
            plt.savefig(plot_save_path)
            print(f"Loss plot saved to {plot_save_path}")
        except Exception as e_plot:
            print(f"Error saving plot: {e_plot}")
        plt.close()


if __name__ == "__main__":
    config = TrainConfig()

    # Example: Override config for a quick test run
    # config.num_epochs = 3
    # config.batch_size = 2
    # config.train_subset_fraction = 0.01
    # config.val_subset_fraction = 0.01
    # config.num_features = 4
    # config.growth_rate = 4
    # config.num_blocks = 2
    # config.num_layers = 2
    # config.save_every_n_epochs = 1
    # config.validation_enabled = True
    # config.lr_scheduler_enabled = True

    print("Running in TRAIN mode with enhanced features.")
    if (
        not config.spynet_m_weights_path
        or config.spynet_m_weights_path == "path/to/your/spynet_m_weights.pth"
        or not os.path.exists(config.spynet_m_weights_path)
    ):
        print(
            f"ERROR: config.spynet_m_weights_path ('{config.spynet_m_weights_path}') is required and was not found or is a placeholder."
        )
        sys.exit(1)

    main_train(config)
    print("--- main_train(config) completed. --- ")
