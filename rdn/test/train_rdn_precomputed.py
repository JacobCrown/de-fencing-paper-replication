import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import json
from typing import Optional, List, Dict, Any
from collections import OrderedDict
import matplotlib.pyplot as plt

# Setup sys.path for sibling imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
rdn_dir = os.path.dirname(current_script_dir)  # rdn/
project_root_dir = os.path.dirname(rdn_dir)  # project root

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
if rdn_dir not in sys.path:
    sys.path.append(rdn_dir)  # To find rdn.models etc.
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)  # To find precomputed_rdn_dataset

from rdn.models import RDNInpainting  # Assuming this is in rdn/models.py
from precomputed_rdn_dataset import PrecomputedRDNDataset  # From rdn/test/


class PrecomputedTrainConfig:
    # Paths
    precomputed_train_data_dir: str = "data_precomputed/rdn_data/train"
    precomputed_val_data_dir: Optional[str] = "data_precomputed/rdn_data/val"
    outputs_dir: str = "./rdn_inpainting_outputs_precomputed"
    checkpoint_dir: str = os.path.join(outputs_dir, "checkpoints")
    resume_checkpoint: Optional[str] = None
    # Path to the generation_config.json used to create the precomputed data
    # This helps ensure model architecture matches the data.
    generation_config_path: Optional[str] = (
        "data_precomputed/rdn_data/generation_config.json"
    )

    # RDN Architecture (will be overridden by generation_config if provided and valid)
    # These are placeholders; ideally, they match what was used for data generation.
    num_features: int = 8
    growth_rate: int = 8
    num_blocks: int = 4
    num_layers: int = 4
    k_frames: int = 5  # Should match data generation
    num_output_channels: int = 3  # Typically 3 for RGB
    num_input_channels: Optional[int] = None  # Derived from k_frames or gen_config
    img_width: int = 320
    img_height: int = 192

    # Training Params
    learning_rate: float = 1e-4
    weight_decay: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_epochs: int = 200
    batch_size: int = 32
    start_epoch: int = 0

    # Checkpointing & Saving
    save_every_n_epochs: int = 10
    save_best_model_only_on_val: bool = True

    # Enhancements (subset fractions are not used as dataset size is fixed by precomputation)
    validation_enabled: bool = True
    lr_scheduler_enabled: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    gradient_clipping_norm: Optional[float] = None

    # Misc
    num_workers: int = (
        0  # For precomputed data, 0 might be fine or even preferred initially
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # This section attempts to load generation_config.
        # It's useful for NEW runs to match data.
        # If resuming from a checkpoint, the checkpoint's config will later override model arch.
        loaded_gen_cfg = self._load_generation_config()
        if loaded_gen_cfg:
            print(
                "Updating PrecomputedTrainConfig with parameters from generation_config.json (for defaults/new runs):"
            )
            self.k_frames = loaded_gen_cfg.get("k_frames", self.k_frames)
            self.img_width = loaded_gen_cfg.get("img_width", self.img_width)
            self.img_height = loaded_gen_cfg.get("img_height", self.img_height)
            # RDN arch params from gen_cfg IF they exist (less common, but supports it)
            self.num_features = loaded_gen_cfg.get("num_features", self.num_features)
            self.growth_rate = loaded_gen_cfg.get("growth_rate", self.growth_rate)
            self.num_blocks = loaded_gen_cfg.get("num_blocks", self.num_blocks)
            self.num_layers = loaded_gen_cfg.get("num_layers", self.num_layers)
            # num_input_channels is critical and should come from gen_config if possible for new runs
            if "num_input_channels" in loaded_gen_cfg:
                self.num_input_channels = loaded_gen_cfg["num_input_channels"]
            print(
                f"  Updated k_frames: {self.k_frames}, img_width: {self.img_width}, img_height: {self.img_height}"
            )
            print(
                f"  Updated RDN arch: num_features={self.num_features}, growth_rate={self.growth_rate}, num_blocks={self.num_blocks}, num_layers={self.num_layers}"
            )
            if "num_input_channels" in loaded_gen_cfg:
                print(
                    f"  Updated num_input_channels (from gen_cfg): {self.num_input_channels}"
                )

        if self.num_input_channels is None:  # Fallback if not in gen_config
            if self.k_frames:
                self.num_input_channels = 4 + (self.k_frames - 1) * 7
                print(
                    f"  Derived num_input_channels (fallback): {self.num_input_channels} from k_frames={self.k_frames}"
                )

            else:
                # This path should ideally not be hit if k_frames is always set.
                raise ValueError(
                    "k_frames is not set, and num_input_channels could not be determined from generation_config.json."
                )
        # print(f"Effective num_input_channels for RDN model (after init): {self.num_input_channels}")

    def _load_generation_config(self):
        if self.generation_config_path and os.path.exists(self.generation_config_path):
            try:
                with open(self.generation_config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load or parse {self.generation_config_path}: {e}"
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


def main_train_precomputed(config: PrecomputedTrainConfig):
    print(
        f"Starting RDN Inpainting training with PRECOMPUTED data using config:\n{config}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    current_device = torch.device(config.device)

    # RDN model hyperparams are now primarily set via __init__ from generation_config
    # or defaults if gen_config is not found/used.
    # WHEN RESUMING, THESE WILL BE OVERRIDDEN BY CHECKPOINT CONFIG.
    model_hyperparams_for_rdn = {
        "num_input_channels": config.num_input_channels,  # Will be updated if resuming
        "num_output_channels": config.num_output_channels,
        "num_features": config.num_features,
        "growth_rate": config.growth_rate,
        "num_blocks": config.num_blocks,
        "num_layers": config.num_layers,
    }

    optimizer_state_dict = None
    scheduler_state_dict = None
    train_loss_history: List[Dict[str, Any]] = []
    val_loss_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")

    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"Resuming training from checkpoint: {config.resume_checkpoint}")
        checkpoint = torch.load(config.resume_checkpoint, map_location=current_device)

        # Config from checkpoint IS THE AUTHORITY for model architecture when resuming.
        loaded_ckpt_config_dict = checkpoint.get("config", {})
        if loaded_ckpt_config_dict:
            print(
                "\nResuming training. RDN model architecture WILL BE LOADED FROM CHECKPOINT, overriding script defaults/generation_config for model setup."
            )

            # Update model_hyperparams_for_rdn directly from checkpoint's config
            # These keys define the RDNInpainting architecture
            arch_keys = [
                "num_input_channels",
                "num_output_channels",
                "num_features",
                "growth_rate",
                "num_blocks",
                "num_layers",
            ]
            for key in arch_keys:
                if key in loaded_ckpt_config_dict:
                    model_hyperparams_for_rdn[key] = loaded_ckpt_config_dict[key]
                    print(
                        f"  Loading from checkpoint for RDN arch: {key} = {model_hyperparams_for_rdn[key]}"
                    )
                elif key in model_hyperparams_for_rdn:
                    print(
                        f"  Using current config for RDN arch (not in ckpt): {key} = {model_hyperparams_for_rdn[key]}"
                    )
                else:
                    print(
                        f"  WARNING: RDN arch key '{key}' not in checkpoint config or current model_hyperparams_for_rdn."
                    )

            # Also update relevant parts of the main `config` object from checkpoint for consistency (e.g. for dataset)
            # K_frames, img_width, img_height are important for dataset compatibility and num_input_channels derivation.
            config.k_frames = loaded_ckpt_config_dict.get("k_frames", config.k_frames)
            config.img_width = loaded_ckpt_config_dict.get(
                "img_width", config.img_width
            )
            config.img_height = loaded_ckpt_config_dict.get(
                "img_height", config.img_height
            )
            # If num_input_channels was in checkpoint, it's already in model_hyperparams_for_rdn.
            # If not, and k_frames changed, model_hyperparams_for_rdn['num_input_channels'] might need re-derivation
            # but it should have been saved. For safety, ensure it's consistent if k_frames was loaded.
            if (
                "k_frames" in loaded_ckpt_config_dict
                and "num_input_channels" not in loaded_ckpt_config_dict
            ):
                model_hyperparams_for_rdn["num_input_channels"] = (
                    4 + (config.k_frames - 1) * 7
                )
                print(
                    f"  Re-derived num_input_channels for RDN arch based on checkpoint k_frames: {model_hyperparams_for_rdn['num_input_channels']}"
                )

            # Update the main config's num_input_channels as well for full consistency
            if "num_input_channels" in model_hyperparams_for_rdn:
                config.num_input_channels = model_hyperparams_for_rdn[
                    "num_input_channels"
                ]

            print("--- End of checkpoint config loading ---")
        else:
            print(
                "WARNING: Checkpoint loaded, but no 'config' dictionary found within. Using script's current config for model architecture."
            )
            # model_hyperparams_for_rdn remains as initialized from current `config`

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
    print(f"RDNInpainting model created with params: {model_hyperparams_for_rdn}")

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

    # --- Datasets and DataLoaders ---
    try:
        train_dataset = PrecomputedRDNDataset(
            data_dir=config.precomputed_train_data_dir
        )
    except FileNotFoundError as e:
        print(f"ERROR: Could not load training data: {e}")
        print(
            f"Please ensure precomputed training data exists at: {config.precomputed_train_data_dir}"
        )
        return

    if len(train_dataset) == 0:
        print(
            f"ERROR: Training dataset at {config.precomputed_train_data_dir} is empty."
        )
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
        if not config.precomputed_val_data_dir:
            print(
                "Warning: Validation enabled but precomputed_val_data_dir not set. Disabling validation."
            )
            config.validation_enabled = False
        else:
            try:
                val_dataset = PrecomputedRDNDataset(
                    data_dir=config.precomputed_val_data_dir
                )
                if len(val_dataset) == 0:
                    print(
                        f"WARNING: Validation dataset at {config.precomputed_val_data_dir} is empty. Disabling validation."
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
            except FileNotFoundError as e:
                print(
                    f"WARNING: Could not load validation data: {e}. Disabling validation."
                )
                config.validation_enabled = False

    if config.validation_enabled and val_dataloader:
        print(
            f"Training dataset: {len(train_dataset)} samples. Validation dataset: {len(val_dataset)} samples."
        )
    else:
        print(f"Training dataset: {len(train_dataset)} samples. Validation disabled.")

    # --- Training Loop ---
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
                    )

            avg_epoch_val_loss = running_val_loss / (len(val_dataloader) + 1e-6)
            val_loss_history.append({"epoch": epoch + 1, "loss": avg_epoch_val_loss})

            if config.lr_scheduler_enabled and scheduler:
                scheduler.step(avg_epoch_val_loss)

            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                best_model_path = os.path.join(
                    config.checkpoint_dir, "rdn_precomp_best.pth"
                )
                # Create a dictionary representing the config to save
                # Start with the current script's config, then overwrite with definitive model arch params
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
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"End of Epoch {epoch + 1}. Train Loss: {avg_epoch_train_loss:.4f}. "
            f"{val_loss_str}Duration: {epoch_duration:.2f}s. LR: {current_lr:.2e}"
        )

        if (epoch + 1) % config.save_every_n_epochs == 0 or (
            epoch + 1
        ) == config.num_epochs:
            chkpt_name = f"rdn_precomp_epoch{epoch + 1}_{timestamp}.pth"
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

    print("--- Precomputed Training finished ---")

    # --- Plotting Loss ---
    if train_loss_history:
        plt.figure(figsize=(12, 7))
        train_epochs = [item["epoch"] for item in train_loss_history]
        train_losses_vals = [item["loss"] for item in train_loss_history]
        plt.plot(train_epochs, train_losses_vals, label="Training Loss", marker="o")

        if val_loss_history and config.validation_enabled:
            val_epochs = [item["epoch"] for item in val_loss_history]
            val_losses_vals = [item["loss"] for item in val_loss_history]
            plt.plot(val_epochs, val_losses_vals, label="Validation Loss", marker="x")

        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.title("RDN Inpainting (Precomputed Data) - Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.minorticks_on()
        plot_save_path = os.path.join(
            config.outputs_dir, f"loss_plot_precomp_{timestamp}.png"
        )
        try:
            plt.savefig(plot_save_path)
            print(f"Loss plot saved to {plot_save_path}")
        except Exception as e_plot:
            print(f"Error saving plot: {e_plot}")
        plt.close()


if __name__ == "__main__":
    run_config = PrecomputedTrainConfig()
    # Example Overrides:
    # run_config.num_epochs = 3
    # run_config.batch_size = 2
    # run_config.precomputed_train_data_dir = "path/to/your/precomputed_train_data"
    # run_config.precomputed_val_data_dir = "path/to/your/precomputed_val_data"
    # run_config.generation_config_path = "path/to_your_data/generation_config.json"

    print("Running RDN training with precomputed data.")
    if not os.path.isdir(run_config.precomputed_train_data_dir):
        print(
            f"ERROR: Precomputed training data directory not found: {run_config.precomputed_train_data_dir}"
        )
        print(
            "Please run the data generation script first or provide the correct path."
        )
        sys.exit(1)

    main_train_precomputed(run_config)
    print("--- main_train_precomputed(run_config) completed. --- ")
