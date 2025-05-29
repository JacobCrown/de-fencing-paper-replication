import os
import time
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
import sys
from typing import Optional, Union

current_dir_rdn_ddp = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn_ddp = os.path.dirname(current_dir_rdn_ddp)
if parent_dir_rdn_ddp not in sys.path:
    sys.path.append(parent_dir_rdn_ddp)

from rdn.models import RDNInpainting
from rdn.datasets import InpaintingDataset
# SPyNetModified might be needed if creating dummy weights in a test scenario, similar to train_rdn.py
# For DDP, it's assumed SPyNetModified is on CPU in the dataset workers.


class DDPConfig:
    # Paths
    vimeo_dir: str = (
        "data_raw/vimeo_septuplet/sequences"  # DDP script used septuplet by default
    )
    defencing_dir: str = "data_raw/De-fencing-master/dataset"
    outputs_dir: str = "./rdn_inpainting_checkpoints_ddp"
    spynet_m_weights_path: Optional[str] = "path/to/your/spynet_m_weights.pth"
    spynet_base_model_name: str = "sintel-final"
    spynet_device: str = "cpu"
    resume_checkpoint: Optional[str] = None

    # RDN Architecture
    num_features: int = 64
    growth_rate: int = 64
    num_blocks: int = 16
    num_layers: int = 8
    k_frames: int = 5
    num_output_channels: int = 3

    # Training Params
    learning_rate: float = 1e-4
    weight_decay: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    num_epochs: int = 1000
    batch_size_per_gpu: int = 8
    start_epoch: int = 0

    # Image/Patch size
    img_width: int = 320
    img_height: int = 192

    # Checkpointing
    save_every_n_epochs: int = 1  # Defaulted to 1 as per user change

    # Data Subset
    subset_fraction: Optional[float] = None
    num_samples: Optional[int] = None

    # DDP specific args
    dist_backend: str = "nccl"

    # Misc
    num_workers: int = 2
    # device is determined by local_rank

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.num_input_channels = 4 + (self.k_frames - 1) * 7

    def to_dict(self):
        """Serializes the config object to a dictionary."""
        d = {}
        for key in dir(self):
            if not key.startswith("__") and not callable(getattr(self, key)):
                d[key] = getattr(self, key)
        return d

    def __str__(self):
        return str(vars(self))


# --- DDP Helper Functions (adapted from spynet/train_flow_ddp.py) ---
def setup_distributed(global_rank, world_size, dist_backend):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # MASTER_ADDR and MASTER_PORT should be set by torchrun/launch utility
    print(
        f"[GlobalRank {global_rank} LocalRank {local_rank}] Initializing process group (backend: {dist_backend})"
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=dist_backend,
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
    )
    print(
        f"[GlobalRank {global_rank} LocalRank {local_rank}] Process group initialized. Device: cuda:{local_rank}"
    )
    dist.barrier()


def cleanup_distributed():
    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[GlobalRank {rank}] Destroying process group...")
        dist.destroy_process_group()
        print(f"[GlobalRank {rank}] Process group destroyed.")


def load_checkpoint_ddp(
    model_wrapper, optimizer, checkpoint_path, device, current_config: DDPConfig
):
    if not os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            print(
                f"WARNING: Checkpoint path does not exist: {checkpoint_path}. Starting from scratch."
            )
        return 0, current_config  # Return 0 epochs and original config

    if dist.get_rank() == 0:
        print(f"Loading checkpoint from: {checkpoint_path} onto device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Update current_config with checkpoint's config for model instantiation consistency
    loaded_config_dict = checkpoint.get("config", {})
    model_hyperparams_to_load = {
        "num_features": current_config.num_features,
        "growth_rate": current_config.growth_rate,
        "num_blocks": current_config.num_blocks,
        "num_layers": current_config.num_layers,
        "k_frames": current_config.k_frames,
        "num_output_channels": current_config.num_output_channels,
        "img_width": current_config.img_width,
        "img_height": current_config.img_height,
        "num_input_channels": current_config.num_input_channels,
    }

    if dist.get_rank() == 0 and loaded_config_dict:
        print("\n--- Model Hyperparameters from Checkpoint (for DDP model init) ---")

    for key in model_hyperparams_to_load.keys():
        if key in loaded_config_dict:
            model_hyperparams_to_load[key] = loaded_config_dict[key]
            if dist.get_rank() == 0:
                print(f"  {key}: {model_hyperparams_to_load[key]} (from checkpoint)")
        elif dist.get_rank() == 0:
            print(
                f"  {key}: {model_hyperparams_to_load[key]} (using current DDPConfig default)"
            )

    # Derive num_input_channels if not in checkpoint config but k_frames is
    if (
        model_hyperparams_to_load.get("num_input_channels") is None
        and model_hyperparams_to_load.get("k_frames") is not None
    ):
        model_hyperparams_to_load["num_input_channels"] = (
            4 + (model_hyperparams_to_load["k_frames"] - 1) * 7
        )
        if dist.get_rank() == 0:
            print(
                f"  num_input_channels: {model_hyperparams_to_load['num_input_channels']} (derived)"
            )

    if dist.get_rank() == 0 and loaded_config_dict:
        print("-----------------------------------------------------------------")

    # Create a new DDPConfig instance reflecting the loaded hyperparameters for the model
    # Other DDP specific params like learning_rate, batch_size are kept from current_config for the new run.
    updated_config_for_model = DDPConfig(
        **{**vars(current_config), **model_hyperparams_to_load}
    )

    # The model passed (model_wrapper) should be instantiated with these updated_config_for_model values *before* loading state_dict
    # This function assumes model_wrapper (the DDP model) already wraps a model of the correct architecture.
    # So, the primary role here is to load state_dict and inform about config mismatches if any.
    # The actual model instantiation using these parameters happens in run_training_ddp.

    raw_model_state_dict = checkpoint["model_state_dict"]
    adjusted_state_dict = OrderedDict()
    for k_ckpt, v_ckpt in raw_model_state_dict.items():
        if not k_ckpt.startswith("module."):
            adjusted_k = "module." + k_ckpt
        else:
            adjusted_k = k_ckpt
        adjusted_state_dict[adjusted_k] = v_ckpt

    model_wrapper.load_state_dict(adjusted_state_dict)

    if "optimizer_state_dict" in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    loss = checkpoint.get("loss", float("nan"))
    if dist.get_rank() == 0:
        print(
            f"Checkpoint loaded. Resuming from epoch {start_epoch}. Last loss: {loss:.4f}"
        )

    # Return the config that should be used for the model architecture (from checkpoint or merged)
    return start_epoch, updated_config_for_model


def save_checkpoint_ddp(
    model, optimizer, epoch, loss, config_to_save: DDPConfig, checkpoint_dir
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_name = f"rdn_inpainting_ddp_epoch{epoch + 1}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    model_state_dict = model.module.state_dict()

    save_dict = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
        "config": config_to_save.to_dict(),
    }
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")


def run_training_ddp(global_rank, world_size, initial_config: DDPConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    current_device = torch.device(f"cuda:{local_rank}")

    setup_distributed(global_rank, world_size, initial_config.dist_backend)

    # Initialize config_for_model with initial_config. It will be updated by load_checkpoint_ddp if resuming.
    config_for_model = DDPConfig(**vars(initial_config))

    # Model instantiation will now use config_for_model, which might be updated by checkpoint loading.
    # If resuming, load_checkpoint_ddp will be called first to get the correct config_for_model.
    start_epoch = initial_config.start_epoch

    # Temporarily create a model instance to pass to load_checkpoint_ddp if resuming.
    # This instance will be correctly configured *after* load_checkpoint_ddp returns the right config.
    # This is a bit of a chicken-and-egg, so we might need to instantiate the model *after* loading checkpoint config.

    # Let's adjust: Load checkpoint first (if any) to get the definitive model config.
    # Then instantiate the model with that config.

    if initial_config.resume_checkpoint and os.path.exists(
        initial_config.resume_checkpoint
    ):
        if global_rank == 0:
            print(
                f"Resuming DDP training from checkpoint: {initial_config.resume_checkpoint}"
            )
        # Dummy model for loading config, state_dict will be loaded into the actual model later
        # The model passed to load_checkpoint_ddp needs to have its state_dict loadable from the checkpoint.
        # This means it must be DDP wrapped. So we need to instantiate model, wrap it, then load.
        temp_model_for_load = (
            RDNInpainting(  # Use initial_config for this temporary model
                num_input_channels=initial_config.num_input_channels,
                num_output_channels=initial_config.num_output_channels,
                num_features=initial_config.num_features,
                growth_rate=initial_config.growth_rate,
                num_blocks=initial_config.num_blocks,
                num_layers=initial_config.num_layers,
            ).to(current_device)
        )
        temp_model_ddp_for_load = DDP(
            temp_model_for_load, device_ids=[local_rank], output_device=local_rank
        )

        # Optimizer is not strictly needed here for config loading, but pass None
        _, config_from_checkpoint = load_checkpoint_ddp(
            temp_model_ddp_for_load,
            None,
            initial_config.resume_checkpoint,
            current_device,
            initial_config,
        )
        config_for_model = (
            config_from_checkpoint  # Use the config derived from the checkpoint
        )
        # start_epoch will be set when we load the checkpoint properly with the final model and optimizer
    else:
        if global_rank == 0:
            print(f"Starting new DDP training with config:\n{initial_config}")
        config_for_model = initial_config  # Use initial config if not resuming

    # 1. Model (instantiated with config_for_model)
    if global_rank == 0:
        print(
            f"Initializing RDNInpainting model with DDP config (effective for model arch):\n{config_for_model}"
        )

    model = RDNInpainting(
        num_input_channels=config_for_model.num_input_channels,
        num_output_channels=config_for_model.num_output_channels,
        num_features=config_for_model.num_features,
        growth_rate=config_for_model.growth_rate,
        num_blocks=config_for_model.num_blocks,
        num_layers=config_for_model.num_layers,
    ).to(current_device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=initial_config.learning_rate,  # LR, WD etc. from the new run's config (initial_config)
        weight_decay=initial_config.weight_decay,
        betas=(initial_config.beta1, initial_config.beta2),
        eps=(initial_config.epsilon),
    )
    criterion = nn.L1Loss().to(current_device)

    # Now, load the checkpoint properly into the final model and optimizer
    if initial_config.resume_checkpoint and os.path.exists(
        initial_config.resume_checkpoint
    ):
        start_epoch, _ = (
            load_checkpoint_ddp(  # We already have config_for_model, ignore the returned one here
                model,
                optimizer,
                initial_config.resume_checkpoint,
                current_device,
                initial_config,
            )
        )
    else:
        start_epoch = initial_config.start_epoch

    # 3. Dataset and DataLoader
    if global_rank == 0:
        print("Preparing InpaintingDataset for DDP...")
    if not config_for_model.spynet_m_weights_path or not os.path.exists(
        config_for_model.spynet_m_weights_path
    ):
        if global_rank == 0:
            print(
                f"ERROR: SPyNetModified weights '{config_for_model.spynet_m_weights_path}' not found."
            )
        cleanup_distributed()
        return

    train_dataset = InpaintingDataset(
        config=config_for_model,  # Use config_for_model for dataset (img_size, k_frames etc.)
        spynet_model_path=config_for_model.spynet_m_weights_path,
        spynet_model_name_for_gt_flow_in_spynet_m=config_for_model.spynet_base_model_name,
        is_train=True,
        spynet_device=config_for_model.spynet_device,
    )

    if len(train_dataset) == 0:
        if global_rank == 0:
            print("ERROR: Training dataset is empty.")
        cleanup_distributed()
        return

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=initial_config.batch_size_per_gpu,
        shuffle=False,  # Sampler handles shuffle
        num_workers=initial_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    if (
        hasattr(train_dataset, "spynet_m")
        and train_dataset.spynet_m is not None
        and global_rank == 0
    ):
        print(
            f"SPyNet-m in dataset is on device: {next(train_dataset.spynet_m.parameters()).device} (expected CPU for workers)"
        )

    if global_rank == 0:
        print(f"--- Starting DDP training for {initial_config.num_epochs} epochs ---")
        print(f"Effective batch size: {initial_config.batch_size_per_gpu * world_size}")

    # 4. Training Loop
    for epoch in range(start_epoch, initial_config.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        running_loss = 0.0

        progress_bar = None
        if global_rank == 0:
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch + 1}/{initial_config.num_epochs} (Rank 0)",
            )

        for batch_idx, (f_in_batch, i_k_m_batch, b_k_batch) in enumerate(
            train_dataloader
        ):
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
            if global_rank == 0 and progress_bar:
                progress_bar.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")
                progress_bar.update(1)

        if global_rank == 0 and progress_bar:
            progress_bar.close()

        # Synchronize losses or report per rank. For simplicity, rank 0 reports its average.
        # A more robust way is to gather all losses.
        epoch_loss_gpu = running_loss / len(train_dataloader)
        avg_loss_tensor = torch.tensor(
            [epoch_loss_gpu], dtype=torch.float, device=current_device
        )
        dist.all_reduce(
            avg_loss_tensor, op=dist.ReduceOp.AVG
        )  # Average the loss across all GPUs
        final_epoch_loss = avg_loss_tensor.item()

        epoch_duration = time.time() - epoch_start_time
        if global_rank == 0:
            print(
                f"End of Epoch {epoch + 1}. Avg Loss (all GPUs): {final_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s"
            )
            if (epoch + 1) % initial_config.save_every_n_epochs == 0 or (
                epoch + 1
            ) == initial_config.num_epochs:
                # Save with config_for_model to store the actual model architecture parameters
                save_checkpoint_ddp(
                    model,
                    optimizer,
                    epoch,
                    final_epoch_loss,
                    config_for_model,  # Save the config that defines the model architecture
                    initial_config.outputs_dir,  # Save to the output dir of the current run
                )

        dist.barrier()  # Ensure all processes finished epoch before next one / saving

    if global_rank == 0:
        print("--- DDP Training finished ---")
    cleanup_distributed()


if __name__ == "__main__":
    config = DDPConfig()
    # --- Example DDP run config overrides ---
    # user_ddp_overrides = {
    #     "spynet_m_weights_path": "/path/to/your/actual/spynet_m_weights.pth",
    #     "num_epochs": 200,
    #     "subset_fraction": 0.1, # For testing with 10% of data
    #     "batch_size_per_gpu": 4
    # }
    # config = DDPConfig(**user_ddp_overrides)

    # world_size, global_rank are set by DDP launch utility (e.g., torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and not dist.is_available():
        print("Distributed training requested but torch.distributed is not available.")
        sys.exit(1)

    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        print(
            f"ERROR: DDP training requires CUDA and enough GPUs. Found {torch.cuda.device_count()} GPUs, World size: {world_size}"
        )
        sys.exit(1)

    if (
        not config.spynet_m_weights_path
        or config.spynet_m_weights_path == "path/to/your/spynet_m_weights.pth"
        or not os.path.exists(config.spynet_m_weights_path)
    ):
        if global_rank == 0:
            print(
                f"ERROR: config.spynet_m_weights_path ('{config.spynet_m_weights_path}') is required and was not found or is a placeholder."
            )
            print("Please set it in the DDPConfig or provide it via override.")
        # All ranks should exit if critical config is missing
        if world_size > 1:
            dist.barrier()  # Ensure all see the message before exiting if possible
        sys.exit(1)

    print(
        f"--- Starting DDP main script (GlobalRank {global_rank}, LocalRank {local_rank}, WorldSize {world_size}) ---"
    )
    # Note: No separate basic_test for DDP script, rely on subset_fraction/num_samples for quick runs.
    # For full DDP testing, one would typically launch it via torchrun.

    run_training_ddp(global_rank, world_size, config)

    print(f"--- DDP main script finished (GlobalRank {global_rank}) ---")
