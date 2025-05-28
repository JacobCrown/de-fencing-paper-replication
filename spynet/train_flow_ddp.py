#!/usr/bin/env python
# train_flow_ddp.py - Distributed version of train_flow.py
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import typing
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict

# Imports from our spynet module
from spynet_modified import SPyNetModified
from dataset_flow import FlowDataset

# --- Training Configuration ---
class TrainConfig:
    # Data paths
    VIMEO_CLEAN_TEST_DIR: str = "data_raw/vimeo_test_clean"
    DEFENCING_DIR: str = "data_raw/De-fencing-master/dataset"

    # Model names for vanilla SPyNet (used in FlowDataset for GT)
    VANILLA_SPYNET_MODEL_NAME: str = "sintel-final"

    # Parameters for SPyNetModified
    MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME: str = "sintel-final"
    LOAD_PRETRAINED_FOR_MODIFIED: bool = True

    # Training parameters
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 1000
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 4e-5
    START_EPOCH: int = 0

    # Resume training
    RESUME_TRAINING: bool = True
    RESUME_CHECKPOINT_PATH: typing.Optional[str] = "spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth"

    # Checkpointing
    CHECKPOINT_DIR: str = "./spynet_checkpoints"
    SAVE_EVERY_N_EPOCHS: int = 1

    # Distributed training settings
    DIST_BACKEND: str = "nccl"  # Use NCCL for GPU training
    
    # Other
    NUM_WORKERS: int = 2  # Reduced workers per process in distributed setting
    LOG_INTERVAL: int = 10  # Log every N batches


def setup_distributed(global_rank, world_size):
    """
    Initialize the distributed environment.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Attempting to initialize process group...")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] MASTER_ADDR: {master_addr}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] MASTER_PORT: {master_port}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Global RANK: {global_rank}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Env RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Env LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] WORLD_SIZE: {world_size}")
    
    # Log NCCL specific environment variables that might be relevant
    nccl_debug = os.environ.get('NCCL_DEBUG', 'Not set')
    nccl_socket_ifname = os.environ.get('NCCL_SOCKET_IFNAME', 'Not set (auto-detect)')
    nccl_ib_disable = os.environ.get('NCCL_IB_DISABLE', 'Not set (0 if IB enabled, 1 if disabled)')
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] NCCL_DEBUG: {nccl_debug}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] NCCL_SOCKET_IFNAME: {nccl_socket_ifname}")
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] NCCL_IB_DISABLE: {nccl_ib_disable}")

    # Set this process to use the specified GPU
    torch.cuda.set_device(local_rank)
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Set CUDA device to: cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})")
    
    # Initialize the process group
    # The init_method 'env://' will use MASTER_ADDR and MASTER_PORT from environment variables.
    # RANK and WORLD_SIZE also need to be set in the environment for 'env://' to work reliably.
    # Torchrun typically sets these.
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Calling dist.init_process_group (backend: {TrainConfig.DIST_BACKEND}, init_method: env://, global_rank: {global_rank})")
    dist.init_process_group(
        backend=TrainConfig.DIST_BACKEND,
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )
    
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Process group initialized successfully. is_initialized: {dist.is_initialized()}")
    
    # Barrier to ensure all processes have initialized
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Waiting for all processes at barrier after init_process_group...")
    dist.barrier()
    print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Barrier passed. All processes synchronized.")


def cleanup():
    """
    Clean up the distributed environment.
    """
    if dist.is_initialized():
        current_global_rank = os.environ.get('RANK', 'N/A')
        local_rank_env = os.environ.get('LOCAL_RANK', 'N/A')
        print(f"[GlobalRank {current_global_rank} LocalRank {local_rank_env}] Destroying process group...")
        dist.destroy_process_group()
        print(f"[GlobalRank {current_global_rank} LocalRank {local_rank_env}] Process group destroyed.")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint if it exists."""
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint path does not exist: {checkpoint_path}. Starting from scratch.")
        return 0, None
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Ensure the map_location is correctly set for the current process's device
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        raw_model_state_dict_from_checkpoint = None
        if "model_state_dict" in checkpoint:
            raw_model_state_dict_from_checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:  # Another common pattern
            raw_model_state_dict_from_checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and not any(k in checkpoint for k in ["epoch", "optimizer_state_dict"]):
            # Assume checkpoint is the state_dict itself if other specific keys aren't there
            raw_model_state_dict_from_checkpoint = checkpoint
        else:
            print(f"ERROR: Could not find model state_dict in checkpoint: {checkpoint_path}")
            return 0, None

        if raw_model_state_dict_from_checkpoint:
            adjusted_state_dict = OrderedDict()
            # The model passed here is the DDP-wrapped model
            is_ddp_model = isinstance(model, DDP) 

            for k_ckpt, v_ckpt in raw_model_state_dict_from_checkpoint.items():
                if is_ddp_model:
                    if not k_ckpt.startswith('module.'):
                        # Checkpoint key is plain (e.g., from model.module.state_dict()), DDP model needs "module."
                        adjusted_k = 'module.' + k_ckpt
                    else:
                        # Checkpoint key already has "module."
                        adjusted_k = k_ckpt
                else: # Model is not DDP
                    if k_ckpt.startswith('module.'):
                        # Checkpoint key has "module.", non-DDP model needs plain key.
                        adjusted_k = k_ckpt[7:]
                    else:
                        adjusted_k = k_ckpt
                adjusted_state_dict[adjusted_k] = v_ckpt
            
            model.load_state_dict(adjusted_state_dict)
        else:
            print(f"ERROR: Model state_dict was not found or identified in checkpoint: {checkpoint_path}")
            return 0, None

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint and hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1  # Resume from the next epoch
        last_loss = checkpoint.get("loss", float("nan"))
        
        # Get rank for logging
        log_prefix = ""
        if dist.is_initialized():
            log_prefix = f"[GlobalRank {dist.get_rank()} LocalRank {os.environ.get('LOCAL_RANK', 'N/A')}] "
        
        print(f"{log_prefix}Successfully loaded checkpoint. Model and optimizer restored. Resuming from epoch {start_epoch}.")
        print(f"{log_prefix}Loss from last saved epoch ({start_epoch - 1}): {last_loss:.4f}")
        return start_epoch, last_loss
        
    except Exception as e:
        import traceback
        log_prefix_err = ""
        if dist.is_initialized():
            log_prefix_err = f"[GlobalRank {dist.get_rank()} LocalRank {os.environ.get('LOCAL_RANK', 'N/A')}] "
        print(f"{log_prefix_err}ERROR loading checkpoint: {e}. Training will start from scratch.")
        traceback.print_exc()
        return 0, None


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_name = f"spynet_modified_ddp_epoch_ddp{epoch + 1}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # For DDP, save the module's state dict
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    # Save model, optimizer, epoch, and loss
    save_dict = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")


def prepare_dataloader(dataset, batch_size, rank, world_size, num_workers=2):
    """Create a distributed data loader."""
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True
    )
    
    return loader, sampler


def run_training(global_rank, world_size, config):
    """Run the training process on each distributed process."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    try:
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Starting training on GPU: {local_rank} ({torch.cuda.get_device_name(local_rank if torch.cuda.is_available() else 0)})")
        
        # Setup the distributed environment
        setup_distributed(global_rank, world_size)
        
        # Set the device for this process using its local_rank
        device = torch.device(f"cuda:{local_rank}")
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Active CUDA device: {torch.cuda.current_device()} (should be {local_rank})")
        
        # 1. Model
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Initializing SPyNetModified...")
        model = SPyNetModified(
            model_name=config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME,
            pretrained=config.LOAD_PRETRAINED_FOR_MODIFIED if not config.RESUME_TRAINING else False,
        )
        model = model.to(device)
        
        # Wrap the model with DDP, using local_rank for device_ids
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Wrapping model with DDP (device_ids=[{local_rank}], output_device={local_rank}, find_unused_parameters=True)...")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Model wrapped with DDP")
        
        # 2. Optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        criterion = torch.nn.L1Loss()
        
        # 3. Resume training (if applicable)
        start_epoch = config.START_EPOCH
        if config.RESUME_TRAINING and config.RESUME_CHECKPOINT_PATH:
            start_epoch, _ = load_checkpoint(model, optimizer, config.RESUME_CHECKPOINT_PATH, device)
        
        # 4. Dataset and DataLoader
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Preparing dataset and DataLoader...")
        try:
            train_dataset = FlowDataset(
                vimeo_clean_test_dir=config.VIMEO_CLEAN_TEST_DIR,
                defencing_dir=config.DEFENCING_DIR,
                vanilla_spynet_model_name=config.VANILLA_SPYNET_MODEL_NAME,
            )
            
            if len(train_dataset) == 0:
                print(f"[GlobalRank {global_rank} LocalRank {local_rank}] ERROR: Training dataset is empty. Aborting training.")
                cleanup()
                return
                
            # Synchronize before creating dataloaders
            dist.barrier()
            
            train_loader, train_sampler = prepare_dataloader(
                train_dataset, 
                config.BATCH_SIZE, 
                global_rank,
                world_size, 
                config.NUM_WORKERS
            )
            
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Dataset and DataLoader ready. Batches per epoch: {len(train_loader)}")
            
        except Exception as e:
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] ERROR: Failed to create dataset: {e}")
            cleanup()
            return
        
        # Wait for all processes to be ready before training
        dist.barrier()
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] All processes synchronized, beginning training loop at {datetime.datetime.now()}")
        
        # 5. Training loop
        model.train()
        
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            epoch_start_time = time.time()
            running_loss = 0.0
            
            # Set the epoch for the sampler
            train_sampler.set_epoch(epoch)
            
            for i, batch_data in enumerate(train_loader):
                input1_rgbm = batch_data["input1_rgbm"].to(device)
                input2_rgbm = batch_data["input2_rgbm"].to(device)
                gt_flow = batch_data["gt_flow"].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predicted_flow = model(input1_rgbm, input2_rgbm)
                
                # Calculate loss
                loss = criterion(predicted_flow, gt_flow)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i + 1) % config.LOG_INTERVAL == 0:
                    print(
                        f"[GlobalRank {global_rank} LocalRank {local_rank}] Epoch [{epoch + 1}/{config.NUM_EPOCHS}], "
                        f"Batch [{i + 1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f} (Average: {running_loss / (i + 1):.4f})"
                    )
            
            # Gather loss from all processes
            epoch_loss = running_loss / len(train_loader)
            avg_loss_tensor = torch.tensor([epoch_loss], dtype=torch.float, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / world_size
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            # Log on all ranks for now to see if they complete epochs
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] --- End of Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Average loss (local): {epoch_loss:.4f}")
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Epoch duration: {epoch_duration:.2f}s")

            # Only the master process (global_rank 0) logs the final epoch stats and saves checkpoints
            if global_rank == 0:
                print(f"--- MASTER (GlobalRank 0) End of Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
                print(f"--- MASTER (GlobalRank 0) Average loss (all processes): {avg_loss:.4f} ---")
                
                # Save checkpoint
                if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == config.NUM_EPOCHS:
                    save_checkpoint(model, optimizer, epoch, avg_loss, config.CHECKPOINT_DIR)
                
                print("-" * 30)
            
            # Synchronize processes at the end of each epoch
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Waiting for barrier at end of epoch {epoch+1}...")
            dist.barrier()
            print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Barrier passed at end of epoch {epoch+1}.")
        
        print(f"[GlobalRank {global_rank} LocalRank {local_rank}] Training completed successfully at {datetime.datetime.now()}")
        cleanup()
        
    except Exception as e:
        import traceback
        current_global_rank_exc = os.environ.get('RANK', 'N/A')
        local_rank_env_exc = os.environ.get('LOCAL_RANK', 'N/A')
        print(f"[GlobalRank {current_global_rank_exc} LocalRank {local_rank_env_exc}] ERROR: Exception in training process: {str(e)}")
        traceback.print_exc()
        # Try to clean up gracefully
        try:
            cleanup()
        except:
            pass


def main():
    # These are set by torchrun
    local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
    global_rank_env = int(os.environ.get("RANK", 0))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    
    # Print debugging information
    print(f"--- Main process started (PID: {os.getpid()}) ---")
    print(f"Local Rank (from env): {local_rank_env}")
    print(f"Global Rank (from env): {global_rank_env}")
    print(f"World Size (from env): {world_size_env}")
    print(f"Master Addr (from env): {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"Master Port (from env): {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES (from env): {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Configure multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
        print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] Multiprocessing context already set, likely 'spawn'.")
    
    # Print CUDA device information
    if torch.cuda.is_available():
        print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] CUDA available: {torch.cuda.is_available()}")
        print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] Number of CUDA devices seen by this process: {torch.cuda.device_count()}")
        # Each process should only see the GPUs assigned to it by CUDA_VISIBLE_DEVICES (managed by torchrun/slurm)
        # For local_rank 0 on a node with 1 GPU, device_count should be 1.
        try:
            print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] CUDA device {local_rank_env} name: {torch.cuda.get_device_name(local_rank_env)}")
        except Exception as e:
            print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] Could not get device name for local_rank {local_rank_env}: {e}")

    else:
        print(f"[GlobalRank {global_rank_env} LocalRank {local_rank_env}] ERROR: CUDA is not available! Distributed training requires GPU.")
        return
    
    # Create configuration
    config = TrainConfig()
    
    # Validate paths - only on global_rank 0 to avoid redundant checks and potential FS issues
    if global_rank_env == 0:
        print("[GlobalRank 0] Validating paths...")
        if "data_raw/" in config.VIMEO_CLEAN_TEST_DIR or "data_raw/" in config.DEFENCING_DIR:
            if not os.path.exists(config.VIMEO_CLEAN_TEST_DIR) or not os.path.exists(config.DEFENCING_DIR):
                print(f"[GlobalRank 0] ERROR: Data paths do not exist. Please check your configuration.")
                print(f"VIMEO_CLEAN_TEST_DIR: {config.VIMEO_CLEAN_TEST_DIR} exists: {os.path.exists(config.VIMEO_CLEAN_TEST_DIR)}")
                print(f"DEFENCING_DIR: {config.DEFENCING_DIR} exists: {os.path.exists(config.DEFENCING_DIR)}")
                # Signal other processes to exit or handle error
                # Consider using dist.abort(error_code) if init_process_group has already run,
                # but here it's before, so a simple return should be okay for rank 0.
                # Other ranks might hang if they proceed and rank 0 exits early.
                # For robustness, one might implement a more sophisticated error propagation.
                return 
    
    # Run the training function, passing the global_rank and world_size
    # The run_training function will then use LOCAL_RANK internally for device placement.
    run_training(global_rank_env, world_size_env, config)


if __name__ == "__main__":
    main() 