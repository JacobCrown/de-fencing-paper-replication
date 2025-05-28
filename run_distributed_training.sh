#!/bin/bash
#SBATCH --job-name=spynet-ddp
#SBATCH --partition=lem-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00 # Shortened time for testing
#SBATCH --mem=64gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --output="logs3/slurm_output_%j.out"
#SBATCH --error="logs3/slurm_error_%j.err"

# Create logs directory if it doesn't exist
mkdir -p logs3

# Load required modules
source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.3.0 # Ensure this matches PyTorch's CUDA version expectation

# Setup for PyTorch distributed
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get the primary IP address of the head node
# Using srun to execute hostname -I on the head_node and taking the first IP
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address -I | awk '{print $1}')
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# NCCL Debugging - Set to INFO for verbose output, or WARN for warnings
export NCCL_DEBUG=INFO
# export NCCL_DEBUG=WARN

# Optional: Specify network interface if auto-detection fails (e.g., for InfiniBand)
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=0 # Set to 0 to enable InfiniBand, 1 to disable

# Activate your Python virtual environment
# IMPORTANT: Update this path to your virtual environment
PROJECT_DIR="/home/jacob.crown/de-fencing-paper-replication"
VENV_DIR="$PROJECT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    echo "Activating Python virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR" >&2
    exit 1
fi

# Python path for imports (if spynet is a module)
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Print environment for debugging
echo "--- SLURM SETTINGS ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-not_set} (Note: --gres=gpu:hopper:1 implies 1)"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Number of nodes: $(echo ${#nodes_array[@]})"

echo "--- PYTORCH DISTRIBUTED SETTINGS ---"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
# These will be set by torchrun
# echo "RANK: \$RANK"
# echo "WORLD_SIZE: \$WORLD_SIZE"
# echo "LOCAL_RANK: \$LOCAL_RANK"

echo "--- NCCL SETTINGS ---"
echo "NCCL_DEBUG: $NCCL_DEBUG"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-auto-detect}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-not_set (default typically 0 or 1 based on build)}"

echo "--- PYTHON & SCRIPT ---"
echo "PYTHONPATH: $PYTHONPATH"
echo "Which python: $(which python)"
echo "Python version: $(python --version)"
TRAINING_SCRIPT="$PROJECT_DIR/spynet/train_flow_ddp.py"
echo "Training script: $TRAINING_SCRIPT"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAINING_SCRIPT" >&2
    exit 1
fi

echo "--- Starting torchrun on each node ---"

# --nproc_per_node should match the number of GPUs per node requested from SLURM
# If --gres=gpu:hopper:1, then it's 1 GPU per node.
NPROC_PER_NODE=$(echo $SLURM_JOB_GRES | sed 's/.*gpu:[^:]*://' || echo 1) # Extracts count from --gres
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then NPROC_PER_NODE=1; fi # Default to 1 if extraction failed

echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# The --master_addr and --master_port are passed to torchrun.
# RANK and WORLD_SIZE are automatically managed by srun + torchrun when launching.

srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    $TRAINING_SCRIPT

echo "--- torchrun finished ---"

# Deactivate virtual environment (optional, as script exits)
if type deactivate > /dev/null 2>&1; then
    deactivate
fi 