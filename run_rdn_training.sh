#!/bin/bash
#SBATCH --job-name=rdn-training
#SBATCH --partition=lem-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00 # Shortened time for testing
#SBATCH --mem=64gb
#SBATCH --gres=gpu:hopper:1
#SBATCH --output="logs/rdn2/slurm_output_%j.out"
#SBATCH --error="logs/rdn2/slurm_error_%j.err"

# Create logs directory if it doesn't exist
mkdir -p logs/rdn2

# Load required modules
source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.3.0 # Ensure this matches PyTorch's CUDA version expectation

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


echo "--- PYTHON & SCRIPT ---"
echo "PYTHONPATH: $PYTHONPATH"
echo "Which python: $(which python)"
echo "Python version: $(python --version)"

DATA_GENERATION_SCRIPT="$PROJECT_DIR/rdn/test/generate_precomputed_rdn_data.py"
echo "Data generation script: $DATA_GENERATION_SCRIPT"

TRAINING_SCRIPT="$PROJECT_DIR/rdn/test/train_rdn_precomputed.py"
echo "Training script: $TRAINING_SCRIPT"

if [ ! -f "$DATA_GENERATION_SCRIPT" ]; then
    echo "Error: Data generation script not found at $DATA_GENERATION_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAINING_SCRIPT" >&2
    exit 1
fi

echo "--- Starting data generation ---"

python $DATA_GENERATION_SCRIPT

echo "--- data generation finished ---"

echo "-------------------------"

echo "--- Starting training ---"

python $TRAINING_SCRIPT

echo "--- Training finished ---"

# Deactivate virtual environment (optional, as script exits)
if type deactivate > /dev/null 2>&1; then
    deactivate
fi