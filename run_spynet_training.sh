#!/bin/bash -l
#SBATCH -J spynet_training_defencing
#SBATCH --partition lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4     # Dopasuj do NUM_WORKERS w TrainConfig
#SBATCH --mem=32G             # Dostosuj ilość RAM, jeśli potrzebujesz więcej (domyślnie może być mało)
#SBATCH --gres=gpu:1          # Alokacja 1 GPU
#SBATCH --time=24:00:00       # Przykładowy czas, dostosuj
#SBATCH --output="spynet_training_output_%j.out" # Plik wyjściowy (%j to ID zadania)
#SBATCH --error="spynet_training_error_%j.err"   # Plik z błędami

echo "------------------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Run on host: `hostname`"
echo "Operating system: `uname -s -r`"
echo "Username: `whoami`"
echo "Started at: `date`"
echo "------------------------------------------------------------------------"

# Załaduj potrzebne moduły (przykładowe, sprawdź dostępne na WCSS i dopasuj wersje)
# Użyj `module avail` na klastrze, aby zobaczyć dostępne moduły.
module load tools/Anaconda3/2023.09-0 # Lub inna wersja Anacondy/Minicondy
module load lib/cuDNN/8.9.2.26-CUDA-11.8 # Dopasuj wersję cuDNN do CUDA i PyTorch
module load system/CUDA/11.8.0          # Dopasuj wersję CUDA do PyTorch i sterowników GPU

# Aktywuj swoje środowisko (jeśli używasz)
# Zastąp 'your_pytorch_env' nazwą swojego środowiska Conda
source activate your_pytorch_env
# LUB dla venv:
# source /path/to/your/Defencing_paper_replication/venv/bin/activate

# Przejdź do katalogu projektu (zakładając, że skrypt sbatch jest w głównym katalogu)
# lub dostosuj ścieżkę do skryptu Pythona poniżej.
# cd /path/to/your/Defencing_paper_replication
# Jeśli skrypt sbatch jest w głównym katalogu projektu, to cd nie jest potrzebne,
# jeśli ścieżki w Pythonie są relatywne lub poprawne absolutne.

# Sprawdź, czy GPU jest dostępne dla PyTorch
echo "Checking PyTorch CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')"

echo "------------------------------------------------------------------------"
echo "Starting Python training script..."

# Uruchom skrypt treningowy
# Upewnij się, że ścieżki w TrainConfig w pliku train_flow.py są POPRAWNE
# dla systemu plików klastra WCSS (np. /lustre/scratch/your_user/data_raw/...)
# Zamiast `python spynet/train_flow.py` możesz chcieć użyć `python -u spynet/train_flow.py`
# dla niebuforowanego outputu, co może pomóc w śledzeniu postępów.
python -u spynet/train_flow.py

echo "------------------------------------------------------------------------"
echo "Python script finished."
echo "Finished at: `date`"
echo "------------------------------------------------------------------------"