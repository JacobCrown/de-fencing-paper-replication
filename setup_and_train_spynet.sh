#!/bin/bash
#SBATCH -J setup_train_spynet
#SBATCH --partition lem-gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4     # Dopasuj do NUM_WORKERS w TrainConfig
#SBATCH --mem=32G             # Dostosuj RAM
#SBATCH --gres=gpu:1          # Alokacja 1 GPU
#SBATCH --time=24:00:00       # Przykładowy czas, dostosuj
#SBATCH --output="logs/slurm_output_%j.out" # Plik wyjściowy (%j to ID zadania)
#SBATCH --error="logs/slurm_error_%j.err"   # Plik z błędami

# Utwórz katalog na logi, jeśli nie istnieje
mkdir -p logs

echo "------------------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s -r)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "------------------------------------------------------------------------"

# --- Konfiguracja Środowiska ---
# Wersje do ustawienia:
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.1.0" # Sprawdź kompatybilne wersje na stronie PyTorch
CUDA_VERSION_FOR_PYTORCH="11.8" # Wersja CUDA, z którą PyTorch ma być zbudowany
CONDA_ENV_NAME="spynet_env_py${PYTHON_VERSION}_pt${PYTORCH_VERSION}_cuda${CUDA_VERSION_FOR_PYTORCH}"

# Ścieżka do katalogu projektu (zakładamy, że skrypt sbatch jest w głównym katalogu)
PROJECT_DIR=$(pwd)
CONDA_ENVS_PATH="${PROJECT_DIR}/.conda_envs" # Instaluj środowiska lokalnie w projekcie

echo "Katalog projektu: ${PROJECT_DIR}"
echo "Nazwa środowiska Conda: ${CONDA_ENV_NAME}"
echo "Ścieżka instalacji środowisk Conda: ${CONDA_ENVS_PATH}"
echo "Python ${PYTHON_VERSION}, PyTorch ${PYTORCH_VERSION}, CUDA for PyTorch ${CUDA_VERSION_FOR_PYTORCH}"
echo "------------------------------------------------------------------------"

# Załaduj moduł Anaconda (sprawdź `module avail tools/Anaconda3` dla dostępnych wersji)
module purge # Zalecane przed ładowaniem nowych modułów
module load tools/Anaconda3/2023.09-0 # Lub inna stabilna wersja
module load lib/cuDNN/8.9.2.26-CUDA-11.8 # Dopasuj do CUDA_VERSION_FOR_PYTORCH
module load system/CUDA/11.8.0          # Dopasuj do CUDA_VERSION_FOR_PYTORCH

# Ustawienie ścieżki dla lokalnych środowisk Conda
mkdir -p "${CONDA_ENVS_PATH}"
conda config --add envs_dirs "${CONDA_ENVS_PATH}"
conda config --set env_prompt '({name}) ' # Dla ładniejszego promptu

# Sprawdź, czy środowisko Conda istnieje
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
    echo "Środowisko Conda '${CONDA_ENV_NAME}' już istnieje."
else
    echo "Tworzenie nowego środowiska Conda '${CONDA_ENV_NAME}'..."
    # Utwórz środowisko z określoną wersją Pythona
    # Użyj `conda create` bez `-p` aby użyć skonfigurowanej `envs_dirs`
    conda create --name "${CONDA_ENV_NAME}" python=${PYTHON_VERSION} -y
    if [ $? -ne 0 ]; then
        echo "BŁĄD: Nie udało się stworzyć środowiska Conda."
        exit 1
    fi
    echo "Środowisko Conda '${CONDA_ENV_NAME}' utworzone."
fi

# Aktywuj środowisko Conda
echo "Aktywowanie środowiska Conda: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}" # Dla `conda create --name`
# Jeśli używałeś `conda create -p path/to/env`, aktywuj przez `conda activate path/to/env`

if [ $? -ne 0 ]; then
    echo "BŁĄD: Nie udało się aktywować środowiska Conda."
    exit 1
fi
echo "Środowisko aktywowane. Aktualna ścieżka Python: $(which python)"
python --version

# Instalacja PyTorch z odpowiednią wersją CUDA
# Odwiedź https://pytorch.org/get-started/previous-versions/ aby znaleźć prawidłowe polecenie
# dla kombinacji PyTorch, OS, Conda, Python, CUDA.
echo "Instalowanie PyTorch ${PYTORCH_VERSION} dla CUDA ${CUDA_VERSION_FOR_PYTORCH}..."
conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio pytorch-cuda=${CUDA_VERSION_FOR_PYTORCH} -c pytorch -c nvidia -y

if [ $? -ne 0 ]; then
    echo "BŁĄD: Nie udało się zainstalować PyTorch."
    exit 1
fi
echo "PyTorch zainstalowany."

# Instalacja pozostałych zależności z requirements.txt
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo "Instalowanie zależności z ${PROJECT_DIR}/requirements.txt..."
    pip install -r "${PROJECT_DIR}/requirements.txt"
    if [ $? -ne 0 ]; then
        echo "BŁĄD: Nie udało się zainstalować zależności z requirements.txt."
        exit 1
    fi
    echo "Zależności z requirements.txt zainstalowane."
else
    echo "OSTRZEŻENIE: Plik requirements.txt nie został znaleziony w ${PROJECT_DIR}."
fi

# Sprawdzenie PyTorch i CUDA po instalacji
echo "------------------------------------------------------------------------"
echo "Weryfikacja instalacji PyTorch i dostępności CUDA w środowisku:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version reported by PyTorch: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'Num GPUs available: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A"}')"
echo "------------------------------------------------------------------------"

echo "Rozpoczynanie skryptu treningowego Python..."
# Upewnij się, że ścieżki w TrainConfig w pliku train_flow.py są POPRAWNE
# dla systemu plików klastra WCSS (np. /lustre/scratch/your_user/data_raw/...)
# Użyj `python -u` dla niebuforowanego outputu.
python -u "${PROJECT_DIR}/spynet/train_flow.py"

echo "------------------------------------------------------------------------"
echo "Skrypt Pythona zakończony."
echo "Zakończono o: $(date)"
echo "Dezaktywacja środowiska Conda..."
conda deactivate
echo "------------------------------------------------------------------------" 