import os
import subprocess
import urllib.request
import zipfile
import shutil
import sys

# Define paths relative to the workspace root
WORKSPACE_ROOT = (
    os.getcwd()
)  # Assumes script is run from workspace root, or use an absolute base path
DATA_RAW_DIR = os.path.join(WORKSPACE_ROOT, "data_raw")

# De-fencing dataset (GitHub)
DEFENCING_REPO_URL = "https://github.com/chen-du/De-fencing.git"
DEFENCING_TARGET_DIR_NAME = "De-fencing-master"  # To match your config
DEFENCING_CLONE_PATH = os.path.join(DATA_RAW_DIR, DEFENCING_TARGET_DIR_NAME)
DEFENCING_DATASET_SUBDIR = os.path.join(
    DEFENCING_CLONE_PATH, "dataset"
)  # The actual dataset folder

# Vimeo dataset (zip)
VIMEO_ZIP_URL = "http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip"
VIMEO_ZIP_NAME = "vimeo_test_clean.zip"
VIMEO_ZIP_PATH = os.path.join(DATA_RAW_DIR, VIMEO_ZIP_NAME)
VIMEO_EXTRACT_TARGET_DIR = os.path.join(DATA_RAW_DIR, "vimeo_test_clean")


def check_git_availability():
    """Checks if git is installed and available on PATH."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("Git is available.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "ERROR: Git is not installed or not found in PATH. Please install Git to download the De-fencing dataset."
        )
        return False


def download_defencing_dataset():
    """Downloads the De-fencing dataset using Git."""
    if not check_git_availability():
        return

    if os.path.exists(DEFENCING_DATASET_SUBDIR):
        print(f"De-fencing dataset already found at: {DEFENCING_DATASET_SUBDIR}")
        print("Skipping download.")
        return

    if os.path.exists(DEFENCING_CLONE_PATH):
        print(
            f"Target clone directory {DEFENCING_CLONE_PATH} already exists but doesn't contain the 'dataset' subfolder as expected, or is incomplete."
        )
        print(
            "Consider removing it manually and re-running the script if you want a fresh clone."
        )
        # Optionally, could add logic to remove and re-clone, but safer to ask user.
        # For now, we'll try to clone if DEFENCING_DATASET_SUBDIR is missing, even if DEFENCING_CLONE_PATH exists.
        # This might fail if git clone cannot operate on a non-empty dir that's not a git repo.
        # A better check for git clone failure due to existing dir is needed if we don't remove it.

    print(
        f"Cloning De-fencing dataset from {DEFENCING_REPO_URL} into {DEFENCING_CLONE_PATH}..."
    )
    try:
        # Clone directly into the desired folder name
        subprocess.run(
            ["git", "clone", DEFENCING_REPO_URL, DEFENCING_CLONE_PATH], check=True
        )
        print("De-fencing dataset cloned successfully.")
        if not os.path.exists(DEFENCING_DATASET_SUBDIR):
            print(
                f"ERROR: Git clone seemed successful, but the expected 'dataset' subdirectory is missing in {DEFENCING_CLONE_PATH}."
            )
            print("Please check the repository structure or the clone process.")
        else:
            print(f"Verified 'dataset' subdirectory at {DEFENCING_DATASET_SUBDIR}")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone De-fencing dataset. Git command failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during De-fencing dataset download: {e}")


def download_vimeo_dataset():
    """Downloads and extracts the Vimeo Septuplet clean test dataset."""
    if os.path.exists(VIMEO_EXTRACT_TARGET_DIR):
        print(f"Vimeo dataset directory already found at: {VIMEO_EXTRACT_TARGET_DIR}")
        print("Skipping download and extraction.")
        return

    print(f"Downloading Vimeo dataset from {VIMEO_ZIP_URL} to {VIMEO_ZIP_PATH}...")
    try:
        # Create a progress bar hook for urlretrieve
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\rDownload progress: {int(percent):3d}%% {readsofar / (1024 * 1024):.2f}MB / {totalsize / (1024 * 1024):.2f}MB"
                sys.stdout.write(s)
                if readsofar >= totalsize:  # near the end
                    sys.stdout.write("\n")
            else:  # total size is unknown
                sys.stdout.write(f"\rRead {readsofar / (1024 * 1024):.2f}MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(VIMEO_ZIP_URL, VIMEO_ZIP_PATH, reporthook=reporthook)
        print("Vimeo dataset ZIP downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download Vimeo dataset ZIP: {e}")
        if os.path.exists(VIMEO_ZIP_PATH):  # Clean up partial download
            os.remove(VIMEO_ZIP_PATH)
        return

    print(f"Extracting {VIMEO_ZIP_NAME} to {DATA_RAW_DIR}...")
    try:
        with zipfile.ZipFile(VIMEO_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_RAW_DIR)
        print("Vimeo dataset extracted successfully.")
        if not os.path.exists(VIMEO_EXTRACT_TARGET_DIR):
            print(
                f"ERROR: Extraction seemed successful, but the expected directory {VIMEO_EXTRACT_TARGET_DIR} was not created."
            )
            print("Please check the ZIP file contents or extraction process.")

    except zipfile.BadZipFile:
        print(
            f"ERROR: Failed to extract Vimeo dataset. The downloaded file {VIMEO_ZIP_NAME} may be corrupted or not a valid ZIP file."
        )
    except Exception as e:
        print(f"An unexpected error occurred during Vimeo dataset extraction: {e}")
    finally:
        # Clean up the downloaded ZIP file
        if os.path.exists(VIMEO_ZIP_PATH):
            print(f"Removing downloaded ZIP file: {VIMEO_ZIP_PATH}")
            os.remove(VIMEO_ZIP_PATH)


def main():
    print(f"Workspace root identified as: {WORKSPACE_ROOT}")
    print(f"Target data directory: {DATA_RAW_DIR}")

    # Create data_raw directory if it doesn't exist
    if not os.path.exists(DATA_RAW_DIR):
        print(f"Creating directory: {DATA_RAW_DIR}")
        os.makedirs(DATA_RAW_DIR)
    else:
        print(f"Directory {DATA_RAW_DIR} already exists.")

    print("\n--- Downloading De-fencing Dataset ---")
    download_defencing_dataset()

    print("\n--- Downloading Vimeo Dataset ---")
    download_vimeo_dataset()

    print("\n--- Dataset setup script finished ---")
    print(
        f"Please ensure your configuration paths in training/evaluation scripts are correct:"
    )
    print(f"  Vimeo dir (e.g., VIMEO_DIR): should point to {VIMEO_EXTRACT_TARGET_DIR}")
    print(
        f"  De-fencing dir (e.g., DEFENCING_DIR): should point to the 'dataset' subfolder within {DEFENCING_CLONE_PATH}, i.e., {DEFENCING_DATASET_SUBDIR}"
    )


if __name__ == "__main__":
    main()
