import os
import sys
import torch
import random
import glob
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from collections import OrderedDict
from typing import Union, Any, List, Dict, Optional

# Setup sys.path for sibling imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
rdn_dir = os.path.dirname(current_script_dir)  # rdn/
project_root_dir = os.path.dirname(rdn_dir)  # project root

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
if rdn_dir not in sys.path:
    sys.path.append(rdn_dir)

# Imports from the project
try:
    from rdn.augmentations import augment_background_burst, augment_fence_for_burst
    from rdn.utils import warp_frame_with_flow, create_validity_mask
    from spynet.spynet_modified import SPyNetModified
    import torchvision.transforms as T
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure rdn and spynet packages are structured correctly and in PYTHONPATH.")
    sys.exit(1)


class GenerationConfig:
    # Paths
    data_directory: str = "data/"
    vimeo_dir: str = data_directory + "data_raw/vimeo_test_clean/sequences"
    defencing_dir: str = data_directory + "data_raw/De-fencing-master/dataset"
    output_base_dir: str = data_directory + "data_precomputed/rdn_data"
    spynet_m_weights_path: str = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"
    )

    # Generation parameters
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 300
    k_frames: int = 5
    img_width: int = 320
    img_height: int = 192
    spynet_base_model_name: str = "sintel-final"

    # Data splitting parameters (Vimeo)
    # Option 1: Specify ratios (test_split_ratio is derived: 1 - train - val)
    train_split_ratio: Optional[float] = 0.8
    val_split_ratio: Optional[float] = 0.1
    vimeo_master_list_file: str = "sep_testlist.txt"

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 1924

    # Derived
    num_input_channels: int  # RDN input channels: 4 + (k_frames - 1) * 7

    def __init__(self, **kwargs):
        # Set defaults from class attributes to instance attributes
        class_attrs = {
            attr_name: getattr(GenerationConfig, attr_name)
            for attr_name in dir(GenerationConfig)
            if not attr_name.startswith("__")
            and not callable(getattr(GenerationConfig, attr_name))
            and attr_name
            not in ["num_input_channels", "to_dict"]  # Exclude derived and methods
        }
        for key, value in class_attrs.items():
            setattr(self, key, value)

        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            # else:
            #     print(f"Warning: GenerationConfig received unexpected kwarg: {key}")

        # Calculate and set derived attributes
        self.num_input_channels = 4 + (self.k_frames - 1) * 7

        # Validate split ratios if provided
        if self.train_split_ratio is not None and self.val_split_ratio is not None:
            if not (
                0.0 <= self.train_split_ratio <= 1.0
                and 0.0 <= self.val_split_ratio <= 1.0
                and (self.train_split_ratio + self.val_split_ratio) <= 1.0
            ):
                raise ValueError(
                    "Invalid split ratios. Train and Val ratios must be in [0,1] and their sum <= 1."
                )

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }


def load_spynet(
    config: GenerationConfig, target_device: torch.device
) -> Union[SPyNetModified, Any]:
    print(
        f"Loading SPyNetModified from {config.spynet_m_weights_path} for flow generation..."
    )
    spynet_m = SPyNetModified(
        model_name=config.spynet_base_model_name, pretrained=False
    )
    if not os.path.exists(config.spynet_m_weights_path):
        raise FileNotFoundError(
            f"SPyNet-M weights not found: {config.spynet_m_weights_path}"
        )

    checkpoint = torch.load(
        config.spynet_m_weights_path, map_location=torch.device("cpu")
    )
    state_dict_key = (
        "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    )
    if state_dict_key not in checkpoint:
        state_dict_to_load = checkpoint  # Assume checkpoint IS the state_dict
    else:
        state_dict_to_load = checkpoint[state_dict_key]

    new_state_dict = OrderedDict()
    for k, v in state_dict_to_load.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    spynet_m.load_state_dict(new_state_dict)
    spynet_m.to(target_device)
    spynet_m.eval()
    print(f"SPyNetModified loaded to {target_device}.")

    return spynet_m


def generate_single_sample(
    vimeo_frame_paths,
    fence_items,
    spynet_m: Union[SPyNetModified, Any],
    config,
    to_tensor,
    spynet_device,
):
    """Generates a single training/validation sample (f_in, I_k_m, B_k)."""
    # Adapted from InpaintingDataset.__getitem__

    start_frame_idx = random.randint(0, len(vimeo_frame_paths) - config.k_frames)
    burst_frame_paths = vimeo_frame_paths[
        start_frame_idx : start_frame_idx + config.k_frames
    ]
    bg_frames_pil = [Image.open(p).convert("RGB") for p in burst_frame_paths]

    bg_frames_aug_pil = augment_background_burst(
        bg_frames_pil, config.img_height, config.img_width
    )
    B_j_tensors = [to_tensor(frame) for frame in bg_frames_aug_pil]

    keyframe_idx_in_burst = config.k_frames // 2
    B_k_tensor = B_j_tensors[keyframe_idx_in_burst]

    selected_fence_item = random.choice(fence_items)
    F_raw_pil = Image.open(selected_fence_item["img"]).convert("RGB")
    M_raw_fence_pil = Image.open(selected_fence_item["mask"]).convert("L")

    augmented_fences_pil = augment_fence_for_burst(
        F_raw_pil,
        M_raw_fence_pil,
        config.k_frames,
        config.img_height,
        config.img_width,
    )

    I_j_tensors = []
    S_j_tensors = []

    for i in range(config.k_frames):
        B_j_i_tensor = B_j_tensors[i]
        F_prime_pil, M_prime_pil = augmented_fences_pil[i]
        F_prime_tensor = to_tensor(F_prime_pil)
        M_prime_tensor = (to_tensor(M_prime_pil) > 0.5).float()
        I_j_i_tensor = (
            M_prime_tensor * F_prime_tensor + (1 - M_prime_tensor) * B_j_i_tensor
        )
        I_j_tensors.append(I_j_i_tensor)
        S_j_tensors.append(M_prime_tensor)

    I_k_tensor = I_j_tensors[keyframe_idx_in_burst]
    S_k_tensor = S_j_tensors[keyframe_idx_in_burst]

    I_k_m_tensor = I_k_tensor.to(spynet_device) * (1 - S_k_tensor.to(spynet_device))
    all_features_for_f_in = [I_k_m_tensor, S_k_tensor.to(spynet_device)]

    for j in range(config.k_frames):
        if j == keyframe_idx_in_burst:
            continue

        I_j_current_tensor = I_j_tensors[j].to(spynet_device)
        S_j_current_tensor = S_j_tensors[j].to(spynet_device)

        input_k_rgbm = torch.cat(
            [I_k_tensor.to(spynet_device), S_k_tensor.to(spynet_device)], dim=0
        )
        input_j_rgbm = torch.cat([I_j_current_tensor, S_j_current_tensor], dim=0)

        with torch.inference_mode():
            f_kj_m_tensor = spynet_m(
                input_k_rgbm.unsqueeze(0), input_j_rgbm.unsqueeze(0)
            ).squeeze(0)

        I_j_m_content_to_warp = I_j_current_tensor * (1 - S_j_current_tensor)

        Î_j_m_tensor = warp_frame_with_flow(
            I_j_m_content_to_warp.to(f_kj_m_tensor.device), f_kj_m_tensor
        )
        Š_j_tensor = warp_frame_with_flow(
            S_j_current_tensor.to(f_kj_m_tensor.device), f_kj_m_tensor
        )
        Š_j_tensor = (Š_j_tensor > 0.5).float()
        V_j_tensor = create_validity_mask(
            config.img_height, config.img_width, f_kj_m_tensor
        )

        all_features_for_f_in.extend(
            [Î_j_m_tensor, Š_j_tensor, V_j_tensor, f_kj_m_tensor]
        )

    f_in_tensor = torch.cat(all_features_for_f_in, dim=0)

    if f_in_tensor.shape[0] != config.num_input_channels:
        print(
            f"WARNING: Generated f_in_tensor has {f_in_tensor.shape[0]} channels, expected {config.num_input_channels}."
        )

    # Return tensors on CPU for saving
    return f_in_tensor.cpu(), I_k_m_tensor.cpu(), B_k_tensor.cpu()


def _load_vimeo_sequences_from_file(
    list_file_path: str, vimeo_base_seq_dir: str, k_frames: int
) -> List[List[str]]:
    """Loads Vimeo sequence paths from a single list file."""
    sequences = []
    if not os.path.exists(list_file_path):
        # print(f"Info: List file not found at {list_file_path}")
        return sequences  # Return empty if file not found, handled by caller

    with open(list_file_path, "r") as f:
        for line in f:
            seq_folder_path = os.path.join(vimeo_base_seq_dir, line.strip())
            frames_in_seq = sorted(glob.glob(os.path.join(seq_folder_path, "im*.png")))
            if len(frames_in_seq) >= k_frames:
                sequences.append(frames_in_seq)
    return sequences


def get_vimeo_data_splits(config: GenerationConfig) -> Dict[str, List[List[str]]]:
    """
    Loads Vimeo sequence paths and splits them into train, val, and test sets.
    Prioritizes specific list files. If not found, uses ratios on a master list.
    """
    vimeo_list_dir = os.path.dirname(
        config.vimeo_dir.rstrip("/\\ ")
    )  # Dir containing sep_*.txt files

    data_splits: Dict[str, List[List[str]]] = {"train": [], "val": [], "test": []}

    master_list_path = os.path.join(vimeo_list_dir, config.vimeo_master_list_file)
    all_sequences = _load_vimeo_sequences_from_file(
        master_list_path, config.vimeo_dir, config.k_frames
    )

    if not all_sequences:
        raise FileNotFoundError(
            f"Master Vimeo list file not found or empty: {master_list_path}"
        )

    if config.train_split_ratio is None or config.val_split_ratio is None:
        raise ValueError(
            "train_split_ratio and val_split_ratio must be set if not using specific list files."
        )

    random.shuffle(all_sequences)  # Shuffle for random split
    num_total = len(all_sequences)
    num_train = int(config.train_split_ratio * num_total)
    num_val = int(config.val_split_ratio * num_total)

    data_splits["train"] = all_sequences[:num_train]
    data_splits["val"] = all_sequences[num_train : num_train + num_val]
    data_splits["test"] = all_sequences[num_train + num_val :]

    print(
        f"Split master list: {len(data_splits['train'])} train, {len(data_splits['val'])} val, {len(data_splits['test'])} test sequences."
    )
    return data_splits


def main():
    gen_config = GenerationConfig()
    print(
        f"Starting data generation with config:\n{json.dumps(gen_config.to_dict(), indent=2)}"
    )

    torch.manual_seed(gen_config.random_seed)
    np.random.seed(gen_config.random_seed)
    random.seed(gen_config.random_seed)

    spynet_processing_device = torch.device(gen_config.device)
    spynet_m_loaded: Union[SPyNetModified, Any] = load_spynet(
        gen_config, spynet_processing_device
    )

    to_tensor_transform = T.ToTensor()

    # --- Load dataset file lists and split them ---
    try:
        vimeo_splits = get_vimeo_data_splits(gen_config)
        vimeo_train_sequences = vimeo_splits["train"]
        vimeo_val_sequences = vimeo_splits["val"]
        vimeo_test_sequences = vimeo_splits["test"]

        if not vimeo_train_sequences and gen_config.num_train_samples > 0:
            print(
                "Warning: No Vimeo training sequences loaded, but num_train_samples > 0."
            )
        if not vimeo_val_sequences and gen_config.num_val_samples > 0:
            print(
                "Warning: No Vimeo validation sequences loaded, but num_val_samples > 0."
            )
        if not vimeo_test_sequences and gen_config.num_test_samples > 0:
            print("Warning: No Vimeo test sequences loaded, but num_test_samples > 0.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading or splitting Vimeo data: {e}")
        return

    # Simplified fence item loading (adapt from InpaintingDataset._load_defencing_items for robustness)
    # Fences are typically common assets, not split into train/val/test for this purpose.
    # The split applies to the background (Vimeo) sequences.
    fence_img_dir = os.path.join(
        gen_config.defencing_dir, "Training Set", "Training_Images"
    )
    fence_mask_dir = os.path.join(
        gen_config.defencing_dir, "Training Set", "Training_Labels"
    )
    loaded_fence_items = []
    if os.path.isdir(fence_img_dir) and os.path.isdir(fence_mask_dir):
        fence_img_names = sorted(os.listdir(fence_img_dir))
        for name in fence_img_names:
            base, _ = os.path.splitext(name)
            mask_path = os.path.join(
                fence_mask_dir, base + ".png"
            )  # Assuming .png for masks
            if not os.path.exists(mask_path):
                mask_path = os.path.join(
                    fence_mask_dir, name
                )  # Try exact name match for mask

            if os.path.exists(mask_path):
                loaded_fence_items.append(
                    {"img": os.path.join(fence_img_dir, name), "mask": mask_path}
                )
    if not loaded_fence_items:
        raise ValueError("No fence items loaded.")
    print(f"Loaded {len(loaded_fence_items)} fence items.")

    # --- Loop for Train and Val data generation ---
    for dataset_type, num_samples_to_gen, source_vimeo_sequences in [
        ("train", gen_config.num_train_samples, vimeo_train_sequences),
        ("val", gen_config.num_val_samples, vimeo_val_sequences),
        ("test", gen_config.num_test_samples, vimeo_test_sequences),
    ]:
        if num_samples_to_gen == 0:
            print(f"Skipping {dataset_type} set as num_samples is 0.")
            continue

        if not source_vimeo_sequences:
            print(
                f"Skipping {dataset_type} set as no source Vimeo sequences are available for it."
            )
            continue

        output_dir_specific = os.path.join(gen_config.output_base_dir, dataset_type)
        os.makedirs(output_dir_specific, exist_ok=True)
        print(
            f"\nGenerating {num_samples_to_gen} samples for {dataset_type} set in {output_dir_specific}..."
        )
        print(
            f"  (using {len(source_vimeo_sequences)} available source sequences for this set)"
        )

        # Adjust num_samples_to_gen if it exceeds the available unique sequences for this split,
        # to avoid errors if random.choice is on an empty list or if we want to ensure variety.
        # However, random.choice allows replacement by default, so it's more about managing expectations.
        # For now, we'll proceed and let it sample with replacement if num_samples_to_gen > len(source_vimeo_sequences)

        existing_files = sorted(
            glob.glob(os.path.join(output_dir_specific, "sample_*.pt"))
        )
        existing_indices = [
            int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing_files
        ]
        start_idx = max(existing_indices) + 1 if existing_indices else 0
        num_to_generate = max(0, num_samples_to_gen - len(existing_indices))

        for i in tqdm(
            range(start_idx, start_idx + num_to_generate),
            desc=f"Generating {dataset_type} data",
        ):
            # Ustaw seed zależny od numeru sampla
            torch.manual_seed(gen_config.random_seed + i)
            np.random.seed(gen_config.random_seed + i)
            random.seed(gen_config.random_seed + i)

            if not source_vimeo_sequences:  # Should be caught above, but as a safeguard
                print(
                    f"Error: No source sequences for {dataset_type} to generate sample {i}. Stopping generation for this set."
                )
                break
            selected_vimeo_paths = random.choice(source_vimeo_sequences)

            try:
                f_in, i_k_m, b_k = generate_single_sample(
                    selected_vimeo_paths,
                    loaded_fence_items,
                    spynet_m_loaded,
                    gen_config,
                    to_tensor_transform,
                    spynet_processing_device,
                )

                save_path = os.path.join(output_dir_specific, f"sample_{i:06d}.pt")
                torch.save({"f_in": f_in, "i_k_m": i_k_m, "b_k": b_k}, save_path)
            except Exception as e_sample:
                print(f"\nError generating sample {i} for {dataset_type}: {e_sample}")
                import traceback

                traceback.print_exc()
                print(f"Skipping sample {i}.")
                continue

        print(f"Finished generating {dataset_type} data.")

    # Save the generation config
    config_save_path = os.path.join(
        gen_config.output_base_dir, "generation_config.json"
    )
    with open(config_save_path, "w") as f_config:
        json.dump(gen_config.to_dict(), f_config, indent=4)
    print(f"Generation config saved to {config_save_path}")
    print("Data generation process complete.")


if __name__ == "__main__":
    main()
