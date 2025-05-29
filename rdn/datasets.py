import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from collections import OrderedDict
from typing import Union  # For type hinting device

import sys

# Add parent directory to sys.path to allow sibling imports
current_dir_datasets = os.path.dirname(os.path.abspath(__file__))
parent_dir_datasets = os.path.dirname(current_dir_datasets)
if parent_dir_datasets not in sys.path:
    sys.path.append(parent_dir_datasets)

try:
    from spynet.spynet_modified import SPyNetModified
except ImportError as e:
    print(
        f"Could not import SPyNetModified from spynet.spynet_modified. Ensure spynet module is accessible: {e}"
    )
    SPyNetModified = None

from rdn.utils import warp_frame_with_flow, create_validity_mask
from rdn.augmentations import augment_background_burst, augment_fence_for_burst


class InpaintingDataset(Dataset):
    def __init__(
        self,
        config,
        spynet_model_path,
        spynet_model_name_for_gt_flow_in_spynet_m,
        is_train=True,
        spynet_device: Union[str, torch.device] = "cpu",
    ):
        self.config = config
        self.is_train = is_train
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.k_frames = config.k_frames
        self.spynet_target_device = torch.device(spynet_device)

        self.vimeo_root_dir = config.vimeo_dir
        self.defencing_root_dir = config.defencing_dir

        self.vimeo_sequences_paths = self._load_vimeo_sequences()
        self.fence_items = self._load_defencing_items()

        if not self.vimeo_sequences_paths:
            raise ValueError(
                f"No Vimeo sequences found. Check path and list file: {self.vimeo_root_dir}"
            )
        if (
            not self.fence_items and self.is_train
        ):  # Only critical if training and need fences
            # For evaluation, we might not need synthetic fences if evaluating on real data
            print(
                f"Warning: No fence items found. Check path: {self.defencing_root_dir}. This might be an issue for training."
            )
            # raise ValueError(f"No fence items found. Check path: {self.defencing_root_dir}")

        print(f"Loaded {len(self.vimeo_sequences_paths)} Vimeo sequence groups.")
        if self.fence_items:
            print(f"Loaded {len(self.fence_items)} fence items.")
        else:
            print(
                "No fence items loaded (this might be ok for certain evaluation modes)."
            )

        print("Loading pre-trained SPyNetModified for flow calculation...")
        if SPyNetModified is None:
            raise ImportError(
                "SPyNetModified class could not be imported. Cannot proceed with InpaintingDataset initialization."
            )

        print(
            f"Initializing SPyNetModified (will load to {self.spynet_target_device})..."
        )
        self.spynet_m = SPyNetModified(
            model_name=spynet_model_name_for_gt_flow_in_spynet_m, pretrained=False
        )
        try:
            map_location_load = torch.device("cpu")
            checkpoint = torch.load(spynet_model_path, map_location=map_location_load)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            self.spynet_m.load_state_dict(new_state_dict)
            self.spynet_m.to(self.spynet_target_device)
            print(
                f"Successfully loaded SPyNetModified weights from {spynet_model_path} to {self.spynet_target_device}"
            )
        except Exception as e:
            raise IOError(
                f"Error loading SPyNetModified weights from {spynet_model_path}: {e}"
            )

        self.spynet_m.eval()

        # Attempt to compile SPyNet-M for potential speedup
        if hasattr(torch, "compile") and self.spynet_target_device.type == "cuda":
            original_spynet_m_for_fallback = self.spynet_m  # Store original model
            compile_mode = "max-autotune"
            is_compiled = False
            try:
                print(
                    f"Attempting to torch.compile SPyNetModified on {self.spynet_target_device} with mode '{compile_mode}'..."
                )
                pt_version = torch.__version__
                major_version = int(pt_version.split(".")[0])

                if major_version >= 2:
                    compiled_model_candidate = torch.compile(
                        original_spynet_m_for_fallback, mode=compile_mode
                    )
                    print(
                        f"SPyNetModified compilation call successful on {self.spynet_target_device} with mode '{compile_mode}'. Testing with a dummy forward pass..."
                    )

                    # Determine appropriate dummy input size based on config, ensuring divisibility by 32 for pyramid
                    h_config = self.config.img_height
                    w_config = self.config.img_width

                    # Ensure dimensions are at least 32 and multiples of 32 for robust SPyNet pyramid processing
                    dummy_h = max(
                        32, (h_config + 31) // 32 * 32 if h_config > 0 else 32
                    )
                    dummy_w = max(
                        32, (w_config + 31) // 32 * 32 if w_config > 0 else 32
                    )

                    dummy_input1 = torch.randn(
                        1, 4, dummy_h, dummy_w, device=self.spynet_target_device
                    )
                    dummy_input2 = torch.randn(
                        1, 4, dummy_h, dummy_w, device=self.spynet_target_device
                    )

                    with torch.inference_mode():
                        _ = compiled_model_candidate(dummy_input1, dummy_input2)

                    self.spynet_m = compiled_model_candidate  # Assign successfully tested compiled model
                    is_compiled = True
                    print(
                        f"Dummy forward pass successful. Using compiled SPyNetModified (is_compiled = {is_compiled})."
                    )
                else:
                    print(
                        f"Skipping torch.compile for SPyNetModified: PyTorch version {pt_version} is less than 2.0."
                    )
            except Exception as e_compile_runtime:
                print(
                    f"WARNING: Failed to compile or run dummy forward pass for SPyNetModified with mode '{compile_mode}'. Reverting to non-compiled model. Error: {e_compile_runtime}"
                )
                self.spynet_m = original_spynet_m_for_fallback  # Revert to original

            # Store compilation status for __getitem__ to decide on cloning
            self.spynet_m_is_compiled_and_cuda = is_compiled and (
                self.spynet_target_device.type == "cuda"
            )

        elif self.spynet_target_device.type != "cuda":
            print(
                "Skipping torch.compile for SPyNetModified as spynet_target_device is not CUDA."
            )
            self.spynet_m_is_compiled_and_cuda = False
        else:  # hasattr(torch, "compile") is False
            print(
                "Skipping torch.compile for SPyNetModified: torch.compile not available (likely PyTorch < 2.0)."
            )
            self.spynet_m_is_compiled_and_cuda = False

        self.to_tensor = T.ToTensor()
        self.perspective_distorter = T.RandomPerspective(distortion_scale=0.3, p=1.0)

    def _load_vimeo_sequences(self):
        # Assumes vimeo_dir is the root of vimeo_septuplet (contains sep_testlist.txt etc.)
        # And sequences are in vimeo_dir/sequences/
        # Adjust to use the correct list file based on is_train
        list_file_name = "sep_testlist.txt"
        # Correctly navigate to parent of 'sequences' directory to find list files
        vimeo_list_file_containing_dir = os.path.dirname(
            self.vimeo_root_dir.rstrip("/\ ")
        )
        vimeo_list_file = os.path.join(vimeo_list_file_containing_dir, list_file_name)

        sequence_base_path = (
            self.vimeo_root_dir
        )  # This should be path to 'sequences' folder

        if not os.path.exists(vimeo_list_file):
            # Try another common location if the first fails (e.g. vimeo_septuplet/sep_testlist.txt)
            vimeo_septuplet_root = os.path.dirname(vimeo_list_file_containing_dir)
            vimeo_list_file_alt = os.path.join(vimeo_septuplet_root, list_file_name)
            if os.path.exists(vimeo_list_file_alt):
                vimeo_list_file = vimeo_list_file_alt
            else:
                raise FileNotFoundError(
                    f"Vimeo list file not found at {vimeo_list_file} or {vimeo_list_file_alt}"
                )

        vimeo_sequences = []
        with open(vimeo_list_file, "r") as f:
            for line in f:
                seq_folder_path = os.path.join(sequence_base_path, line.strip())
                frames_in_seq = sorted(
                    glob.glob(os.path.join(seq_folder_path, "im*.png"))
                )
                if len(frames_in_seq) >= self.k_frames:
                    vimeo_sequences.append(frames_in_seq)
        return vimeo_sequences

    def _load_defencing_items(self):
        set_type_folder = (
            "Training Set" if self.is_train else "Test Set"
        )  # Paper might use specific splits
        # The paper implies DeFencing dataset might not have standard train/test splits used this way
        # For now, assume Training_Images/Labels are always used for fence assets.
        # This might need adjustment based on how DeFencing is structured for this project.
        img_dir_name = "Training_Images" if self.is_train else "Test_Images"
        mask_dir_name = "Training_Labels" if self.is_train else "Test_Labels"

        img_dir = os.path.join(self.defencing_root_dir, set_type_folder, img_dir_name)
        mask_dir = os.path.join(self.defencing_root_dir, set_type_folder, mask_dir_name)

        # Fallback if Test Set structure is not present, try to use Training Set for assets
        if not os.path.isdir(img_dir) and not self.is_train:
            print(
                f"Warning: Test set image directory {img_dir} not found. Trying Training Set for fence assets."
            )
            img_dir = os.path.join(
                self.defencing_root_dir, "Training Set", img_dir_name
            )
            mask_dir = os.path.join(
                self.defencing_root_dir, "Training Set", mask_dir_name
            )

        fence_items = []
        if not os.path.isdir(img_dir):
            print(f"Warning: Fence image directory not found: {img_dir}")
            return []
        if not os.path.isdir(mask_dir):
            print(f"Warning: Fence mask directory not found: {mask_dir}")
            return []

        fence_img_names = sorted(os.listdir(img_dir))
        for name in fence_img_names:
            base_name, ext = os.path.splitext(name)
            potential_mask_names = [
                base_name + ".png",
                base_name + ".jpg",
                base_name + ".jpeg",
                name,
            ]
            found_mask_path = None
            for m_name in potential_mask_names:
                p = os.path.join(mask_dir, m_name)
                if os.path.exists(p):
                    found_mask_path = p
                    break
            if found_mask_path:
                fence_items.append(
                    {"img": os.path.join(img_dir, name), "mask": found_mask_path}
                )
            # else:
            # print(f"Warning: Mask not found for fence image {name} in {mask_dir}")
        return fence_items

    def __len__(self):
        # If using a subset for testing/debugging
        if (
            hasattr(self.config, "subset_fraction")
            and self.config.subset_fraction is not None
            and self.config.subset_fraction < 1.0
        ):
            return int(len(self.vimeo_sequences_paths) * self.config.subset_fraction)
        if hasattr(self.config, "num_samples") and self.config.num_samples is not None:
            return min(self.config.num_samples, len(self.vimeo_sequences_paths))
        return len(self.vimeo_sequences_paths)

    def __getitem__(self, idx):
        # Adjust index if using subset and shuffle is true for dataloader (though idx should be managed by sampler)
        # This __getitem__ assumes idx is valid for the (potentially subset) dataset length

        if not self.fence_items and self.is_train:
            raise RuntimeError(
                "No fence items loaded, which are required for training item generation."
            )

        selected_vimeo_frame_paths = self.vimeo_sequences_paths[
            idx % len(self.vimeo_sequences_paths)
        ]  # Use modulo for safety with subset logic
        start_frame_idx = random.randint(
            0, len(selected_vimeo_frame_paths) - self.k_frames
        )
        burst_frame_paths = selected_vimeo_frame_paths[
            start_frame_idx : start_frame_idx + self.k_frames
        ]
        bg_frames_pil = [Image.open(p).convert("RGB") for p in burst_frame_paths]

        bg_frames_aug_pil = augment_background_burst(
            bg_frames_pil, self.img_height, self.img_width
        )
        B_j_tensors = [self.to_tensor(frame) for frame in bg_frames_aug_pil]

        keyframe_idx_in_burst = self.k_frames // 2
        B_k_tensor = B_j_tensors[keyframe_idx_in_burst]

        # Only perform fence operations if fence_items are available (e.g. for training)
        if self.fence_items:
            selected_fence_item = random.choice(self.fence_items)
            F_raw_pil = Image.open(selected_fence_item["img"]).convert("RGB")
            M_raw_fence_pil = Image.open(selected_fence_item["mask"]).convert("L")

            augmented_fences_pil = augment_fence_for_burst(
                F_raw_pil,
                M_raw_fence_pil,
                self.k_frames,
                self.img_height,
                self.img_width,
                self.perspective_distorter,
            )
        else:  # If no fences (e.g. evaluation on clean data or if fence loading failed for non-train)
            # Create dummy/empty fence structures if necessary for pipeline consistency
            # This branch needs to ensure S_j_tensors and I_j_tensors are still validly formed
            # For now, let's assume this is primarily for training where fences are required.
            # If used for eval without fences, the logic might need to adjust what it returns or how S_j is handled.
            pass  # This implies an issue if fences were expected

        I_j_tensors = []
        S_j_tensors = []  # Fence masks (S_k in paper)

        for i in range(self.k_frames):
            B_j_i_tensor = B_j_tensors[i]
            if self.fence_items:
                F_prime_pil, M_prime_pil = augmented_fences_pil[i]
                F_prime_tensor = self.to_tensor(F_prime_pil)
                M_prime_tensor = (
                    self.to_tensor(M_prime_pil) > 0.5
                ).float()  # Binarized S_j
                I_j_i_tensor = (
                    M_prime_tensor * F_prime_tensor
                    + (1 - M_prime_tensor) * B_j_i_tensor
                )  # I_j
            else:  # No fences, I_j is just B_j
                I_j_i_tensor = B_j_i_tensor
                M_prime_tensor = torch.zeros_like(
                    B_j_i_tensor[0:1, :, :]
                )  # Dummy zero mask S_j

            I_j_tensors.append(I_j_i_tensor)
            S_j_tensors.append(M_prime_tensor)

        I_k_tensor = I_j_tensors[
            keyframe_idx_in_burst
        ]  # Keyframe (obstructed if fences applied)
        S_k_tensor = S_j_tensors[keyframe_idx_in_burst]  # Keyframe mask S_k

        # I_k^m = I_k * (1 - S_k) -- masked keyframe background content
        # Ensure I_k_tensor and S_k_tensor are on the target_warp_device for this operation and subsequent concatenation
        # The features for f_in will eventually be moved to the RDN model's device in the training loop.
        # For dataset internal ops, self.spynet_target_device is the primary compute device.
        target_features_device = self.spynet_target_device

        I_k_m_tensor = I_k_tensor.to(target_features_device) * (
            1 - S_k_tensor.to(target_features_device)
        )

        all_features_for_f_in = [
            I_k_m_tensor,  # on target_features_device
            S_k_tensor.to(target_features_device),  # on target_features_device
        ]

        for j in range(self.k_frames):
            if j == keyframe_idx_in_burst:
                continue

            I_j_current_tensor = I_j_tensors[j].to(target_features_device)
            S_j_current_tensor = S_j_tensors[j].to(target_features_device)

            # SPyNetModified inputs [I_k; S_k] and [I_j; S_j]
            # These are already on target_features_device (which is self.spynet_target_device)
            input_k_rgbm = torch.cat(
                [
                    I_k_tensor.to(self.spynet_target_device),
                    S_k_tensor.to(self.spynet_target_device),
                ],
                dim=0,
            )
            input_j_rgbm = torch.cat([I_j_current_tensor, S_j_current_tensor], dim=0)

            with torch.inference_mode():
                f_kj_m_tensor = self.spynet_m(
                    input_k_rgbm.unsqueeze(0), input_j_rgbm.unsqueeze(0)
                ).squeeze(0)
            # f_kj_m_tensor is now on self.spynet_target_device

            # CRITICAL FIX for CUDAGraphs error: Clone the output of the compiled model
            if (
                hasattr(self, "spynet_m_is_compiled_and_cuda")
                and self.spynet_m_is_compiled_and_cuda
            ):
                f_kj_m_tensor = f_kj_m_tensor.clone()

            I_j_m_content_to_warp = I_j_current_tensor * (1 - S_j_current_tensor)

            # Warping inputs (image, mask to warp) and flow must be on the same device for grid_sample.
            # The warp_frame_with_flow utility adapts to the flow_tensor's device.
            # So, I_j_m_content_to_warp and S_j_current_tensor should be on f_kj_m_tensor.device.
            Î_j_m_tensor = warp_frame_with_flow(
                I_j_m_content_to_warp.to(f_kj_m_tensor.device), f_kj_m_tensor
            )
            Š_j_tensor = warp_frame_with_flow(
                S_j_current_tensor.to(f_kj_m_tensor.device), f_kj_m_tensor
            )
            Š_j_tensor = (Š_j_tensor > 0.5).float()

            V_j_tensor = create_validity_mask(
                self.img_height, self.img_width, f_kj_m_tensor
            )
            # V_j_tensor will be on f_kj_m_tensor.device

            all_features_for_f_in.extend(
                [
                    Î_j_m_tensor,  # on f_kj_m_tensor.device (spynet_target_device)
                    Š_j_tensor,  # on f_kj_m_tensor.device
                    V_j_tensor,  # on f_kj_m_tensor.device
                    f_kj_m_tensor,  # on f_kj_m_tensor.device
                ]
            )

        try:
            # Concatenate all features. They should all be on target_features_device (spynet_target_device).
            f_in_tensor = torch.cat(all_features_for_f_in, dim=0)
        except Exception as e:
            print(f"Error concatenating features for f_in (idx: {idx}): {e}")
            for i, feat in enumerate(all_features_for_f_in):
                print(
                    f"Feature {i} shape: {feat.shape}, type: {feat.dtype}, device: {feat.device}"
                )
            raise

        # Expected channels: I_k^m (3), S_k (1), then for each of (K-1) non-keyframes:
        # Î_j^m (3), Š_j (1), V_j (1), f_kj_m (2) -> total 7 per non-keyframe.
        # Total = 3 + 1 + (self.k_frames - 1) * (3 + 1 + 1 + 2) = 4 + (K-1)*7
        expected_f_in_channels = (
            self.config.num_input_channels
        )  # Use config value directly
        if f_in_tensor.shape[0] != expected_f_in_channels:
            print(
                f"WARNING (idx: {idx}): f_in_tensor has {f_in_tensor.shape[0]} channels, expected {expected_f_in_channels} based on config."
            )
            # Recalculate expected based on formula for detailed warning
            calculated_expected = 4 + (self.k_frames - 1) * 7
            if f_in_tensor.shape[0] != calculated_expected:
                print(
                    f"WARNING (idx: {idx}): f_in_tensor also mismatch with calculated {calculated_expected} channels."
                )

        return (
            f_in_tensor.cpu(),  # RDN input - ensure CPU for DataLoader if RDN on different device
            I_k_m_tensor.cpu(),  # Masked keyframe background (I_k^m)
            B_k_tensor.cpu(),  # Clean keyframe background (B_k) - RDN target
        )


def run_basic_dataset_test(config_overrides=None):
    print("--- Running Basic Dataset Test (using configured paths) ---")

    class TestDatasetConfig:
        # Default paths - user should ensure these are valid or override for testing
        vimeo_dir = "data_raw/vimeo_test_clean/sequences"
        defencing_dir = "data_raw/De-fencing-master/dataset"
        # Path to SPyNet weights - CRITICAL, this MUST be valid for test to run
        spynet_m_weights_path = (
            "spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth"
        )
        spynet_base_model_name = "sintel-final"

        img_width = 80
        img_height = 48
        k_frames = 3
        num_input_channels = 4 + (k_frames - 1) * 7
        num_output_channels = 3
        device = "cuda"
        subset_fraction = None
        num_samples = 2  # Test with a small number of samples

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            # Recalculate if k_frames changed via kwargs
            self.num_input_channels = 4 + (self.k_frames - 1) * 7

    config_dict = {}
    if config_overrides:
        config_dict.update(config_overrides)

    test_config = TestDatasetConfig(**config_dict)

    print(
        f"Using test config: Vimeo: '{test_config.vimeo_dir}', DeFencing: '{test_config.defencing_dir}', SPyNet: '{test_config.spynet_m_weights_path}'"
    )

    if (
        not test_config.spynet_m_weights_path
        or test_config.spynet_m_weights_path == "path/to/your/spynet_m_weights.pth"
        or not os.path.exists(test_config.spynet_m_weights_path)
    ):
        print(
            f"CRITICAL ERROR: spynet_m_weights_path ('{test_config.spynet_m_weights_path}') is not valid or not set. Dataset test cannot proceed."
        )
        print(
            "Please provide a valid path to SPyNetModified weights in TestDatasetConfig or via overrides."
        )
        return

    try:
        print("\nAttempting to initialize InpaintingDataset...")
        dataset = InpaintingDataset(
            config=test_config,
            spynet_model_path=test_config.spynet_m_weights_path,
            spynet_model_name_for_gt_flow_in_spynet_m=test_config.spynet_base_model_name,
            is_train=True,
            spynet_device=test_config.device,
        )
        print(f"InpaintingDataset initialized. Length: {len(dataset)}")

        if len(dataset) == 0:
            print(
                "WARNING: Dataset is empty. This might be due to incorrect paths, empty dataset, or subset parameters."
            )
            print(
                "Check provided vimeo_dir and defencing_dir, and subset_fraction/num_samples."
            )
            return  # Cant test __getitem__ if empty

        print("\nAttempting to retrieve an item from the dataset...")
        f_in, i_k_m, b_k = dataset[0]
        print(f"Successfully retrieved item 0 from dataset.")
        print(f"  f_in shape: {f_in.shape}, dtype: {f_in.dtype}")
        print(f"  i_k_m shape: {i_k_m.shape}, dtype: {i_k_m.dtype}")
        print(f"  b_k shape: {b_k.shape}, dtype: {b_k.dtype}")

        assert f_in.shape == (
            test_config.num_input_channels,
            test_config.img_height,
            test_config.img_width,
        ), "f_in shape mismatch"
        assert i_k_m.shape == (
            test_config.num_output_channels,
            test_config.img_height,
            test_config.img_width,
        ), "i_k_m shape mismatch"
        assert b_k.shape == (
            test_config.num_output_channels,
            test_config.img_height,
            test_config.img_width,
        ), "b_k shape mismatch"
        print("Dataset item shapes are correct.")

    except FileNotFoundError as fnf_e:
        print(f"ERROR during dataset test (FileNotFoundError): {fnf_e}")
        print(
            "Please ensure dataset paths in TestDatasetConfig are correct and data exists."
        )
    except ValueError as val_e:
        print(f"ERROR during dataset test (ValueError): {val_e}")
        print(
            "This might be due to missing sequences or fence items based on the provided paths."
        )
    except RuntimeError as rt_e:
        print(f"ERROR during dataset test (RuntimeError): {rt_e}")
        print(
            "This could be an issue with SPyNet loading or tensor operations within the dataset."
        )
    except Exception as e:
        print(f"An unexpected error occurred during InpaintingDataset test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Dataset Test Finished ---")


if __name__ == "__main__":
    # Example of how to run with overridden paths if defaults are not set up:
    # user_test_params = {
    #     "vimeo_dir": "/path/to/your/vimeo/sequences",
    #     "defencing_dir": "/path/to/your/defencing_dataset/dataset",
    #     "spynet_m_weights_path": "/path/to/your/actual/spynet_m_weights.pth",
    #     "num_samples": 5
    # }
    # run_basic_dataset_test(config_overrides=user_test_params)

    # Run with default config (expects data at default paths & valid spynet_m_weights_path in TestDatasetConfig)
    print(
        "Running dataset test with default config. Ensure paths are set correctly in TestDatasetConfig or provide overrides."
    )
    run_basic_dataset_test()
