import os
import torch
from torch.utils.data import Dataset
import glob
import json


class PrecomputedRDNDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir (str): Directory with all the precomputed .pt files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sample_files = sorted(glob.glob(os.path.join(data_dir, "sample_*.pt")))

        if not self.sample_files:
            raise FileNotFoundError(f"No .pt sample files found in {data_dir}.")

        # Optionally, load a generation config if it exists to verify parameters
        # or store some metadata, but not strictly required for basic loading.
        gen_config_path = os.path.join(
            os.path.dirname(data_dir), "generation_config.json"
        )
        self.generation_config = None
        if os.path.exists(gen_config_path):
            try:
                with open(gen_config_path, "r") as f:
                    self.generation_config = json.load(f)
                print(f"Loaded generation config from {gen_config_path}")
            except Exception as e:
                print(f"Warning: Could not load or parse generation_config.json: {e}")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.item())

        sample_path = self.sample_files[idx]

        try:
            sample_data = torch.load(sample_path)
        except Exception as e:
            print(f"Error loading sample file {sample_path}: {e}")
            # Return a dummy sample or raise error, depending on desired robustness
            # For now, let's re-raise to be aware of corrupted/problematic files
            raise

        f_in = sample_data["f_in"]
        i_k_m = sample_data["i_k_m"]
        b_k = sample_data["b_k"]

        sample = {"f_in": f_in, "i_k_m": i_k_m, "b_k": b_k}

        if self.transform:
            sample = self.transform(
                sample
            )  # Note: transform needs to handle dict of tensors

        return sample["f_in"], sample["i_k_m"], sample["b_k"]


if __name__ == "__main__":
    # Example usage (assuming you have run generate_precomputed_rdn_data.py first)
    print("Testing PrecomputedRDNDataset...")
    # Create dummy data for testing if it doesn't exist
    dummy_data_root = "data_precomputed/rdn_data"
    dummy_train_dir = os.path.join(dummy_data_root, "train")
    dummy_gen_config_path = os.path.join(dummy_data_root, "generation_config.json")

    if not os.path.exists(os.path.join(dummy_train_dir, "sample_000000.pt")):
        print(
            f"Dummy data not found in {dummy_train_dir}. Creating some for test purposes."
        )
        os.makedirs(dummy_train_dir, exist_ok=True)
        # Create a dummy generation_config.json
        dummy_config_content = {
            "k_frames": 5,
            "img_width": 32,
            "img_height": 32,
            "num_input_channels": 4 + (5 - 1) * 7,
        }
        with open(dummy_gen_config_path, "w") as f_cfg:
            json.dump(dummy_config_content, f_cfg)

        # Create a few dummy sample files
        num_dummy_samples = 3
        k_frames_dummy = dummy_config_content["k_frames"]
        h_dummy, w_dummy = (
            dummy_config_content["img_height"],
            dummy_config_content["img_width"],
        )
        c_fin_dummy = dummy_config_content["num_input_channels"]
        c_img_dummy = 3

        for i in range(num_dummy_samples):
            dummy_f_in = torch.randn(c_fin_dummy, h_dummy, w_dummy)
            dummy_i_k_m = torch.randn(c_img_dummy, h_dummy, w_dummy)
            dummy_b_k = torch.randn(c_img_dummy, h_dummy, w_dummy)
            torch.save(
                {"f_in": dummy_f_in, "i_k_m": dummy_i_k_m, "b_k": dummy_b_k},
                os.path.join(dummy_train_dir, f"sample_{i:06d}.pt"),
            )
        print(f"Created {num_dummy_samples} dummy samples in {dummy_train_dir}")

    if not os.path.exists(dummy_train_dir) or not os.listdir(dummy_train_dir):
        print(
            f"Cannot run test: Precomputed data directory {dummy_train_dir} is empty or does not exist."
        )
        print(
            "Please run generate_precomputed_rdn_data.py first or ensure dummy data creation works."
        )
    else:
        try:
            dataset = PrecomputedRDNDataset(data_dir=dummy_train_dir)
            print(f"Dataset loaded. Number of samples: {len(dataset)}")

            if len(dataset) > 0:
                f_in_sample, i_k_m_sample, b_k_sample = dataset[0]
                print("Successfully retrieved sample 0:")
                print(f"  f_in shape: {f_in_sample.shape}, dtype: {f_in_sample.dtype}")
                print(
                    f"  i_k_m shape: {i_k_m_sample.shape}, dtype: {i_k_m_sample.dtype}"
                )
                print(f"  b_k shape: {b_k_sample.shape}, dtype: {b_k_sample.dtype}")

                # Verify with generation config if loaded
                if dataset.generation_config:
                    assert f_in_sample.shape[0] == dataset.generation_config.get(
                        "num_input_channels"
                    ), "f_in channel mismatch with gen_config"
                    assert i_k_m_sample.shape[1] == dataset.generation_config.get(
                        "img_height"
                    ), "i_k_m height mismatch with gen_config"
                    assert i_k_m_sample.shape[2] == dataset.generation_config.get(
                        "img_width"
                    ), "i_k_m width mismatch with gen_config"
                    print(
                        "Sample shapes consistent with loaded generation_config.json (if found)."
                    )

            # Test with DataLoader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=True
            )
            for batch_num, (f_in_b, i_k_m_b, b_k_b) in enumerate(dataloader):
                print(f"Batch {batch_num + 1}:")
                print(f"  f_in_batch shape: {f_in_b.shape}")
                print(f"  i_k_m_batch shape: {i_k_m_b.shape}")
                print(f"  b_k_batch shape: {b_k_b.shape}")
                if batch_num >= 1:  # Show a couple of batches
                    break
            print("DataLoader test successful.")

        except Exception as e:
            print(f"Error during PrecomputedRDNDataset test: {e}")
            import traceback

            traceback.print_exc()
