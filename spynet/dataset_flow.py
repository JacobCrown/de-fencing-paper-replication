# spynet/dataset_flow.py
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import math

# Zaimportuj augmentacje
from augmentations import (
    pv_augmentation_flow,
    IMG_HEIGHT,
    IMG_WIDTH,
)  # Importujemy też IMG_HEIGHT, IMG_WIDTH

# Zaimportuj oryginalną sieć SPyNet do generowania GT flow
try:
    from spynet_original import Network as VanillaSPyNet
    import spynet_original as original_spynet_module  # Do ustawienia args_strModel
except ImportError:
    print(
        "BŁĄD KRYTYCZNY: Nie można zaimportować `Network` lub modułu z `spynet_original.py`!"
    )
    raise


class FlowDataset(Dataset):
    def __init__(
        self,
        vimeo_clean_test_dir,
        defencing_dir,
        vanilla_spynet_model_name="sintel-final",
        sequence_length=2,
    ):
        """
        Args:
            vimeo_clean_test_dir (str): Ścieżka do głównego katalogu Vimeo Septuplet Clean Testing Set
                                       (np. .../vimeo_test_clean).
            defencing_dir (str): Ścieżka do głównego katalogu De-fencing dataset.
            vanilla_spynet_model_name (str): Nazwa modelu dla oryginalnego SPyNet.
            sequence_length (int): Długość sekwencji klatek do wzięcia (dla par to 2).
        """
        super().__init__()
        self.vimeo_sequences_root = os.path.join(vimeo_clean_test_dir, "sequences")
        self.defencing_dir = defencing_dir

        # Dostosuj ścieżki do struktury twojego De-fencing dataset
        # Przykładowa struktura, którą zakładaliśmy wcześniej:
        self.defencing_img_dir = os.path.join(
            defencing_dir, "Training Set", "Training_Images"
        )
        self.defencing_mask_dir = os.path.join(
            defencing_dir, "Training Set", "Training_Labels"
        )
        # Jeśli struktura De-fencing jest inna, zmień powyższe.

        self.sequence_length = sequence_length
        self.train_frame_height = IMG_HEIGHT  # Użyj zaimportowanych stałych
        self.train_frame_width = IMG_WIDTH  # Użyj zaimportowanych stałych

        # Inicjalizuj oryginalny SPyNet do GT Flow
        original_spynet_module.args_strModel = vanilla_spynet_model_name

        self.vanilla_spynet = VanillaSPyNet()
        self.vanilla_spynet.eval()
        if torch.cuda.is_available():
            self.vanilla_spynet = self.vanilla_spynet.cuda()

        # Wczytaj listę sekwencji z sep_testlist.txt
        vimeo_list_file = os.path.join(vimeo_clean_test_dir, "sep_testlist.txt")
        if not os.path.exists(vimeo_list_file):
            raise FileNotFoundError(f"Vimeo list file not found: {vimeo_list_file}")

        self.vimeo_sequences_paths = []  # Przechowuje ścieżki względne np. '00001/0001'
        with open(vimeo_list_file, "r") as f:
            for line in f:
                self.vimeo_sequences_paths.append(line.strip())

        # Wczytaj listę plików płotów
        # breakpoint()
        if not os.path.isdir(self.defencing_img_dir):
            raise NotADirectoryError(
                f"Katalog obrazów płotów nie istnieje: {self.defencing_img_dir}"
            )
        self.fence_images_names = sorted(
            [
                f
                for f in os.listdir(self.defencing_img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if not self.vimeo_sequences_paths:
            raise ValueError("No Vimeo sequences found from sep_testlist.txt.")
        if not self.fence_images_names:
            raise ValueError(f"No fence images found in {self.defencing_img_dir}.")

        print(
            f"FlowDataset: Loaded {len(self.vimeo_sequences_paths)} Vimeo sequence paths from '{os.path.basename(vimeo_list_file)}'."
        )
        print(
            f"FlowDataset: Loaded {len(self.fence_images_names)} fence structures from '{os.path.basename(self.defencing_img_dir)}'."
        )
        print(
            f"FlowDataset: Vanilla SPyNet for GT flow will use model '{vanilla_spynet_model_name}'."
        )

    def __len__(self):
        return len(self.vimeo_sequences_paths)

    def __getitem__(self, index):
        # 1. Wybierz sekwencję Vimeo i parę klatek
        relative_seq_path = self.vimeo_sequences_paths[index]  # np. '00001/0001'

        # Septuplet ma 7 klatek (im1.png do im7.png)
        frame_numbers = list(range(1, 8))
        if len(frame_numbers) < self.sequence_length:
            # To nie powinno się zdarzyć dla septupletów i sequence_length=2
            start_idx_num = 0
            selected_frame_numbers = frame_numbers + [frame_numbers[-1]] * (
                self.sequence_length - len(frame_numbers)
            )
        else:
            start_idx_num = random.randint(0, len(frame_numbers) - self.sequence_length)
            selected_frame_numbers = frame_numbers[
                start_idx_num : start_idx_num + self.sequence_length
            ]

        # Pełne ścieżki do klatek tła
        bg_img1_path = os.path.join(
            self.vimeo_sequences_root,
            relative_seq_path,
            f"im{selected_frame_numbers[0]}.png",
        )
        bg_img2_path = os.path.join(
            self.vimeo_sequences_root,
            relative_seq_path,
            f"im{selected_frame_numbers[1]}.png",
        )

        try:
            bg_img1_pil = Image.open(bg_img1_path).convert("RGB")
            bg_img2_pil = Image.open(bg_img2_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing BG frames for {relative_seq_path}: {bg_img1_path} or {bg_img2_path}"
            )

        # 2. Wybierz losowy płot i jego maskę
        fence_file_name = random.choice(self.fence_images_names)
        fence_img_path = os.path.join(self.defencing_img_dir, fence_file_name)

        base_fence_name, _ = os.path.splitext(fence_file_name)
        fence_mask_path = None
        # Spróbuj znaleźć maskę z taką samą nazwą bazową i popularnymi rozszerzeniami
        for ext in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
        ]:  # Maska może mieć inne rozszerzenie niż obraz
            potential_mask_name = base_fence_name + ext
            p = os.path.join(self.defencing_mask_dir, potential_mask_name)
            if os.path.exists(p):
                fence_mask_path = p
                break
        if (
            fence_mask_path is None
        ):  # Jeśli nie znaleziono, spróbuj oryginalną nazwę pliku obrazu płotu
            p_orig_name = os.path.join(self.defencing_mask_dir, fence_file_name)
            if os.path.exists(p_orig_name):
                fence_mask_path = p_orig_name
            else:
                raise FileNotFoundError(
                    f"Mask not found for fence '{fence_file_name}' in '{self.defencing_mask_dir}'. Tried base name + extensions and original name."
                )

        fence_img_pil = Image.open(fence_img_path).convert("RGB")
        fence_mask_pil = Image.open(fence_mask_path).convert(
            "L"
        )  # Maska jako skala szarości

        # 3. Zastosuj augmentacje
        bg1_aug_pil, bg2_aug_pil, fence_aug_pil, fence_mask_aug_pil = (
            pv_augmentation_flow(
                bg_img1_pil, bg_img2_pil, fence_img_pil, fence_mask_pil
            )
        )

        # Konwersja do tensorów
        bg1_aug_tensor = TF.to_tensor(bg1_aug_pil)
        bg2_aug_tensor = TF.to_tensor(bg2_aug_pil)
        fence_aug_tensor = TF.to_tensor(fence_aug_pil)
        fence_mask_aug_tensor = TF.to_tensor(fence_mask_aug_pil)
        fence_mask_aug_tensor = (
            fence_mask_aug_tensor > 0.5
        ).float()  # Upewnij się, że maska jest binarna [0,1]

        # 4. Nałóż płot na zaaugmentowane tła
        obstructed_img1 = (
            fence_mask_aug_tensor * fence_aug_tensor
            + (1 - fence_mask_aug_tensor) * bg1_aug_tensor
        )
        obstructed_img2 = (
            fence_mask_aug_tensor * fence_aug_tensor
            + (1 - fence_mask_aug_tensor) * bg2_aug_tensor
        )

        # Przygotuj wejście RGBM dla SPyNetModified
        input1_rgbm = torch.cat([obstructed_img1, fence_mask_aug_tensor], dim=0)
        input2_rgbm = torch.cat([obstructed_img2, fence_mask_aug_tensor], dim=0)

        # 5. Oblicz Ground Truth Flow
        gt_flow = torch.zeros(
            2, self.train_frame_height, self.train_frame_width
        )  # Domyślnie
        with torch.inference_mode():
            input_gt_flow_1 = bg1_aug_tensor.unsqueeze(0)
            input_gt_flow_2 = bg2_aug_tensor.unsqueeze(0)

            if torch.cuda.is_available():
                input_gt_flow_1 = input_gt_flow_1.cuda()
                input_gt_flow_2 = input_gt_flow_2.cuda()

            flow_output_vanilla = self.vanilla_spynet(input_gt_flow_1, input_gt_flow_2)
            gt_flow = flow_output_vanilla.squeeze(0).cpu()

        return {
            "input1_rgbm": input1_rgbm,
            "input2_rgbm": input2_rgbm,
            "gt_flow": gt_flow,
        }


if __name__ == "__main__":
    print(
        "Testing FlowDataset with spynet_original.py for GT Flow (Vimeo Clean Test Set structure)..."
    )

    # --- Zaktualizuj te ścieżki! ---
    VIMEO_CLEAN_TEST_ROOT_DIR = (
        "data_raw/vimeo_test_clean"  # Główny katalog vimeo_test_clean
    )
    DEFENCING_ROOT_DIR = (
        "data_raw/De-fencing-master/dataset"  # Główny katalog DeFencing
    )
    VANILLA_SPYNET_MODEL_NAME = "sintel-final"
    # --- Koniec aktualizacji ścieżek ---

    if "path/to/" in VIMEO_CLEAN_TEST_ROOT_DIR or "path/to/" in DEFENCING_ROOT_DIR:
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! Zaktualizuj ścieżki VIMEO_CLEAN_TEST_ROOT_DIR i DEFENCING_ROOT_DIR   !!!"
        )
        print(
            "!!! w kodzie, aby uruchomić testy datasetu.                             !!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        exit()

    try:
        dataset = FlowDataset(
            vimeo_clean_test_dir=VIMEO_CLEAN_TEST_ROOT_DIR,
            defencing_dir=DEFENCING_ROOT_DIR,
            vanilla_spynet_model_name=VANILLA_SPYNET_MODEL_NAME,
        )

        if len(dataset) > 0:
            print(f"Dataset created successfully. Number of samples: {len(dataset)}")

            # Przetestuj kilka przykładów
            for i in range(min(3, len(dataset))):
                print(f"\n--- Sample {i} ---")
                sample = dataset[
                    random.randint(0, len(dataset) - 1)
                ]  # Pobierz losowy przykład

                print("Sample keys:", sample.keys())
                print("Input1 RGBM shape:", sample["input1_rgbm"].shape)
                print("Input2 RGBM shape:", sample["input2_rgbm"].shape)
                print("GT Flow shape:", sample["gt_flow"].shape)
                print(
                    f"GT Flow min: {sample['gt_flow'].min():.4f}, max: {sample['gt_flow'].max():.4f}, mean: {sample['gt_flow'].mean():.4f}"
                )

                if i == 0:  # Zapisz tylko pierwszy przykład
                    try:
                        from torchvision.utils import save_image

                        save_image(
                            sample["input1_rgbm"][0:3, :, :],
                            "test/test_ds_vct_input1_rgb.png",
                        )
                        save_image(
                            sample["input1_rgbm"][3:4, :, :],
                            "test/test_ds_vct_input1_mask.png",
                        )

                        flow_viz_u = sample["gt_flow"][0, :, :].unsqueeze(0)
                        flow_viz_v = sample["gt_flow"][1, :, :].unsqueeze(0)
                        flow_viz_u_norm = (flow_viz_u - flow_viz_u.min()) / (
                            flow_viz_u.max() - flow_viz_u.min() + 1e-6
                        )
                        flow_viz_v_norm = (flow_viz_v - flow_viz_v.min()) / (
                            flow_viz_v.max() - flow_viz_v.min() + 1e-6
                        )
                        save_image(flow_viz_u_norm, "test/test_ds_vct_gt_flow_u.png")
                        save_image(flow_viz_v_norm, "test/test_ds_vct_gt_flow_v.png")
                        print(
                            "\nPrzykładowe obrazy i przepływ (pierwszy przykład) zapisane jako test_ds_vct_*.png"
                        )
                    except ImportError:
                        print("\nInstall torchvision to save sample images.")
                    except Exception as e_save:
                        print(f"\nError saving sample images: {e_save}")
        else:
            print("Dataset jest pusty. Sprawdź ścieżki i zawartość plików list.")

    except FileNotFoundError as e:
        print(f"BŁĄD PLIKU: {e}")
    except ValueError as e:
        print(f"BŁĄD WARTOŚCI: {e}")
    except Exception as e:
        import traceback

        print(f"INNY BŁĄD PODCZAS TESTOWANIA DATASETU: {e}")
        traceback.print_exc()
