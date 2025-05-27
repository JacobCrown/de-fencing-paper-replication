# spynet/visualize_batch_data.py
import os
import sys
import random
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Dodaj katalog nadrzędny do sys.path, aby umożliwić importy z pakietu spynet
# To jest potrzebne, jeśli uruchamiasz ten skrypt bezpośrednio spoza głównego katalogu projektu
# lub jeśli struktura importu tego wymaga.
# Lepszym podejściem jest uruchamianie z `python -m spynet.visualize_batch_data`
# z katalogu nadrzędnego, wtedy ten blok nie jest konieczny.
if __name__ == "__main__":
    # Upewnij się, że katalog nadrzędny (zawierający pakiet 'spynet') jest w sys.path
    # To pozwala na `from spynet.module import ...`
    current_dir = os.path.dirname(os.path.abspath(__file__))  # spynet/
    project_root = os.path.dirname(current_dir)  # Defencing_paper_replication/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Importy z naszego modułu spynet (teraz powinny działać)
from spynet.dataset_flow import FlowDataset


# --- Konfiguracja Wizualizacji ---
class VisualizeConfig:
    # Ścieżki do danych (MUSISZ JE ZAKTUALIZOWAĆ!)
    VIMEO_CLEAN_TEST_DIR = "data_raw/vimeo_test_clean"
    DEFENCING_DIR = "data_raw/De-fencing-master/dataset"

    # Nazwa modelu dla oryginalnego SPyNet (używanego w FlowDataset do GT)
    VANILLA_SPYNET_MODEL_NAME = "sintel-final"  # lub 'chairs-final' itp.

    # Parametry DataLoader
    BATCH_SIZE = 4  # Liczba próbek do załadowania i zwizualizowania
    NUM_WORKERS = (
        0  # Dla prostoty wizualizacji i unikania problemów z multiprocessing na Windows
    )
    # Ustaw na 0, aby DataLoader działał w głównym procesie.

    # Ile próbek z batcha faktycznie pokazać na wykresie
    NUM_SAMPLES_TO_SHOW = 4


def visualize_training_batch(batch_data, num_samples_to_show=4, title_prefix=""):
    """
    Wizualizuje kilka przykładów z batcha danych treningowych dla SPyNetModified.
    Pokazuje: input1_rgb, input1_mask, input2_rgb, input2_mask, gt_flow (u, v)
    """
    input1_rgbm = batch_data["input1_rgbm"].cpu()  # [B, 4, H, W]
    input2_rgbm = batch_data["input2_rgbm"].cpu()  # [B, 4, H, W]
    gt_flow = batch_data["gt_flow"].cpu()  # [B, 2, H, W]

    actual_batch_size = input1_rgbm.size(0)
    num_to_show = min(num_samples_to_show, actual_batch_size)

    if num_to_show == 0:
        print("Brak przykładów do wizualizacji w tym batchu.")
        return

    # Wyodrębnij komponenty
    input1_rgb = input1_rgbm[:num_to_show, 0:3, :, :]
    input1_mask = input1_rgbm[:num_to_show, 3:4, :, :]
    input2_rgb = input2_rgbm[:num_to_show, 0:3, :, :]
    input2_mask = input2_rgbm[:num_to_show, 3:4, :, :]
    gt_flow_u = gt_flow[:num_to_show, 0:1, :, :]
    gt_flow_v = gt_flow[:num_to_show, 1:2, :, :]

    def normalize_flow_component_for_viz(flow_comp_batch):
        normalized_list = []
        for i in range(flow_comp_batch.size(0)):
            comp = flow_comp_batch[i]  # [1, H, W]
            min_val = comp.min()
            max_val = comp.max()
            if (max_val - min_val).abs() > 1e-6:
                normalized = (comp - min_val) / (max_val - min_val)
            else:
                normalized = torch.zeros_like(comp)
            normalized_list.append(normalized.repeat(3, 1, 1))  # Powiel na 3 kanały
        return torch.stack(normalized_list) if normalized_list else torch.empty(0)

    gt_flow_u_viz = normalize_flow_component_for_viz(gt_flow_u)
    gt_flow_v_viz = normalize_flow_component_for_viz(gt_flow_v)

    # Przygotuj listę obrazów do wyświetlenia w siatce
    # Każdy wiersz w siatce będzie reprezentował jeden przykład
    # Kolumny: Input1 RGB, Input1 Mask, Input2 RGB, Input2 Mask, GT Flow U, GT Flow V

    # Tworzymy listę wszystkich tensorów do wyświetlenia
    display_tensors = []
    for i in range(num_to_show):
        display_tensors.append(input1_rgb[i])
        display_tensors.append(input1_mask[i].repeat(3, 1, 1))  # Maska jako RGB
        display_tensors.append(input2_rgb[i])
        display_tensors.append(input2_mask[i].repeat(3, 1, 1))  # Maska jako RGB
        if gt_flow_u_viz.numel() > 0:
            display_tensors.append(gt_flow_u_viz[i])  # Przepływ U jako RGB
        if gt_flow_v_viz.numel() > 0:
            display_tensors.append(gt_flow_v_viz[i])  # Przepływ V jako RGB

    if not display_tensors:
        print("Brak tensorów do wyświetlenia.")
        return

    # Ile obrazów na jeden przykład (sample)
    images_per_sample = 6
    grid = make_grid(
        display_tensors, nrow=images_per_sample, padding=5, normalize=False
    )

    plt.figure(figsize=(images_per_sample * 3, num_to_show * 3))  # Dostosuj rozmiar
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")

    # Dodaj tytuły nad kolumnami (uproszczone)
    column_titles = [
        "Input 1 (RGB+Płot)",
        "Maska Płotu 1",
        "Input 2 (RGB+Płot)",
        "Maska Płotu 2",
        "GT Flow (U)",
        "GT Flow (V)",
    ]
    title_str = " | ".join(column_titles)

    plt.title(
        f"{title_prefix}{title_str}\n(Każdy wiersz to jeden przykład z batcha)",
        fontsize=10,
    )
    plt.show()


def main():
    config = VisualizeConfig()

    if "path/to/" in config.VIMEO_CLEAN_TEST_DIR or "path/to/" in config.DEFENCING_DIR:
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! Zaktualizuj ścieżki VIMEO_CLEAN_TEST_DIR i DEFENCING_DIR w klasie     !!!"
        )
        print(
            "!!! VisualizeConfig, aby uruchomić skrypt wizualizacji.                 !!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        return

    print("Inicjalizacja FlowDataset do wizualizacji...")
    try:
        dataset = FlowDataset(
            vimeo_clean_test_dir=config.VIMEO_CLEAN_TEST_DIR,
            defencing_dir=config.DEFENCING_DIR,
            vanilla_spynet_model_name=config.VANILLA_SPYNET_MODEL_NAME,
        )
    except Exception as e:
        import traceback

        print(f"BŁĄD KRYTYCZNY podczas tworzenia datasetu: {e}")
        traceback.print_exc()
        return

    if len(dataset) == 0:
        print("Dataset jest pusty. Nie można zwizualizować danych.")
        return

    print(f"Dataset załadowany. Liczba przykładów: {len(dataset)}")

    # Wybierz losowy indeks startowy, aby za każdym razem widzieć inne dane
    start_index = random.randint(0, max(0, len(dataset) - config.BATCH_SIZE))

    # Możemy stworzyć podzbiór datasetu, aby załadować konkretne przykłady
    # lub po prostu iterować przez DataLoader. Dla prostoty, weźmiemy batch z DataLoader.

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Losowe przykłady przy każdym uruchomieniu
        num_workers=config.NUM_WORKERS,
    )

    print(f"Pobieranie batcha danych (batch_size={config.BATCH_SIZE})...")
    try:
        batch_data = next(iter(loader))
        print("Batch danych pobrany.")
        visualize_training_batch(
            batch_data,
            num_samples_to_show=min(config.NUM_SAMPLES_TO_SHOW, config.BATCH_SIZE),
            title_prefix="Dane Treningowe: ",
        )
    except StopIteration:
        print(
            "DataLoader nie zwrócił żadnych danych (prawdopodobnie dataset jest mniejszy niż batch_size)."
        )
    except Exception as e:
        import traceback

        print(f"BŁĄD podczas pobierania lub wizualizacji batcha: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
