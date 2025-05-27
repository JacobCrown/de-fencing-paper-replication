# spynet/train_flow.py
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime  # Do nazywania plików wag

# Importy z naszego modułu spynet
from spynet_modified import SPyNetModified
from dataset_flow import FlowDataset
# Jeśli będziesz chciał dodać walidację, zaimportuj też odpowiedni dataset


# --- Konfiguracja Treningu ---
class TrainConfig:
    # Ścieżki do danych
    VIMEO_CLEAN_TEST_DIR = "C:/Users/jkoro/Desktop/Studia/Defencing_paper_replication/data_raw/vimeo_test_clean"  # Główny katalog vimeo_test_clean
    DEFENCING_DIR = "C:/Users/jkoro/Desktop/Studia/Defencing_paper_replication/data_raw/De-fencing-master/dataset"  # Główny katalog DeFencing

    # Nazwa modelu dla oryginalnego SPyNet (używanego w FlowDataset do GT)
    VANILLA_SPYNET_MODEL_NAME = "sintel-final"  # lub 'chairs-final' itp.

    # Parametry SPyNetModified
    MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME = (
        "sintel-final"  # Model, z którego ładujemy wagi bazowe dla SPyNetModified
    )
    LOAD_PRETRAINED_FOR_MODIFIED = True  # Czy ładować wagi bazowe do SPyNetModified

    # Parametry treningu
    BATCH_SIZE = 4  # Dostosuj do pamięci GPU
    NUM_EPOCHS = 2  # Docelowo 1000 jak w pracy, ale zacznij od mniejszej liczby
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 4e-5  # Z pracy (4*10^-5)

    # Zapisywanie modelu
    CHECKPOINT_DIR = "./spynet_checkpoints"  # Katalog do zapisywania wag
    SAVE_EVERY_N_EPOCHS = 5  # Co ile epok zapisywać model

    # Inne
    NUM_WORKERS = 4  # Liczba wątków dla DataLoader
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(config):
    print(f"Rozpoczęcie treningu SPyNetModified na urządzeniu: {config.DEVICE}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 1. Model
    print(
        f"Inicjalizacja SPyNetModified (wagi bazowe z: {config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME if config.LOAD_PRETRAINED_FOR_MODIFIED else 'losowe'})..."
    )
    model = SPyNetModified(
        model_name=config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME,
        pretrained=config.LOAD_PRETRAINED_FOR_MODIFIED,
    )
    model.to(config.DEVICE)

    # 2. Dataset i DataLoader
    print("Przygotowywanie datasetu i DataLoader...")
    try:
        train_dataset = FlowDataset(
            vimeo_clean_test_dir=config.VIMEO_CLEAN_TEST_DIR,
            defencing_dir=config.DEFENCING_DIR,
            vanilla_spynet_model_name=config.VANILLA_SPYNET_MODEL_NAME,
        )
    except FileNotFoundError as e:
        print(
            f"BŁĄD KRYTYCZNY: Nie można utworzyć datasetu. Sprawdź ścieżki. Błąd: {e}"
        )
        return
    except Exception as e_ds:
        import traceback

        print(f"BŁĄD KRYTYCZNY podczas tworzenia datasetu: {e_ds}")
        traceback.print_exc()
        return

    if len(train_dataset) == 0:
        print("BŁĄD KRYTYCZNY: Dataset treningowy jest pusty. Przerwanie treningu.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True,  # Odrzuć ostatni niepełny batch
    )
    print(f"Dataset i DataLoader gotowe. Liczba batchy na epokę: {len(train_loader)}")

    # 3. Optymalizator i funkcja straty
    # Praca używa ADAM: lr=1e-4, weight_decay=4e-5, beta1=0.9, beta2=0.999, eps=1e-8
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),  # Domyślne dla Adama w PyTorch, ale można ustawić jawnie
        eps=1e-8,  # Domyślne dla Adama w PyTorch
    )

    # Funkcja straty: L1 loss (Mean Absolute Error)
    criterion = (
        torch.nn.L1Loss()
    )  # Suma w pracy, ale średnia jest bardziej standardowa dla batchy
    # Papier: (1/2N) * Sum(|...|) - Lf w Eq.3 ma 1/2N, N to liczba pikseli
    # L1Loss w PyTorch domyślnie liczy średnią. To powinno być OK.

    # 4. Pętla treningowa
    print("\n--- Rozpoczęcie pętli treningowej ---")
    model.train()  # Ustaw model w tryb treningowy

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        running_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            input1_rgbm = batch_data["input1_rgbm"].to(config.DEVICE)  # [B, 4, H, W]
            input2_rgbm = batch_data["input2_rgbm"].to(config.DEVICE)  # [B, 4, H, W]
            gt_flow = batch_data["gt_flow"].to(config.DEVICE)  # [B, 2, H, W]

            optimizer.zero_grad()

            # Forward pass
            predicted_flow = model(input1_rgbm, input2_rgbm)  # Wynik: [B, 2, H, W]

            # Oblicz stratę
            loss = criterion(predicted_flow, gt_flow)

            # Backward pass i optymalizacja
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 20 == 0:  # Loguj co 20 batchy
                print(
                    f"Epoka [{epoch + 1}/{config.NUM_EPOCHS}], Batch [{i + 1}/{len(train_loader)}], "
                    f"Strata: {loss.item():.4f} (Średnia dotychczas: {running_loss / (i + 1):.4f})"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"--- Koniec Epoki {epoch + 1}/{config.NUM_EPOCHS} ---")
        print(f"Średnia strata: {epoch_loss:.4f}")
        print(f"Czas trwania epoki: {epoch_duration:.2f}s")

        # Zapisywanie modelu
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0 or (
            epoch + 1
        ) == config.NUM_EPOCHS:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_name = f"spynet_modified_epoch{epoch + 1}_{timestamp}.pth"
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model zapisany w: {checkpoint_path}")
        print("-" * 30)

    print("--- Trening zakończony ---")


if __name__ == "__main__":
    # --- Ustaw tutaj ścieżki do swoich danych ---
    # To są tylko przykładowe wartości, MUSISZ je zaktualizować!
    config = TrainConfig()
    config.VIMEO_CLEAN_TEST_DIR = "C:/Users/jkoro/Desktop/Studia/Defencing_paper_replication/data_raw/vimeo_test_clean"
    config.DEFENCING_DIR = "C:/Users/jkoro/Desktop/Studia/Defencing_paper_replication/data_raw/De-fencing-master/dataset"  # Katalog z DeFencingDataset

    # Opcjonalnie: jeśli chcesz zacząć od mniejszej liczby epok do testów
    config.NUM_EPOCHS = 5
    config.BATCH_SIZE = 2  # Zmniejsz, jeśli masz mało pamięci GPU
    config.SAVE_EVERY_N_EPOCHS = 1  # Zapisuj częściej podczas testów

    if "path/to/" in config.VIMEO_CLEAN_TEST_DIR or "path/to/" in config.DEFENCING_DIR:
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print(
            "!!! Zaktualizuj ścieżki VIMEO_CLEAN_TEST_DIR i DEFENCING_DIR w klasie     !!!"
        )
        print(
            "!!! TrainConfig lub w bloku if __name__ == '__main__':                   !!!"
        )
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
    else:
        main(config)
