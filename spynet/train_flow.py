# spynet/train_flow.py
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime  # Do nazywania plików wag
import typing  # Import typing for Optional
import torch.multiprocessing as mp

# Importy z naszego modułu spynet
from spynet_modified import SPyNetModified
from dataset_flow import FlowDataset
# Jeśli będziesz chciał dodać walidację, zaimportuj też odpowiedni dataset


# --- Konfiguracja Treningu ---
class TrainConfig:
    # Ścieżki do danych
    VIMEO_CLEAN_TEST_DIR: str = (
        "data_raw/vimeo_test_clean"  # Główny katalog vimeo_test_clean
    )
    DEFENCING_DIR: str = (
        "data_raw/De-fencing-master/dataset"  # Główny katalog DeFencing
    )

    # Nazwa modelu dla oryginalnego SPyNet (używanego w FlowDataset do GT)
    VANILLA_SPYNET_MODEL_NAME: str = "sintel-final"  # lub 'chairs-final' itp.

    # Parametry SPyNetModified
    MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME: str = (
        "sintel-final"  # Model, z którego ładujemy wagi bazowe dla SPyNetModified
    )
    LOAD_PRETRAINED_FOR_MODIFIED: bool = (
        True  # Czy ładować wagi bazowe do SPyNetModified
    )

    # Parametry treningu
    BATCH_SIZE: int = 64  # Dostosuj do pamięci GPU
    NUM_EPOCHS: int = 1000  # Docelowo 1000 jak w pracy
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 4e-5  # Z pracy (4*10^-5)
    START_EPOCH: int = 0  # Domyślnie zaczynamy od epoki 0

    # Wznawianie treningu
    RESUME_TRAINING: bool = True
    RESUME_CHECKPOINT_PATH: typing.Optional[str] = (
        "spynet_checkpoints/spynet_modified_epoch5_20250527-183441.pth"
    )

    # Zapisywanie modelu
    CHECKPOINT_DIR: str = "./spynet_checkpoints"  # Katalog do zapisywania wag
    SAVE_EVERY_N_EPOCHS: int = 10  # Co ile epok zapisywać model

    # Inne
    NUM_WORKERS: int = 4  # Liczba wątków dla DataLoader
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(config: TrainConfig):
    print(f"Rozpoczęcie treningu SPyNetModified na urządzeniu: {config.DEVICE}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 1. Model
    print(
        f"Inicjalizacja SPyNetModified (wagi bazowe z: {config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME if config.LOAD_PRETRAINED_FOR_MODIFIED and not config.RESUME_TRAINING else 'checkpointu lub losowe'})..."
    )
    model = SPyNetModified(
        model_name=config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME,  # Nazwa bazowego modelu do załadowania wag jeśli pretrained=True
        pretrained=config.LOAD_PRETRAINED_FOR_MODIFIED
        if not config.RESUME_TRAINING
        else False,  # Nie ładuj wag bazowych jeśli wznawiamy z checkpointu
    )
    model.to(config.DEVICE)

    # 2. Optymalizator i funkcja straty
    # Praca używa ADAM: lr=1e-4, weight_decay=4e-5, beta1=0.9, beta2=0.999, eps=1e-8
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = torch.nn.L1Loss()

    # 3. Wznawianie treningu (jeśli dotyczy)
    if config.RESUME_TRAINING:
        if config.RESUME_CHECKPOINT_PATH and os.path.exists(
            config.RESUME_CHECKPOINT_PATH
        ):
            print(f"Wznawianie treningu z checkpointu: {config.RESUME_CHECKPOINT_PATH}")
            try:
                checkpoint = torch.load(
                    config.RESUME_CHECKPOINT_PATH,
                    map_location=config.DEVICE,
                    weights_only=False,
                )  # Set weights_only based on content

                # Check for common key for model state_dict
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:  # Another common pattern
                    model.load_state_dict(checkpoint["state_dict"])
                else:  # Assume checkpoint is the state_dict itself
                    model.load_state_dict(checkpoint)

                # Load optimizer state if available
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                config.START_EPOCH = (
                    checkpoint.get("epoch", config.START_EPOCH) + 1
                )  # Resume from the next epoch
                last_loss = checkpoint.get("loss", float("nan"))
                print(
                    f"Pomyślnie załadowano checkpoint. Model, optymalizator przywrócone. Wznowienie od epoki {config.START_EPOCH}."
                )
                print(
                    f"Strata z ostatniej zapisanej epoki ({config.START_EPOCH - 1}): {last_loss:.4f}"
                )
            except Exception as e:
                print(
                    f"BŁĄD podczas ładowania checkpointu: {e}. Trening rozpocznie się od nowa."
                )
                config.START_EPOCH = 0
                # If resuming failed, and LOAD_PRETRAINED_FOR_MODIFIED was true, we might want to re-load base weights.
                if config.LOAD_PRETRAINED_FOR_MODIFIED:
                    print(
                        "Ponowne ładowanie wag bazowych, ponieważ wznawianie z checkpointu się nie powiodło."
                    )
                    model = SPyNetModified(
                        model_name=config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME,
                        pretrained=True,
                    ).to(config.DEVICE)
                    # Re-initialize optimizer for the new model instance
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=config.LEARNING_RATE,
                        weight_decay=config.WEIGHT_DECAY,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                    )

        else:
            print(
                "OSTRZEŻENIE: RESUME_TRAINING=True, ale RESUME_CHECKPOINT_PATH jest nieprawidłowy lub nie istnieje."
            )
            print("Trening rozpocznie się od nowa.")
            # Ensure base weights are loaded if RESUME_TRAINING was true but path was invalid
            if config.LOAD_PRETRAINED_FOR_MODIFIED:
                model = SPyNetModified(
                    model_name=config.MODIFIED_SPYNET_PRETRAINED_VANILLA_NAME,
                    pretrained=True,
                ).to(config.DEVICE)
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config.LEARNING_RATE,
                    weight_decay=config.WEIGHT_DECAY,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

    # 4. Dataset i DataLoader
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

    # 5. Pętla treningowa
    print("\n--- Rozpoczęcie pętli treningowej ---")
    model.train()  # Ustaw model w tryb treningowy

    for epoch in range(config.START_EPOCH, config.NUM_EPOCHS):
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

            # Save model, optimizer, epoch, and loss
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,  # Save loss for this epoch
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint zapisany w: {checkpoint_path}")
        print("-" * 30)

    print("--- Trening zakończony ---")


if __name__ == "__main__":

    # Set the multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set, likely 'spawn'.")

    config = TrainConfig()

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
