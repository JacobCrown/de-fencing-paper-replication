import torch
import time
import os
import sys
from collections import OrderedDict
from typing import Optional

# Setup sys.path for sibling imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

try:
    from spynet.spynet_modified import SPyNetModified
except ImportError as e:
    print(
        f"CRITICAL ERROR: Could not import SPyNetModified. Ensure spynet module is accessible from project root: {e}"
    )
    print(
        "Please check your PYTHONPATH or run the script from the project's root directory if needed."
    )
    sys.exit(1)


class TestSpeedConfig:
    # --- User should verify this path ---
    spynet_m_weights_path: str = (
        "spynet_checkpoints/spynet_modified_ddp_epoch_ddp50_20250528-110600.pth"
    )
    # This is the name of the base SPyNet model structure, not necessarily the weights that will be loaded if pretrained=False.
    # It influences the initial architecture before custom weights are loaded.
    spynet_base_model_name: str = "sintel-final"

    img_width: int = 320
    img_height: int = 192
    batch_size: int = 1  # Typically, inference in this context might be on single frames or small batches

    num_warmup_runs: int = 50
    num_test_runs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_mode: str = (
        "reduce-overhead"  # "default", "reduce-overhead", or "max-autotune"
    )

    # Derived attributes
    actual_img_height: int
    actual_img_width: int

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Calculate actual dimensions ensuring they are multiples of 32 for SPyNet pyramid
        self.actual_img_height = max(
            32, (self.img_height + 31) // 32 * 32 if self.img_height > 0 else 32
        )
        self.actual_img_width = max(
            32, (self.img_width + 31) // 32 * 32 if self.img_width > 0 else 32
        )


def load_spynet_model(
    config: TestSpeedConfig, target_device: torch.device
) -> SPyNetModified:
    """Loads the SPyNetModified model and specified weights."""
    # Initialize model without loading default pretrained weights from URL, as we load from file
    model = SPyNetModified(model_name=config.spynet_base_model_name, pretrained=False)

    print(
        f"Loading SPyNetModified weights for inference test from: {config.spynet_m_weights_path}"
    )
    if not os.path.exists(config.spynet_m_weights_path):
        raise FileNotFoundError(
            f"SPyNet weights not found at {config.spynet_m_weights_path}"
        )

    try:
        # Load to CPU first to avoid GPU memory spike if model is large or checkpoint is from different device setup
        checkpoint = torch.load(
            config.spynet_m_weights_path, map_location=torch.device("cpu")
        )

        state_dict_to_load = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict_to_load = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict_to_load = checkpoint["state_dict"]
        elif isinstance(
            checkpoint, (OrderedDict, dict)
        ):  # Checkpoint is the state_dict itself
            state_dict_to_load = checkpoint
        else:
            raise ValueError(
                "Checkpoint format not recognized or model_state_dict not found."
            )

        # Adjust for 'module.' prefix if checkpoint came from DDP training
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("SPyNetModified weights loaded successfully.")

    except Exception as e:
        print(f"ERROR loading SPyNetModified weights: {e}")
        raise

    model.to(target_device)
    model.eval()
    return model


def measure_inference_time(
    model: torch.nn.Module,
    input1: torch.Tensor,
    input2: torch.Tensor,
    num_runs: int,
    model_description: str,
    target_device: torch.device,
):
    """Measures and prints average inference time for the given model."""
    total_time_ns = 0

    # Ensure inputs are on the target device
    input1 = input1.to(target_device)
    input2 = input2.to(target_device)

    for _ in range(num_runs):
        if target_device.type == "cuda":
            torch.cuda.synchronize()  # Wait for all preceding CUDA ops to finish

        start_time_ns = time.perf_counter_ns()
        with torch.no_grad():
            _ = model(input1, input2)

        if target_device.type == "cuda":
            torch.cuda.synchronize()  # Wait for model inference to finish

        end_time_ns = time.perf_counter_ns()
        total_time_ns += end_time_ns - start_time_ns

    avg_time_ms = (total_time_ns / num_runs) / 1_000_000  # Convert ns to ms
    print(
        f"{model_description}: Avg inference time over {num_runs} runs: {avg_time_ms:.3f} ms"
    )
    return avg_time_ms


def main():
    config = TestSpeedConfig()
    current_device = torch.device(config.device)
    print(f"Starting SPyNetModified inference speed test on device: {current_device}")
    print(f"Using weights: {config.spynet_m_weights_path}")

    if not os.path.exists(config.spynet_m_weights_path):
        print(
            f"ERROR: SPyNet-M weights path not found: {config.spynet_m_weights_path}. Please check TestSpeedConfig."
        )
        return

    # --- Prepare Dummy Inputs ---
    # Use actual_img_height/width from config which are ensured to be multiples of 32
    assert config.actual_img_height is not None, "actual_img_height should be set"
    assert config.actual_img_width is not None, "actual_img_width should be set"
    dummy_input1 = torch.randn(
        config.batch_size, 4, config.actual_img_height, config.actual_img_width
    )
    dummy_input2 = torch.randn(
        config.batch_size, 4, config.actual_img_height, config.actual_img_width
    )
    print(f"Using dummy input shape: {dummy_input1.shape}")

    # --- Test Original Model ---
    print("\\n--- Testing Original SPyNetModified ---")
    try:
        original_model = load_spynet_model(config, current_device)
        print(f"Warm-up for original model ({config.num_warmup_runs} runs)...")
        measure_inference_time(
            original_model,
            dummy_input1,
            dummy_input2,
            config.num_warmup_runs,
            "Original (Warm-up)",
            current_device,
        )
        avg_time_original_ms = measure_inference_time(
            original_model,
            dummy_input1,
            dummy_input2,
            config.num_test_runs,
            "Original Model",
            current_device,
        )
    except Exception as e:
        print(f"Error during original model test: {e}")
        avg_time_original_ms = float("inf")  # Indicate failure

    # --- Test Compiled Model ---
    avg_time_compiled_ms = float("inf")
    can_compile = hasattr(torch, "compile") and current_device.type == "cuda"

    if not can_compile:
        if current_device.type != "cuda":
            print(
                "\\nSkipping compilation test: torch.compile is most effective with CUDA devices."
            )
        else:  # implies torch.compile not available
            print(
                "\\nSkipping compilation test: torch.compile not available (requires PyTorch 2.0+)."
            )
    else:
        print(
            f"\\n--- Testing Compiled SPyNetModified (mode: {config.compile_mode}) ---"
        )
        try:
            compiled_model_candidate = load_spynet_model(
                config, current_device
            )  # Fresh instance for compilation

            pt_version = torch.__version__
            major_version = int(pt_version.split(".")[0])
            if major_version < 2:
                print(
                    f"Skipping compilation: PyTorch version {pt_version} is less than 2.0."
                )
            else:
                print(
                    f"Attempting to torch.compile model with mode='{config.compile_mode}'..."
                )

                # The result of torch.compile is a callable that acts like a torch.nn.Module
                # for inference purposes. The type checker might need a hint or ignore.
                compiled_model: torch.nn.Module = torch.compile(
                    compiled_model_candidate, mode=config.compile_mode
                )  # type: ignore
                print("Model compilation call successful.")

                print(
                    f"Warm-up for compiled model ({config.num_warmup_runs} runs, includes compilation time on first passes)..."
                )
                # The first few calls to a compiled model might be slower due to deferred compilation.
                measure_inference_time(
                    compiled_model,
                    dummy_input1,
                    dummy_input2,
                    config.num_warmup_runs,
                    "Compiled (Warm-up)",
                    current_device,
                )

                print(
                    f"Actual test runs for compiled model ({config.num_test_runs} runs)..."
                )
                avg_time_compiled_ms = measure_inference_time(
                    compiled_model,
                    dummy_input1,
                    dummy_input2,
                    config.num_test_runs,
                    "Compiled Model",
                    current_device,
                )

        except Exception as e_compile:
            print(f"ERROR during compilation or compiled model test: {e_compile}")
            import traceback

            traceback.print_exc()
            print("Proceeding without compiled model results.")

    # --- Comparison ---
    if avg_time_original_ms != float("inf") and avg_time_compiled_ms != float("inf"):
        print("\\n--- Comparison ---")
        if avg_time_compiled_ms < avg_time_original_ms:
            speedup_factor = avg_time_original_ms / avg_time_compiled_ms
            print(
                f"Compiled model is approximately {speedup_factor:.2f}x faster than the original model."
            )
            print(
                f"Original: {avg_time_original_ms:.3f} ms, Compiled: {avg_time_compiled_ms:.3f} ms"
            )
        else:
            slowdown_factor = avg_time_compiled_ms / avg_time_original_ms
            print(
                f"Compiled model is approximately {slowdown_factor:.2f}x slower (or similar speed) compared to the original model."
            )
            print(
                f"Original: {avg_time_original_ms:.3f} ms, Compiled: {avg_time_compiled_ms:.3f} ms"
            )
    elif avg_time_original_ms == float("inf"):
        print("\\nOriginal model test failed. Cannot compare.")
    elif avg_time_compiled_ms == float("inf") and can_compile:
        print("\\nCompiled model test failed or was skipped. Cannot compare.")


if __name__ == "__main__":
    main()
