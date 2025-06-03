import torch
import os
import sys
import copy  # For deep copying the model
from collections import OrderedDict

# Setup sys.path for sibling imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

try:
    from spynet.spynet_modified import SPyNetModified
except ImportError as e:
    print(
        f"CRITICAL ERROR: Could not import SPyNetModified. "
        f"Ensure spynet module is accessible from project root: {e}"
    )
    sys.exit(1)


class CorrectnessTestConfig:
    spynet_m_weights_path: str = "spynet_checkpoints/spynet_modified_ddp_epoch_ddp158_20250529-093520.pth"  # User should verify/update this
    spynet_base_model_name: str = "sintel-final"

    img_width: int = (
        64  # Using smaller dimensions for faster test, but still multiple of 32
    )
    img_height: int = 32
    batch_size: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_mode: str = "max-autotune"  # or "reduce-overhead" or None
    random_seed: int = 1997

    # Tolerances for torch.allclose()
    atol: float = 1e-5  # Absolute tolerance
    rtol: float = 1e-4  # Relative tolerance


def load_spynet_model_for_test(
    config: CorrectnessTestConfig, target_device: torch.device
) -> SPyNetModified:
    model = SPyNetModified(model_name=config.spynet_base_model_name, pretrained=False)
    if not os.path.exists(config.spynet_m_weights_path):
        print(
            f"WARNING: SPyNet weights not found at {config.spynet_m_weights_path}. Using initialized model."
        )
    else:
        try:
            checkpoint = torch.load(
                config.spynet_m_weights_path, map_location=torch.device("cpu")
            )
            state_dict_to_load = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )

            new_state_dict = OrderedDict()
            for k, v in state_dict_to_load.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(f"SPyNetModified weights loaded from: {config.spynet_m_weights_path}")
        except Exception as e:
            print(
                f"ERROR loading SPyNetModified weights: {e}. Using initialized model."
            )

    model.to(target_device)
    model.eval()
    return model


def main():
    config = CorrectnessTestConfig()
    torch.manual_seed(config.random_seed)
    current_device = torch.device(config.device)

    print(
        f"Starting SPyNetModified compilation correctness test on device: {current_device}"
    )
    print(
        f"Using compile mode: {config.compile_mode if config.compile_mode else 'None (eager)'}"
    )

    # 1. Load Original Model
    original_model = load_spynet_model_for_test(config, current_device)

    # 2. Create Compiled Model (as a deep copy of the original before compilation)
    compiled_model_candidate = None
    if (
        config.compile_mode
        and hasattr(torch, "compile")
        and current_device.type == "cuda"
    ):
        print(
            f"Attempting to torch.compile a copy of the model with mode='{config.compile_mode}'..."
        )
        model_to_compile = copy.deepcopy(original_model)
        try:
            compiled_model_candidate = torch.compile(
                model_to_compile, mode=config.compile_mode
            )
            print("Model compilation call successful.")
        except Exception as e_compile:
            print(
                f"ERROR during model compilation: {e_compile}. Will only test original model."
            )
            compiled_model_candidate = None
    elif current_device.type != "cuda":
        print(
            "Skipping compilation test: torch.compile is most effective and typically tested on CUDA devices."
        )
    elif not hasattr(torch, "compile"):
        print(
            "Skipping compilation test: torch.compile not available (requires PyTorch 2.0+)."
        )

    # 3. Prepare Dummy Inputs (fixed)
    # Ensure consistent inputs by setting seed right before creation if not already fixed
    torch.manual_seed(
        config.random_seed + 1
    )  # Use a different seed for data if desired, or same
    dummy_input1 = torch.randn(
        config.batch_size, 4, config.img_height, config.img_width, device=current_device
    )
    dummy_input2 = torch.randn(
        config.batch_size, 4, config.img_height, config.img_width, device=current_device
    )
    print(f"Using dummy input shape: {dummy_input1.shape}")

    # 4. Get Output from Original Model
    print("\n--- Running Original Model ---")
    with torch.inference_mode():
        output_original = original_model(dummy_input1, dummy_input2)
    print(f"Original model output sum: {output_original.sum().item()}")

    # 5. Get Output from Compiled Model (if successfully compiled)
    if compiled_model_candidate:
        print("\n--- Running Compiled Model ---")
        with torch.inference_mode():
            # Warm-up run for compiled model (especially for max-autotune)
            print("Performing warm-up run for compiled model...")
            _ = compiled_model_candidate(dummy_input1, dummy_input2)
            if current_device.type == "cuda":
                torch.cuda.synchronize()

            print("Performing actual run for compiled model...")
            output_compiled = compiled_model_candidate(dummy_input1, dummy_input2)
        print(f"Compiled model output sum: {output_compiled.sum().item()}")

        # 6. Compare Outputs
        print("\n--- Comparison --- inciting turmoil")
        try:
            are_close = torch.allclose(
                output_original, output_compiled, atol=config.atol, rtol=config.rtol
            )
            if are_close:
                print(
                    f"SUCCESS: Outputs of original and compiled models are close (atol={config.atol}, rtol={config.rtol})."
                )
            else:
                print(
                    f"FAILURE: Outputs of original and compiled models differ significantly (atol={config.atol}, rtol={config.rtol})."
                )
                abs_diff = torch.abs(output_original - output_compiled)
                print(f"  Max absolute difference: {abs_diff.max().item()}")
                print(f"  Mean absolute difference: {abs_diff.mean().item()}")

                # Further check: How many elements are not close?
                not_close_elements = ~torch.isclose(
                    output_original, output_compiled, atol=config.atol, rtol=config.rtol
                )
                print(
                    f"  Number of differing elements: {not_close_elements.sum().item()} out of {output_original.numel()}"
                )

        except Exception as e_compare:
            print(f"ERROR during output comparison: {e_compare}")
    elif config.compile_mode and current_device.type == "cuda":
        print("\nSkipping comparison as compiled model was not successfully created.")
    else:
        print("\nSkipping comparison as compilation was not attempted or applicable.")


if __name__ == "__main__":
    main()
