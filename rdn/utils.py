import torch
import torch.nn.functional as F

# Attempt to import SPyNetModified and augmentations
# This assumes 'spynet' is a sibling directory to 'rdn' or in PYTHONPATH
import sys
import os

current_dir_rdn = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn = os.path.dirname(current_dir_rdn)
if parent_dir_rdn not in sys.path:
    sys.path.append(parent_dir_rdn)

try:
    from spynet.spynet_modified import backwarp as spynet_backwarp
except ImportError as e:
    print(f"Could not import spynet_backwarp. Ensure spynet module is accessible: {e}")
    spynet_backwarp = None


# --- Warping function (can use or adapt from SPyNet) ---
def warp_frame_with_flow(frame_tensor, flow_tensor):
    """Warps frame_tensor according to flow_tensor using spynet_backwarp."""
    if spynet_backwarp is None:
        raise RuntimeError("spynet_backwarp function not imported. Cannot warp frame.")
    # spynet_backwarp expects flow to be in pixel displacements.
    # It also expects Batch, C, H, W for frame and Batch, 2, H, W for flow.
    # Ensure input tensors are 4D (add batch dim if necessary) and C is appropriate.
    is_single_frame = frame_tensor.dim() == 3
    is_single_flow = flow_tensor.dim() == 3

    if is_single_frame:
        frame_tensor = frame_tensor.unsqueeze(0)
    if is_single_flow:
        flow_tensor = flow_tensor.unsqueeze(0)

    warped_frame = spynet_backwarp(frame_tensor, flow_tensor)

    if is_single_frame:
        warped_frame = warped_frame.squeeze(0)
    return warped_frame


def create_validity_mask(frame_height, frame_width, flow_tensor_for_warp):
    """
    Creates a validity mask. Pixels are valid if their source coordinates for warping were within bounds.
    The flow tensor (flow_tensor_for_warp) gives displacements (dx, dy).
    Normalized grid coordinates range from -1 to 1.
    A source coordinate (x_s, y_s) is sampled for each target pixel (x_t, y_t).
    grid_x(x_t, y_t) = 2*x_t/(W-1) - 1 + 2*dx(x_t,y_t)/(W-1)
    grid_y(x_t, y_t) = 2*y_t/(H-1) - 1 + 2*dy(x_t,y_t)/(H-1)
    Pixel is valid if -1 <= grid_x <= 1 and -1 <= grid_y <= 1.
    """
    B, _, H, W = (
        flow_tensor_for_warp.shape
        if flow_tensor_for_warp.dim() == 4
        else (1, *flow_tensor_for_warp.shape)
    )
    if flow_tensor_for_warp.dim() == 3:  # Add batch dim if not present
        flow_tensor_for_warp = flow_tensor_for_warp.unsqueeze(0)

    # Create a base grid of [-1, 1] coordinates
    grid_y_base, grid_x_base = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=flow_tensor_for_warp.device),
        torch.linspace(-1.0, 1.0, W, device=flow_tensor_for_warp.device),
        indexing="ij",
    )
    grid_base = (
        torch.stack((grid_x_base, grid_y_base), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    )  # B, 2, H, W

    # Normalize flow to also be in [-1, 1] range relative to total size
    norm_flow_x = flow_tensor_for_warp[:, 0:1, :, :] * (2.0 / (W - 1.0))
    norm_flow_y = flow_tensor_for_warp[:, 1:2, :, :] * (2.0 / (H - 1.0))
    norm_flow = torch.cat([norm_flow_x, norm_flow_y], dim=1)  # B, 2, H, W

    # Final sampling grid used by F.grid_sample
    sampling_grid = grid_base + norm_flow  # B, 2, H, W

    # Check if sampling coordinates are within [-1, 1]
    valid_x = (sampling_grid[:, 0, :, :] >= -1.0) & (sampling_grid[:, 0, :, :] <= 1.0)
    valid_y = (sampling_grid[:, 1, :, :] >= -1.0) & (sampling_grid[:, 1, :, :] <= 1.0)

    validity_mask = (valid_x & valid_y).float().unsqueeze(1)  # B, 1, H, W
    return (
        validity_mask.squeeze(0) if B == 1 else validity_mask
    )  # Return C, H, W or B, C, H, W


def run_basic_utils_test():
    print("--- Running Basic Utils Test ---")

    # Test warp_frame_with_flow
    print("\nTesting warp_frame_with_flow...")
    try:
        dummy_frame = torch.rand(3, 192, 320)
        dummy_flow = torch.rand(2, 192, 320) * 5

        # This will raise RuntimeError if spynet_backwarp is not available, which is desired.
        warped_single = warp_frame_with_flow(dummy_frame, dummy_flow)
        assert warped_single.shape == dummy_frame.shape, (
            f"Shape mismatch for single warp: {warped_single.shape}"
        )
        print("warp_frame_with_flow (single) successful.")

        dummy_frame_batch = torch.rand(2, 3, 192, 320)
        dummy_flow_batch = torch.rand(2, 2, 192, 320) * 5
        warped_batch = warp_frame_with_flow(dummy_frame_batch, dummy_flow_batch)
        assert warped_batch.shape == dummy_frame_batch.shape, (
            f"Shape mismatch for batch warp: {warped_batch.shape}"
        )
        print("warp_frame_with_flow (batched) successful.")

    except RuntimeError as r_e:
        if "spynet_backwarp function not imported" in str(r_e):
            print(
                f"Skipping warp_frame_with_flow test as spynet_backwarp is not imported: {r_e}"
            )
        else:
            print(f"RuntimeError during warp_frame_with_flow test: {r_e}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        print(f"Error during warp_frame_with_flow test: {e}")
        import traceback

        traceback.print_exc()

    # Test create_validity_mask
    print("\nTesting create_validity_mask...")
    try:
        dummy_flow_for_mask = torch.rand(2, 192, 320) * 20
        validity_mask_3d = create_validity_mask(192, 320, dummy_flow_for_mask)
        assert validity_mask_3d.shape == (1, 192, 320), (
            f"Shape mismatch for 3D flow mask: {validity_mask_3d.shape}"
        )
        print(f"create_validity_mask (3D flow) successful.")

        dummy_flow_batch_for_mask = torch.rand(2, 2, 192, 320) * 20
        validity_mask_4d = create_validity_mask(192, 320, dummy_flow_batch_for_mask)
        assert validity_mask_4d.shape == (2, 1, 192, 320), (
            f"Shape mismatch for 4D flow mask: {validity_mask_4d.shape}"
        )
        print(f"create_validity_mask (4D flow) successful.")

    except Exception as e:
        print(f"Error during create_validity_mask test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Utils Test Finished ---")


if __name__ == "__main__":
    run_basic_utils_test()
