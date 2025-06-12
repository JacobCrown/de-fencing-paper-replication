import os
import argparse
from typing import Any, Union
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import sys

current_dir_rdn = os.path.dirname(os.path.abspath(__file__))
parent_dir_rdn = os.path.dirname(current_dir_rdn)
if parent_dir_rdn not in sys.path:
    sys.path.append(parent_dir_rdn)


# Import RDN and SPyNet
from rdn.models import RDNInpainting
from spynet.spynet_modified import SPyNetModified
from rdn.utils import warp_frame_with_flow, create_validity_mask


# --- Argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="RDN Inpainting Inference Script")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder with K images and K masks (img1.png, ..., imgK.png, mask1.png, ..., maskK.png)",
    )
    parser.add_argument(
        "--rdn_checkpoint", type=str, required=True, help="Path to RDN model checkpoint"
    )
    parser.add_argument(
        "--spynet_checkpoint", type=str, required=True, help="Path to SPyNet checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the result (default: data_dir/result.png)",
    )
    return parser.parse_args()


def load_spynet(
    spynet_m_weights_path: str, target_device: torch.device
) -> Union[SPyNetModified, Any]:
    print(f"Loading SPyNetModified from {spynet_m_weights_path} for flow generation...")
    spynet_m = SPyNetModified(model_name="sintel-final", pretrained=False)
    if not os.path.exists(spynet_m_weights_path):
        raise FileNotFoundError(f"SPyNet-M weights not found: {spynet_m_weights_path}")

    checkpoint = torch.load(spynet_m_weights_path, map_location=torch.device("cpu"))
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


# --- Utility functions ---
def load_image(path, size):
    img = Image.open(path).convert("RGB")
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = 3  # BICUBIC fallback
    img = img.resize(size, resample)
    img = np.array(img).astype(np.float32) / 255.0
    return img


def load_mask(path, size):
    mask = Image.open(path).convert("L")
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = 0  # NEAREST fallback
    mask = mask.resize(size, resample)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)  # Binarize
    return mask


# --- Main inference logic ---
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load RDN checkpoint and get model config ---
    checkpoint = torch.load(args.rdn_checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    k_frames = config.get("k_frames", 5)
    img_width = config.get("img_width", 320)
    img_height = config.get("img_height", 192)
    num_input_channels = config.get("num_input_channels", 4 + (k_frames - 1) * 7)
    num_output_channels = config.get("num_output_channels", 3)
    num_features = config.get("num_features", 64)
    growth_rate = config.get("growth_rate", 64)
    num_blocks = config.get("num_blocks", 16)
    num_layers = config.get("num_layers", 8)

    print(f"Loading RDN config: {config}")

    spynet = load_spynet(args.spynet_checkpoint, device)

    # --- Load RDN model ---
    model = RDNInpainting(
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        num_features=num_features,
        growth_rate=growth_rate,
        num_blocks=num_blocks,
        num_layers=num_layers,
    ).to(device)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    for dir_ in os.listdir(args.data_dir):
        if not dir_.startswith("ex"):
            continue

        current_data_dir = os.path.join(args.data_dir, dir_)
        print(f"\nProcessing directory: {dir_}")

        # --- Load images and masks ---
        images = []
        masks = []
        for i in range(1, k_frames + 1):
            img_path = os.path.join(current_data_dir, f"img_{i}.png")
            mask_path = os.path.join(current_data_dir, f"mask_{i}.png")
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing {img_path} or {mask_path}")
            images.append(load_image(img_path, (img_width, img_height)))
            masks.append(load_mask(mask_path, (img_width, img_height)))
        images = np.stack(images, axis=0)  # (K, H, W, 3)
        masks = np.stack(masks, axis=0)  # (K, H, W)

        # --- Prepare tensors ---
        images_t = torch.from_numpy(images).permute(0, 3, 1, 2)  # (K, 3, H, W)
        masks_t = torch.from_numpy(masks).unsqueeze(1)  # (K, 1, H, W)
        images_t = images_t.to(device)
        masks_t = masks_t.to(device)

        # --- Prepare SPyNet and compute flows ---
        spynet_input_t = torch.cat([images_t, masks_t], dim=1)  # (K, 4, H, W)

        spynet.eval()
        with torch.no_grad():
            flows = []
            for i in range(k_frames):
                if i == 0:
                    flows.append(torch.zeros(2, img_height, img_width, device=device))
                else:
                    # Flow from frame i to frame 0 (keyframe)
                    flow = spynet(
                        spynet_input_t[i].unsqueeze(0), spynet_input_t[0].unsqueeze(0)
                    )
                    flows.append(flow.squeeze(0))
            flows = torch.stack(flows, dim=0)  # (K, 2, H, W)

        # --- Prepare RDN input (f_in) ---
        key_idx = 0
        key_img_t = images_t[key_idx]
        key_mask_t = masks_t[key_idx]
        i_k_m_for_rdn_input = key_img_t * (1 - key_mask_t)

        all_features_for_f_in = [i_k_m_for_rdn_input, key_mask_t]

        for i in range(k_frames):
            if i == key_idx:
                continue

            flow_i0 = flows[i]
            img_i_t = images_t[i]
            mask_i_t = masks_t[i]
            i_i_m_content_to_warp = img_i_t * (1 - mask_i_t)

            i_i_m_warped = warp_frame_with_flow(
                i_i_m_content_to_warp.unsqueeze(0), flow_i0.unsqueeze(0)
            ).squeeze(0)
            s_i_warped = warp_frame_with_flow(
                mask_i_t.unsqueeze(0), flow_i0.unsqueeze(0)
            ).squeeze(0)
            s_i_warped = (s_i_warped > 0.5).float()

            v_i = create_validity_mask(img_height, img_width, flow_i0).to(device)

            all_features_for_f_in.extend([i_i_m_warped, s_i_warped, v_i, flow_i0])

        f_in = torch.cat(all_features_for_f_in, dim=0)

        if f_in.shape[0] != num_input_channels:
            print(
                f"Warning: Constructed f_in has {f_in.shape[0]} channels, but model expects {num_input_channels}."
            )
            if f_in.shape[0] > num_input_channels:
                f_in = f_in[:num_input_channels, :, :]
            else:
                padding_needed = num_input_channels - f_in.shape[0]
                pad = torch.zeros(
                    (padding_needed, img_height, img_width), device=device
                )
                f_in = torch.cat([f_in, pad], dim=0)

        # --- Prepare masked keyframe (I_k^m) ---
        key_img = images_t[0]
        key_mask = masks_t[0]
        i_k_m = key_img * (1 - key_mask)  # (3, H, W)

        # --- Inference ---
        with torch.no_grad():
            f_in = f_in.to(device)
            i_k_m = i_k_m.to(device)
            residual_pred = model(f_in.unsqueeze(0))  # Add batch dim
            b_k_pred = i_k_m.unsqueeze(0) + residual_pred  # (1, 3, H, W)
            b_k_pred = b_k_pred.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
            b_k_pred = np.clip(b_k_pred, 0, 1)

        # --- Save result ---
        out_path = os.path.join(current_data_dir, "result.png")
        Image.fromarray((b_k_pred * 255).astype(np.uint8)).save(out_path)
        print(f"Saved inpainted result to {out_path}")


if __name__ == "__main__":
    main()
