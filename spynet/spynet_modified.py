# spynet_modified.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Backwarp function (copied from original, crucial for the network's forward pass)
backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
            .view(1, 1, 1, -1)
            .repeat(1, 1, tenFlow.shape[2], 1)
        )
        tenVer = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, tenFlow.shape[3])
        )
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat(
            [tenHor, tenVer], 1
        )  # .cuda() removed, will use tenFlow.device
    # end

    # Ensure grid is on the same device as tenFlow
    grid_key = str(tenFlow.shape)
    if backwarp_tenGrid[grid_key].device != tenFlow.device:
        backwarp_tenGrid[grid_key] = backwarp_tenGrid[grid_key].to(tenFlow.device)

    # Adjust flow tensor for grid_sample:
    # Normalize flow values to range [-1, 1] based on input dimensions
    # tenFlow[:, 0:1, :, :] is horizontal flow (along width, tenInput.shape[3])
    # tenFlow[:, 1:2, :, :] is vertical flow (along height, tenInput.shape[2])

    # The original multiplication factors are for when tenInput.shape[3] or [2] are the dimensions
    # of the *original full-resolution image*. In SPyNet, tenInput is at a specific pyramid level.
    # The flow values from SPyNet are typically pixel displacements at that pyramid level.
    # grid_sample expects flow values to be offsets in the range [-1, 1] relative to the normalized grid.
    # A flow of W pixels to the right should map to an offset of W * (2 / (InputWidthAtThisLevel - 1)).

    # Let's use the dimensions of tenInput AT THE CURRENT PYRAMID LEVEL for normalization
    # This assumes tenFlow values are pixel displacements for the current tenInput resolution.
    norm_flow_x = tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0))
    norm_flow_y = tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    tenFlow_normalized = torch.cat([norm_flow_x, norm_flow_y], 1)

    return F.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[grid_key] + tenFlow_normalized).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class SPyNetModified(nn.Module):
    def __init__(self, model_name="sintel-final", pretrained=True):
        super().__init__()
        self.model_name = model_name  # Store model_name for weight loading

        class Preprocess(nn.Module):
            def __init__(self):
                super().__init__()
                # For RGB channels (first 3)
                self.mean_rgb = torch.tensor([0.485, 0.456, 0.406])
                self.std_rgb = torch.tensor(
                    [0.229, 1.0 / 0.224, 1.0 / 0.225]
                )  # Original has 1.0/std

            def forward(self, tenInput_rgbm):  # Expects B, C, H, W where C is 4 (RGBM)
                # tenInput_rgbm: [B, 4, H, W]
                # RGB channels are the first 3, Mask is the 4th
                tenRGB = tenInput_rgbm[:, 0:3, :, :]  # [B, 3, H, W]
                tenMask = tenInput_rgbm[:, 3:4, :, :]  # [B, 1, H, W]

                # Preprocess RGB channels
                tenRGB = tenRGB.flip(
                    [1]
                )  # BGR order in original, not relevant if loading from scratch or only using first 3 channels of imagenet mean/std
                # The original flips channels, which means RGB becomes BGR.
                # Let's assume input is RGB and apply normalization directly.
                # If the paper's authors used this exact SPyNet, then input to their modified SPyNet was likely also RGBM.

                # Move mean and std to the correct device if not already there
                if self.mean_rgb.device != tenRGB.device:
                    self.mean_rgb = self.mean_rgb.to(tenRGB.device)
                    self.std_rgb = self.std_rgb.to(tenRGB.device)

                tenRGB_processed = tenRGB - self.mean_rgb.view(1, 3, 1, 1)
                tenRGB_processed = tenRGB_processed * self.std_rgb.view(
                    1, 3, 1, 1
                )  # Original uses 1.0/std

                # Mask channel: typically normalized to [0,1] or [-1,1].
                # The paper doesn't specify mask preprocessing. Let's assume it's already [0,1]
                # and perhaps center it or scale it if needed. For now, pass it through or scale to [-1, 1] for consistency.
                # Simple pass-through or (mask * 2.0 - 1.0)
                tenMask_processed = tenMask  # Or: tenMask * 2.0 - 1.0

                return torch.cat(
                    [tenRGB_processed, tenMask_processed], 1
                )  # [B, 4, H, W]

        class BasicModified(nn.Module):
            def __init__(self, intLevel):
                super().__init__()
                # Original Basic module takes 8 channels: 3 (frame1) + 3 (warped frame2) + 2 (flow)
                # Modified will take 10 channels: 4 (frame1_rgbm) + 4 (warped frame2_rgbm) + 2 (flow)
                self.netBasic = nn.Sequential(
                    nn.Conv2d(
                        in_channels=10,
                        out_channels=32,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),  # MODIFIED
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=16,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=16,
                        out_channels=2,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                )

            def forward(self, tenInput):
                return self.netBasic(tenInput)

        self.netPreprocess = Preprocess()
        self.netBasic = nn.ModuleList(
            [BasicModified(intLevel) for intLevel in range(6)]
        )  # Use BasicModified

        if pretrained:
            self.load_pretrained_spynet_weights(model_name)

    def load_pretrained_spynet_weights(self, model_name):
        print(f"Loading SPyNet pre-trained weights for model: {model_name}")
        try:
            # Weights from official sniklaus/pytorch-spynet
            url = f"http://content.sniklaus.com/github/pytorch-spynet/network-{model_name}.pytorch"
            original_weights = torch.hub.load_state_dict_from_url(
                url=url, file_name=f"spynet-{model_name}"
            )
        except Exception as e:
            print(f"Could not download or load pre-trained weights: {e}")
            print("Model will be randomly initialized.")
            return

        model_dict = self.state_dict()
        processed_weights = {}

        for key_orig, weight_orig in original_weights.items():
            key_new = key_orig.replace("module", "net")  # Adapt key naming

            if key_new in model_dict:
                if model_dict[key_new].shape == weight_orig.shape:
                    processed_weights[key_new] = weight_orig
                elif (
                    key_new.startswith("netBasic.") and ".netBasic.0.weight" in key_new
                ):  # First conv in BasicModified
                    # This is the first Conv2d layer (in_channels=10) of a BasicModified module.
                    # Original was in_channels=8.
                    print(
                        f"Adapting weights for {key_new}: original shape {weight_orig.shape}, new shape {model_dict[key_new].shape}"
                    )

                    # Strategy: Copy weights for the 8 original input channels, zero-initialize for the 2 new mask channels.
                    # Original 8 channels: 3 (im1_rgb) + 3 (im2_rgb_warped) + 2 (flow)
                    # New 10 channels:    3 (im1_rgb) + 1 (im1_mask) + 3 (im2_rgb_warped) + 1 (im2_mask_warped) + 2 (flow)
                    # Or if paper's diagram `[I_i; S_i]` implies RGBM concatenation before passing to basic module,
                    # the concatenation order for BasicModified is:
                    # [ tenOne_rgbm[intLevel] (4ch),
                    #   backwarp(tenTwo_rgbm[intLevel], tenUpsampled) (4ch),
                    #   tenUpsampled (2ch) ]
                    # So the 10 channels are:
                    # C0-3: tenOne_rgbm
                    # C4-7: warped tenTwo_rgbm
                    # C8-9: flow
                    # And the original 8 channels were:
                    # C0-2: tenOne_rgb
                    # C3-5: warped tenTwo_rgb
                    # C6-7: flow

                    new_weight = model_dict[
                        key_new
                    ].clone()  # Get new shape, init with random
                    # Copy RGB from frame1 (original channels 0,1,2 -> new channels 0,1,2)
                    new_weight[:, 0:3, :, :] = weight_orig[:, 0:3, :, :]
                    # Mask for frame1 (new channel 3) -> initialize (e.g. zero or small random)
                    # new_weight[:, 3, :, :].zero_() # Or random init from Conv2d is fine

                    # Copy RGB from frame2_warped (original channels 3,4,5 -> new channels 4,5,6)
                    new_weight[:, 4:7, :, :] = weight_orig[:, 3:6, :, :]
                    # Mask for frame2_warped (new channel 7) -> initialize
                    # new_weight[:, 7, :, :].zero_()

                    # Copy flow (original channels 6,7 -> new channels 8,9)
                    new_weight[:, 8:10, :, :] = weight_orig[:, 6:8, :, :]

                    processed_weights[key_new] = new_weight

                elif (
                    key_new.startswith("netBasic.") and ".netBasic.0.bias" in key_new
                ):  # Bias for first conv
                    # Bias has shape (out_channels), so it's unaffected by in_channels change.
                    processed_weights[key_new] = weight_orig
                else:
                    print(
                        f"Skipping {key_new}: Shape mismatch. Model: {model_dict[key_new].shape}, Pretrained: {weight_orig.shape}"
                    )
            else:
                print(f"Skipping {key_new}: Key not in current model.")

        model_dict.update(processed_weights)
        self.load_state_dict(model_dict)
        print("Pre-trained SPyNet weights loaded and adapted successfully.")

    def forward(self, tenOne_rgbm, tenTwo_rgbm):  # Input images are RGBM (4 channels)
        tenFlow_pyramid = []  # Will store flow at each pyramid level if needed

        # Preprocess inputs (handles RGBM)
        tenOne_processed_pyramid = [
            self.netPreprocess(tenOne_rgbm)
        ]  # Full res preprocessed
        tenTwo_processed_pyramid = [
            self.netPreprocess(tenTwo_rgbm)
        ]  # Full res preprocessed

        # Create image pyramids
        for intLevel in range(5):  # Create 5 more levels (total 6 including original)
            if (
                tenOne_processed_pyramid[0].shape[2] > 32
                or tenOne_processed_pyramid[0].shape[3] > 32
            ):
                tenOne_processed_pyramid.insert(
                    0,
                    F.avg_pool2d(
                        input=tenOne_processed_pyramid[0],
                        kernel_size=2,
                        stride=2,
                        count_include_pad=False,
                    ),
                )
                tenTwo_processed_pyramid.insert(
                    0,
                    F.avg_pool2d(
                        input=tenTwo_processed_pyramid[0],
                        kernel_size=2,
                        stride=2,
                        count_include_pad=False,
                    ),
                )
            else:
                # If image is too small, just replicate the smallest level
                # This logic might need adjustment if we hit this case often with 192x320
                # For 192x320: 192->96->48->24 (level 3 is 24, smallest is 6x10 at level 5)
                # This break ensures we don't go below a certain size if original code implies it.
                # Original code makes 6 levels. 320/2^5 = 10. 192/2^5 = 6.
                # So it seems for 192x320, it will always create 6 levels.
                pass

        # Initialize flow at the coarsest level
        # Shape of tenOne_processed_pyramid[0] is the coarsest level after pyramid creation
        coarsest_H = int(math.floor(tenOne_processed_pyramid[0].shape[2] / 2.0))
        coarsest_W = int(math.floor(tenOne_processed_pyramid[0].shape[3] / 2.0))
        tenFlow = tenOne_processed_pyramid[0].new_zeros(
            [tenOne_processed_pyramid[0].shape[0], 2, coarsest_H, coarsest_W]
        )

        # Iterative flow estimation from coarse to fine
        for intLevel in range(len(tenOne_processed_pyramid)):
            # Upsample flow from previous (coarser) level
            tenUpsampledFlow = (
                F.interpolate(
                    input=tenFlow, scale_factor=2, mode="bilinear", align_corners=True
                )
                * 2.0
            )

            # Pad if shapes do not match due to odd dimensions
            # Target shape for upsampled flow is tenOne_processed_pyramid[intLevel]
            target_H, target_W = tenOne_processed_pyramid[intLevel].shape[2:4]
            if tenUpsampledFlow.shape[2] != target_H:
                padding_h = target_H - tenUpsampledFlow.shape[2]
                tenUpsampledFlow = F.pad(
                    input=tenUpsampledFlow, pad=[0, 0, 0, padding_h], mode="replicate"
                )
            if tenUpsampledFlow.shape[3] != target_W:
                padding_w = target_W - tenUpsampledFlow.shape[3]
                tenUpsampledFlow = F.pad(
                    input=tenUpsampledFlow, pad=[0, padding_w, 0, 0], mode="replicate"
                )

            # Concatenate inputs for the BasicModified module:
            # 1. tenOne_processed_pyramid[intLevel] (current level frame1_rgbm, 4ch)
            # 2. Warped tenTwo_processed_pyramid[intLevel] using tenUpsampledFlow (warped frame2_rgbm, 4ch)
            # 3. tenUpsampledFlow (flow from coarser level, 2ch)
            # Total = 4 + 4 + 2 = 10 channels

            tenWarpedTwo = backwarp(
                tenInput=tenTwo_processed_pyramid[intLevel], tenFlow=tenUpsampledFlow
            )

            concat_input = torch.cat(
                [tenOne_processed_pyramid[intLevel], tenWarpedTwo, tenUpsampledFlow], 1
            )

            # Estimate residual flow and add to upsampled flow
            tenFlow = self.netBasic[intLevel](concat_input) + tenUpsampledFlow
            # tenFlow_pyramid.append(tenFlow) # Store flow at each level if needed later

        # tenFlow is now the flow at the finest (original preprocessed) resolution
        return tenFlow


if __name__ == "__main__":
    # Test instantiation and weight loading
    # Ensure you have internet for torch.hub.load_state_dict_from_url
    model_name_test = "sintel-final"  # or 'chairs-final' etc.

    print(f"Testing SPyNetModified with model: {model_name_test}")
    try:
        model = SPyNetModified(model_name=model_name_test, pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()
        model.train(False)  # Set to eval mode
        print("SPyNetModified instantiated and weights loaded.")
    except Exception as e:
        print(f"Error during instantiation or weight loading: {e}")
        exit()

    # Test forward pass with dummy RGBM input
    batch_size = 1
    # The original SPyNet script has assertions for specific input sizes like 1024x416
    # and then resizes to multiples of 32. Our network should handle typical sizes.
    # Paper uses 320x192. This must be divisible by 2^5 = 32 for 6 pyramid levels.
    # 320/32 = 10. 192/32 = 6. So these dimensions work.
    height, width = 192, 320

    # Ensure height and width are multiples of 32 for the 6 pyramid levels
    # (original resolution / 2^5 should be integer)
    # H_coarsest = height // (2**5)
    # W_coarsest = width // (2**5)
    # print(f"Coarsest dimensions: {H_coarsest}x{W_coarsest}")

    dummy_frame1_rgbm = torch.randn(batch_size, 4, height, width)  # B, C=4, H, W
    dummy_frame2_rgbm = torch.randn(batch_size, 4, height, width)

    if torch.cuda.is_available():
        dummy_frame1_rgbm = dummy_frame1_rgbm.cuda()
        dummy_frame2_rgbm = dummy_frame2_rgbm.cuda()

    print(f"Input shape: {dummy_frame1_rgbm.shape}")

    try:
        with torch.inference_mode():  # Inference mode
            output_flow = model(dummy_frame1_rgbm, dummy_frame2_rgbm)
        print(f"Forward pass successful. Output flow shape: {output_flow.shape}")
        # Expected output_flow shape: (B, 2, H, W) matching the finest preprocessed input dimensions
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
