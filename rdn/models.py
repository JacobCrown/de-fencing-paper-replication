import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate)
                for i in range(num_layers)
            ]
        )

        # local feature fusion
        self.lff = nn.Conv2d(
            in_channels + growth_rate * num_layers, growth_rate, kernel_size=1
        )

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDNInpainting(nn.Module):
    def __init__(
        self,
        num_input_channels,
        num_output_channels,
        num_features,
        growth_rate,
        num_blocks,
        num_layers,
    ):
        super(RDNInpainting, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        # num_input_channels will be 32 (concatenated f_in)
        self.sfe1 = nn.Conv2d(
            num_input_channels, num_features, kernel_size=3, padding=3 // 2
        )
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(
                self.G0 * self.D, self.G0, kernel_size=1
            ),  # Input is D blocks, each G0 features
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )

        # Output layer for the residual
        # num_output_channels will be 3 (for the RGB residual)
        self.output_conv = nn.Conv2d(
            self.G0, num_output_channels, kernel_size=3, padding=3 // 2
        )

    def forward(self, x_in):  # x_in is the 32-channel f_in
        # Shallow Feature Extraction
        sfe1_out = self.sfe1(x_in)
        sfe2_out = self.sfe2(sfe1_out)

        current_features = sfe2_out
        local_features_list = []
        for i in range(self.D):
            current_features = self.rdbs[i](current_features)
            local_features_list.append(current_features)

        gff_input = torch.cat(local_features_list, 1)
        gff_out = self.gff(gff_input)

        globally_fused_features = gff_out + sfe1_out

        residual_pred = self.output_conv(globally_fused_features)

        return residual_pred


def run_basic_model_test():
    print("--- Running Basic Model Test for RDNInpainting ---")

    # Define some test parameters
    batch_size = 2
    img_h, img_w = 48, 80  # Small dimensions for quick test
    k_frames = 3
    num_input_channels_calc = (
        4 + (k_frames - 1) * 7
    )  # I_k^m, S_k, (K-1)*(Î_j^m, Š_j, V_j, f_kj^m)
    num_output_channels_model = 3  # RGB residual
    num_features_model = 16  # Smaller G0 for faster test
    growth_rate_model = 16  # Smaller G
    num_blocks_model = 2  # Smaller D
    num_layers_model = 2  # Smaller C

    print(f"Test params: B={batch_size}, H={img_h}, W={img_w}, K={k_frames}")
    print(f"Calculated input channels for RDN: {num_input_channels_calc}")
    print(
        f"Model arch: G0={num_features_model}, G={growth_rate_model}, D={num_blocks_model}, C={num_layers_model}"
    )

    try:
        # 1. Initialize model
        model = RDNInpainting(
            num_input_channels=num_input_channels_calc,
            num_output_channels=num_output_channels_model,
            num_features=num_features_model,
            growth_rate=growth_rate_model,
            num_blocks=num_blocks_model,
            num_layers=num_layers_model,
        )
        model.eval()  # Set to eval mode for testing (if it affects anything like dropout)
        print("RDNInpainting model initialized successfully.")

        # 2. Create dummy input tensor
        #    f_in = [I_k^m (3), S_k (1), V_k (1, all ones),
        #            Î_1^m (3), Š_1 (1), V_1 (1), f_k1^m (2), ... K-1 times]
        #    For V_k, the paper says I_k^m, S_k for keyframe, and then others are warped + validity.
        #    The RDN input fin is I_k^m, S_k, then concat of [Î_j^m, Š_j, V_j, f_kj^m] for j != k.
        #    So num_input_channels = 3 + 1 + (K-1)*(3+1+1+2) = 4 + (K-1)*7.
        dummy_f_in = torch.rand(batch_size, num_input_channels_calc, img_h, img_w)
        print(f"Created dummy input f_in with shape: {dummy_f_in.shape}")

        # 3. Perform a forward pass
        with torch.no_grad():
            output_residual = model(dummy_f_in)
        print(
            f"Model forward pass successful. Output (residual) shape: {output_residual.shape}"
        )

        # 4. Check output shape
        expected_output_shape = (batch_size, num_output_channels_model, img_h, img_w)
        assert output_residual.shape == expected_output_shape, (
            f"Output shape mismatch. Expected {expected_output_shape}, got {output_residual.shape}"
        )
        print("Output shape is correct.")

        # 5. Test with skip connection (conceptual, as model returns residual)
        #    The actual addition of residual happens outside the model in the training script.
        #    B_k_pred = I_k^m + residual_pred
        #    Here we just check if a dummy I_k^m can be added to the residual.
        dummy_i_k_m = torch.rand(batch_size, num_output_channels_model, img_h, img_w)
        reconstructed_output = dummy_i_k_m + output_residual
        assert reconstructed_output.shape == expected_output_shape, (
            "Reconstructed output shape mismatch"
        )
        print("Residual addition (conceptual) successful.")

    except Exception as e:
        print(f"Error during RDNInpainting model test: {e}")
        import traceback

        traceback.print_exc()

    print("--- Model Test Finished ---")


if __name__ == "__main__":
    run_basic_model_test()
