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
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDNInpainting(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDNInpainting, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        # num_input_channels will be 32 (concatenated f_in)
        self.sfe1 = nn.Conv2d(num_input_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))


        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1), # Input is D blocks, each G0 features
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # Output layer for the residual
        # num_output_channels will be 3 (for the RGB residual)
        self.output_conv = nn.Conv2d(self.G0, num_output_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x_in): # x_in is the 32-channel f_in
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

if __name__ == '__main__':
    # Test RDNInpainting model
    # Parameters based on common RDN configurations and our inpainting task
    num_input_channels_test = 32  # For f_in
    num_output_channels_test = 3   # For RGB residual
    num_features_test = 64         # G0
    growth_rate_test = 64          # G
    num_blocks_test = 16             # D (number of RDBs)
    num_layers_test = 8              # C (layers within each RDB)
    batch_size_test = 2
    height_test = 64               # Example height
    width_test = 64                # Example width

    # Create dummy input tensor (simulating f_in)
    dummy_f_in = torch.randn(batch_size_test, num_input_channels_test, height_test, width_test)

    # Instantiate the model
    print(f"Testing RDNInpainting with input: B{batch_size_test}, C{num_input_channels_test}, H{height_test}, W{width_test}")
    model = RDNInpainting(num_input_channels=num_input_channels_test,
                          num_output_channels=num_output_channels_test,
                          num_features=num_features_test,
                          growth_rate=growth_rate_test,
                          num_blocks=num_blocks_test,
                          num_layers=num_layers_test)
    
    print("RDNInpainting model instantiated successfully.")
    # print(model)

    # Perform a forward pass
    try:
        with torch.no_grad(): # No need for gradients in this test
            output_residual = model(dummy_f_in)
        print(f"Forward pass successful. Output residual shape: {output_residual.shape}")
        # Expected output shape: (batch_size_test, num_output_channels_test, height_test, width_test)
        assert output_residual.shape == (batch_size_test, num_output_channels_test, height_test, width_test)
        print("Output shape is correct.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
