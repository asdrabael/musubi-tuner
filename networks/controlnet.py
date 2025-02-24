import torch
import torch.nn as nn

# In controlnet.py
# In controlnet.py
class ControlNetAdapter(nn.Module):
    def __init__(
        self, 
        control_type: str, 
        in_channels: int = 1, 
        out_channels: int = 16, 
        mid_channels: int = 32, 
        num_layers: int = 3,
        dtype=torch.float32,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.control_type = control_type
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        
        # Set input channels based on control type
        if self.control_type in ['pose', 'depth']:
            in_channels = 16
        else:  # canny
            in_channels = 1
        
        layers = []
        # Add Conv3d layer
        conv_layer = nn.Conv3d(
            in_channels, mid_channels, 
            kernel_size=3, padding=1,
            dtype=self.dtype,
            device=self.device
        )
        layers.append(conv_layer)
        
        # Debug: Check Conv3d weight and bias dtype
        print(f"Conv3d weight dtype: {layers[-1].weight.dtype}")  # Should be bfloat16
        print(f"Conv3d bias dtype: {layers[-1].bias.dtype}")      # Should be bfloat16
        
        # Ensure bias is in the correct dtype
        if layers[-1].bias is not None:
            layers[-1].bias.data = layers[-1].bias.data.to(dtype=self.dtype)
        
        # Add normalization and activation layers
        layers.append(self._get_normalization_layer(mid_channels).to(device=self.device, dtype=self.dtype))
        layers.append(self._get_activation().to(device=self.device, dtype=self.dtype))
        
        # Repeat for all layers...
        for _ in range(num_layers - 2):
            layers.append(nn.Conv3d(
                mid_channels, mid_channels, 
                kernel_size=3, padding=1,
                dtype=self.dtype,
                device=self.device
            ))
            layers.append(self._get_normalization_layer(mid_channels).to(device=self.device, dtype=self.dtype))
            layers.append(self._get_activation().to(device=self.device, dtype=self.dtype))
        
        # Final Conv3d layer
        layers.append(nn.Conv3d(
            mid_channels, out_channels, 
            kernel_size=3, padding=1,
            dtype=self.dtype,
            device=self.device
        ))
        
        # Initialize net as a Sequential module
        self.net = nn.Sequential(*layers)

    def _get_normalization_layer(self, channels: int) -> nn.Module:
        if self.control_type == 'pose':
            return nn.GroupNorm(8, channels)
        elif self.control_type == 'depth':
            return nn.BatchNorm3d(channels)
        elif self.control_type == 'canny':
            return nn.InstanceNorm3d(channels)
        raise ValueError(f"Invalid control type: {self.control_type}")

    def _get_activation(self) -> nn.Module:
        return nn.Sigmoid() if self.control_type == 'canny' else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

