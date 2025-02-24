import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from typing import Optional
from networks.controlnet import ControlNetAdapter
import os
import glob

class DiffusionTransformerWithControl(nn.Module):
    def __init__(
        self, 
        base_transformer: nn.Module, 
        control_adapter: ControlNetAdapter, 
        injection_layers: tuple = (2, 5, 8),
        dtype=torch.bfloat16,
        device=torch.device("cuda")  # Add device parameter
    ):
        super().__init__()
        self.transformer = base_transformer
        self.control_adapter = control_adapter
        self.injection_layers = set(injection_layers)
        self.control_proj = nn.Conv3d(
            control_adapter.out_channels, 
            base_transformer.hidden_size, 
            kernel_size=1,
            dtype=dtype,
            device=device  # Set device for Conv3d
        )
        # Ensure bias is in the correct dtype
        if self.control_proj.bias is not None:
            self.control_proj.bias.data = self.control_proj.bias.data.to(dtype=dtype)
        
        # Debug: Check control_proj weight and bias dtype/device
        print(f"control_proj weight dtype: {self.control_proj.weight.dtype}")  # Should be bfloat16
        print(f"control_proj weight device: {self.control_proj.weight.device}")  # Should be cuda
        print(f"control_proj bias dtype: {self.control_proj.bias.dtype}")      # Should be bfloat16
        print(f"control_proj bias device: {self.control_proj.bias.device}")    # Should be cuda  
            
    @property
    def patch_size(self):
        return self.transformer.patch_size  # Forward the patch_size attribute
    
    @property
    def img_in(self):
        return self.transformer.img_in

    # Forward the enable_block_swap method to the base_transformer
    def enable_block_swap(self, *args, **kwargs):
        return self.transformer.enable_block_swap(*args, **kwargs)

    # Forward the move_to_device_except_swap_blocks method to the base_transformer
    def move_to_device_except_swap_blocks(self, *args, **kwargs):
        return self.transformer.move_to_device_except_swap_blocks(*args, **kwargs)

    # Forward the prepare_block_swap_before_forward method to the base_transformer
    def prepare_block_swap_before_forward(self, *args, **kwargs):
        return self.transformer.prepare_block_swap_before_forward(*args, **kwargs)

    # Forward the enable_gradient_checkpointing method to the base_transformer
    def enable_gradient_checkpointing(self, *args, **kwargs):  # ADD THIS METHOD
        return self.transformer.enable_gradient_checkpointing(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Timestep tensor
        text_states: Optional[torch.Tensor] = None,  # Make text_states optional
        control_input: Optional[torch.Tensor] = None,
        control_alpha: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        # Base image processing
        img = self.transformer.img_in(x)
        B, hidden_size, T, H, W = img.shape
        img = img.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, hidden_size)

        # Control features processing
        control_proj_flat = None
        if control_input is not None:
            control_features = self.control_adapter(control_input)
            control_features = F.interpolate(control_features, size=(T, H, W), mode='trilinear', align_corners=False)
            control_proj = self.control_proj(control_features)
            control_proj_flat = control_proj.permute(0, 2, 3, 4, 1).contiguous().view(B, T * H * W, hidden_size)

        # Text and time conditioning (skip if text_states is None)
        if text_states is not None:
            txt = self.transformer.txt_in(text_states, t)  # Pass t to txt_in
        else:
            # Skip text conditioning
            txt = torch.zeros_like(img)  # Use a zero tensor as a placeholder

        vec = self.transformer.time_in(t)

        # Transformer processing with injections
        for i, block in enumerate(self.transformer.double_blocks):
            img, txt = block(img, txt, vec)
            if control_proj_flat is not None and i in self.injection_layers:
                img = img + control_alpha * control_proj_flat

        x_merged = torch.cat([img, txt], dim=1)
        for block in self.transformer.single_blocks:
            x_merged = block(x_merged, vec)

        output_img = x_merged[:, :T*H*W, :]
        out = self.transformer.final_layer(output_img, vec)
        return self.transformer.unpatchify(out, T, H, W)        
class ConditionedVideoDataset(torch.utils.data.Dataset):
    """Dataset that pairs original latents with control-conditioned latents"""
    def __init__(self, config: dict, control_type: str):
        self.current_epoch = 0
        self.current_step = 0
        self.control_type = control_type
        self.pairs = self._find_pairs(config)

        # Debug: Print the number of pairs found
        print(f"Found {len(self.pairs)} pairs of cache and conditioning files.")

    def _find_pairs(self, config: dict) -> list:
        # Find all .safetensors files in the cache directory
        cache_files = list(Path(config['cache_dir']).glob("*.safetensors"))
        print(f"Found {len(cache_files)} cache files in {config['cache_dir']}")

        # Find corresponding conditioning files
        pairs = []
        for orig in cache_files:
            cond_path = Path(config['conditioning_cache_dir']) / f"{orig.stem}_conditioning.safetensors"
            if cond_path.exists():
                pairs.append((orig, cond_path))
            else:
                print(f"Warning: Conditioning file not found for {orig.stem}")

        # Debug: Print the number of valid pairs
        print(f"Found {len(pairs)} valid pairs of cache and conditioning files.")
        return pairs

    def __len__(self) -> int:
        # Return the number of valid pairs
        return len(self.pairs)

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    def set_current_step(self, step: int):
        self.current_step = step
    
    def __getitem__(self, idx: int) -> dict:
        # Get the paths for the original and conditioned latents
        orig_path, cond_path = self.pairs[idx]  # Retrieve paths from self.pairs
        print(f"Loading cache file: {orig_path}")
        print(f"Loading conditioning file: {cond_path}")

        try:
            # Load the original and conditioned latents
            orig = load_file(orig_path)['latent'].float()  # Shape: (B,16,T,H,W)
            cond = load_file(cond_path)['latent'].float()  # Shape: (B,1,T,H,W) or (B,16,T,H,W)

            # Reshape depth latents to 16 channels (if needed)
            if self.control_type == 'depth':
                # Expand single-channel depth to 16 channels
                if cond.ndim == 4:  # (B, T, H, W)
                    cond = cond.unsqueeze(1)  # Add channel dim -> (B,1,T,H,W)
                cond = cond.repeat(1, 16, 1, 1, 1)  # (B,16,T,H,W)

            return {'original': orig, 'conditioned': cond, 'name': orig_path.stem}
        except Exception as e:
            print(f"Error loading files {orig_path} or {cond_path}: {e}")
            raise
        
# In controlnet_adapter.py (ControlNetWrapper)
class ControlNetWrapper(nn.Module):
    def __init__(
        self, 
        base_transformer: nn.Module, 
        control_type: str, 
        injection_layers: tuple = (2, 5, 8),
        dit_dtype=torch.bfloat16,
        device=torch.device("cuda")
    ):
        super().__init__()
        self.control_type = control_type
        self.adapter = ControlNetAdapter(
            control_type=control_type,
            dtype=dit_dtype,
            device=device
        )
        self.transformer = DiffusionTransformerWithControl(
            base_transformer=base_transformer,
            control_adapter=self.adapter,
            injection_layers=injection_layers,
            dtype=dit_dtype,
            device=device  # Pass device here
        )
        # Initialize projection weights
        self._init_control_projection()

    def _init_control_projection(self):
        nn.init.kaiming_normal_(self.transformer.control_proj.weight, 
                               mode='fan_out', 
                               nonlinearity='leaky_relu' if self.control_type != 'canny' else 'linear')
        nn.init.zeros_(self.transformer.control_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor,
        control_input: Optional[torch.Tensor] = None,
        control_alpha: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        # Handle different dimensionalities for control inputs
        if control_input is not None:
            if self.control_type == 'depth' and control_input.ndim == 4:
                control_input = control_input.unsqueeze(1)  # Add channel dim
            elif self.control_type == 'pose' and control_input.shape[1] != 16:
                control_input = control_input.view(-1, 16, *control_input.shape[-2:])

        return self.transformer(
            x=x,
            t=t,
            text_states=text_states,
            control_input=control_input,
            control_alpha=control_alpha,
            **kwargs
        )

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.transformer.disable_gradient_checkpointing()

    def switch_block_swap(self, enable: bool):
        """Enable/disable block swapping for memory optimization"""
        if hasattr(self.transformer.transformer, 'enable_block_swap'):
            if enable:
                self.transformer.transformer.enable_block_swap()
            else:
                self.transformer.transformer.disable_block_swap()

# In controlnet_adapter.py
def load_controlnet(
    base_model: 'HYVideoDiffusionTransformer', 
    control_type: str, 
    dit_dtype=torch.bfloat16,
    device=torch.device("cuda"),
    **kwargs
) -> ControlNetWrapper:
    return ControlNetWrapper(
        base_transformer=base_model,
        control_type=control_type,
        dit_dtype=dit_dtype,
        device=device,
        **{**adapter_config, **kwargs}
    )

def load_controlnet_weights(model: ControlNetWrapper, checkpoint_path: str, strict: bool = True):
    """Load pretrained weights for control adapter components"""
    state_dict = load_file(checkpoint_path)
    adapter_weights = {k.replace('adapter.', ''): v for k, v in state_dict.items() if 'adapter' in k}
    model.adapter.load_state_dict(adapter_weights, strict=strict)
    return model
