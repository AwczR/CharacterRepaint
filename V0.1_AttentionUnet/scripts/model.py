# model.py (Attention U-Net Version using Diffusers)
import torch
import torch.nn as nn
from diffusers import UNet2DModel

class AttentionUNetDenoisingModel(nn.Module):
    """
    A wrapper for the diffusers.UNet2DModel to make it compatible with the existing training pipeline.
    This model incorporates attention mechanisms.

    It translates parameters from the project's config.yaml format to the format
    expected by the diffusers UNet2DModel.
    """
    def __init__(
        self,
        image_size: int = 288,
        in_channels: int = 1,
        out_channels: int = 1, # Note: This is the original out_channels, not num_classes
        num_classes: int = None,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim_ratio: int = 4, # This parameter is handled by diffusers internally, kept for compatibility
        groups: int = 32,
        # New parameters for attention control
        attention_resolutions: tuple = (18, 9), # Feature map resolutions to apply attention to (e.g., 288/16=18, 288/32=9)
        attention_head_dim: int = 8,
    ):
        super().__init__()

        if num_classes is not None:
            print(f"[Model] `num_classes`={num_classes} is provided. This model will output logits for classification.")
            final_out_channels = num_classes
        else:
            print(f"[Model] `num_classes` not provided. Falling back to standard DDPM output with `out_channels`={out_channels}.")
            final_out_channels = out_channels

        # --- Parameter Translation for Diffusers UNet2DModel ---

        # 1. block_out_channels: The number of output channels for each U-Net block.
        block_out_channels = [model_channels * mult for mult in channel_mult]

        # 2. layers_per_block: The number of ResNet blocks in each U-Net block.
        layers_per_block = num_res_blocks

        # 3. down_block_types & up_block_types: Define the architecture, including attention.
        # We calculate the feature map size at each level to decide where to put attention.
        down_block_types = []
        current_res = image_size
        for i in range(len(block_out_channels)):
            if current_res in attention_resolutions:
                down_block_types.append("AttnDownBlock2D")
            else:
                down_block_types.append("DownBlock2D")
            if i < len(block_out_channels) -1: # Don't downsample on the last block
                 current_res //= 2
        
        # The up blocks are a mirror of the down blocks.
        up_block_types = [block_type.replace("Down", "Up") for block_type in reversed(down_block_types)]

        print(f"[Model Init] Diffusers UNet configuration:")
        print(f"  - block_out_channels: {block_out_channels}")
        print(f"  - layers_per_block: {layers_per_block}")
        print(f"  - down_block_types: {down_block_types}")
        print(f"  - up_block_types: {up_block_types}")
        print(f"  - attention_head_dim: {attention_head_dim}")


        # Instantiate the powerful diffusers model
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=final_out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            norm_num_groups=groups,
            attention_head_dim=attention_head_dim,
        )
        
        # For compatibility with the training script, expose final_out_channels
        self.final_out_channels = final_out_channels


    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        The forward pass call. It matches the signature required by the training script.
        
        Args:
            x (torch.Tensor): The input noisy image tensor. Shape (B, C, H, W).
            timesteps (torch.Tensor): The timestep tensor. Shape (B,).

        Returns:
            torch.Tensor: The predicted output logits. Shape (B, num_classes, H, W).
        """
        # The diffusers model expects `sample` and `timestep` as keyword arguments.
        # It returns a dataclass, and the actual output tensor is in the `.sample` attribute.
        return self.model(sample=x, timestep=timesteps).sample

# --- Test Section (Updated for the new model) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    img_size = 288
    batch_size = 2
    num_classes = 2 # For binary classification (background/foreground)

    print("\n--- Testing Attention U-Net (Diffusers Wrapper) Configuration ---")
    try:
        # Instantiate the new model
        model = AttentionUNetDenoisingModel(
            image_size=img_size,
            in_channels=1,
            num_classes=num_classes,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            groups=32,
            # Attention specific params
            attention_resolutions=(18, 9), # Apply attention at 18x18 and 9x9 feature maps
            attention_head_dim=8
        ).to(device)

        # The input to our discrete model is integer-like, but the model itself expects floats
        dummy_input_images = torch.randint(0, num_classes, (batch_size, 1, img_size, img_size), dtype=torch.long).to(device)
        dummy_timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)
        
        print(f"Input image shape: {dummy_input_images.shape} (dtype: {dummy_input_images.dtype})")
        print(f"Timesteps shape: {dummy_timesteps.shape}")
        
        with torch.no_grad():
            # The model forward pass requires float input
            predicted_logits = model(dummy_input_images.float(), dummy_timesteps)
            
        print(f"Predicted logits shape: {predicted_logits.shape}")
        expected_shape = (batch_size, num_classes, img_size, img_size)
        assert predicted_logits.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {predicted_logits.shape}"
        
        print("Model forward pass successful!")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

    except Exception as e:
        print(f"An error occurred during model test: {e}")
        import traceback
        traceback.print_exc()