# model.py (FIXED FOR LATEST DIFFUSERS)
import torch
import torch.nn as nn
from diffusers import UNet2DModel

class AttentionUNetDenoisingModel(nn.Module):
    def __init__( 
        self,
        image_size: int = 288,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = None,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim_ratio: int = 4, 
        groups: int = 32,
        attention_resolutions: tuple = (18, 9),
        attention_head_dim: int = 8,
    ):
        super().__init__()

        if num_classes is not None:
            print(f"[Model] `num_classes`={num_classes} is provided. This model will output logits for classification.")
            final_out_channels = num_classes
        else:
            print(f"[Model] `num_classes` not provided. Falling back to standard DDPM output with `out_channels`={out_channels}.")
            final_out_channels = out_channels

        # 参数转换
        block_out_channels = [model_channels * mult for mult in channel_mult]
        layers_per_block = num_res_blocks
        
        # 计算注意力位置
        down_block_types = []
        current_res = image_size
        for i in range(len(block_out_channels)):
            if current_res in attention_resolutions:
                down_block_types.append("AttnDownBlock2D")
            else:
                down_block_types.append("DownBlock2D")
            if i < len(block_out_channels) - 1: 
                current_res //= 2
        
        up_block_types = [block_type.replace("Down", "Up") for block_type in reversed(down_block_types)]

        print(f"[Model Init] Diffusers UNet configuration:")
        print(f"  - block_out_channels: {block_out_channels}")
        print(f"  - layers_per_block: {layers_per_block}")
        print(f"  - down_block_types: {down_block_types}")
        print(f"  - up_block_types: {up_block_types}")
        print(f"  - attention_head_dim: {attention_head_dim}")

        # 关键修复: 使用正确的参数名
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
            
            # 时间嵌入相关参数 (最新API)
            time_embedding_type="positional",  # 新增
            freq_shift=0,                      # 新增
            flip_sin_to_cos=False,             # 新增
        )
        
        self.final_out_channels = final_out_channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(sample=x, timestep=timesteps).sample

# 测试部分保持不变
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    img_size = 288
    batch_size = 2
    num_classes = 2

    print("\n--- Testing Attention U-Net (Diffusers Wrapper) Configuration ---")
    try:
        model = AttentionUNetDenoisingModel(
            image_size=img_size,
            in_channels=1,
            num_classes=num_classes,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            groups=32,
            attention_resolutions=(18, 9),
            attention_head_dim=8
        ).to(device)

        dummy_input_images = torch.randint(0, num_classes, (batch_size, 1, img_size, img_size), dtype=torch.long).to(device)
        dummy_timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)
        
        print(f"Input image shape: {dummy_input_images.shape} (dtype: {dummy_input_images.dtype})")
        print(f"Timesteps shape: {dummy_timesteps.shape}")
        
        with torch.no_grad():
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
