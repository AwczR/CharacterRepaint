# model.py (FINAL, ROBUST FIX)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Modules (No changes) ---
class SinusoidalPositionalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim > 1: timesteps = timesteps.squeeze(-1)
        if timesteps.dtype != torch.float32: timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(self.max_period) * torch.arange(half_dim, device=device)
        exponent = exponent / half_dim
        embedding_arg = timesteps[:, None] * exponent[None, :].exp()
        embedding = torch.cat([torch.sin(embedding_arg), torch.cos(embedding_arg)], dim=-1)
        if self.dim % 2 == 1: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if time_emb_dim is not None: self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        else: self.time_mlp = None
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.activation = nn.SiLU()
        if in_channels != out_channels: self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else: self.skip_connection = nn.Identity()
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h_res = x
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        if self.time_mlp is not None and t_emb is not None:
            time_cond = self.time_mlp(t_emb)
            h = h + time_cond[:, :, None, None]
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip_connection(h_res)

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        return self.conv(x)

# --- Main U-Net Model (FINAL, ROBUST FIX) ---

class UNetDenoisingModel(nn.Module):
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
    ):
        super().__init__()

        if num_classes is not None:
            print(f"[Model] `num_classes`={num_classes} is provided. This model will output logits for classification.")
            self.final_out_channels = num_classes
        else:
            print(f"[Model] `num_classes` not provided. Falling back to standard DDPM output with `out_channels`={out_channels}.")
            self.final_out_channels = out_channels

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        time_emb_dim_input = model_channels
        self.time_embedding = SinusoidalPositionalTimestepEmbedding(time_emb_dim_input)
        
        time_emb_dim_intermediate = model_channels * time_emb_dim_ratio
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim_input, time_emb_dim_intermediate),
            nn.SiLU(),
            nn.Linear(time_emb_dim_intermediate, time_emb_dim_intermediate)
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # --- Downsampling Path (Encoder) ---
        channels = [model_channels]
        now_channels = model_channels
        for i, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(
                    now_channels, out_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups
                ))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(now_channels))
                channels.append(now_channels)

        # --- Bottleneck (Middle) ---
        self.mid_block1 = ResnetBlock(now_channels, now_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups)
        self.mid_block2 = ResnetBlock(now_channels, now_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups)

        # --- Upsampling Path (Decoder) ---
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                in_channels = now_channels + channels.pop()
                self.up_blocks.append(ResnetBlock(
                    in_channels, out_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups
                ))
                now_channels = out_channels
            if i != 0:
                self.up_blocks.append(Upsample(now_channels))

        # --- Final Output Layer ---
        self.out_norm = nn.GroupNorm(groups, model_channels)
        self.out_activation = nn.SiLU()
        self.conv_out = nn.Conv2d(model_channels, self.final_out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)

        h = self.conv_in(x)
        
        skips = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h, t_emb)
            else:
                skip = skips.pop()
                h = torch.cat((h, skip), dim=1)
                h = block(h, t_emb)
        
        h = self.out_norm(h)
        h = self.out_activation(h)
        out = self.conv_out(h)
        
        return out

# --- Test Section (Unchanged) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    img_size = 288
    batch_size = 2
    num_classes = 2
    print("\n--- Testing Discrete U-Net Configuration ---")
    try:
        model = UNetDenoisingModel(
            image_size=img_size, in_channels=1, num_classes=num_classes,
            model_channels=64, channel_mult=(1, 2, 4, 8), num_res_blocks=2,
            time_emb_dim_ratio=4, groups=32
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