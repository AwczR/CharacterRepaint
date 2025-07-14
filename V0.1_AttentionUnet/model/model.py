# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Modules ---

class SinusoidalPositionalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)
        if timesteps.dtype != torch.float32:
            timesteps = timesteps.float()

        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(self.max_period) * torch.arange(half_dim, device=device)
        exponent = exponent / half_dim

        embedding_arg = timesteps[:, None] * exponent[None, :].exp()
        embedding = torch.cat([torch.sin(embedding_arg), torch.cos(embedding_arg)], dim=-1)

        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        
        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h_res = x # Store for residual connection

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


# --- Main U-Net Model ---

class UNetDenoisingModel(nn.Module):
    def __init__(
        self,
        image_size: int = 288,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim_ratio: int = 4,
        groups: int = 32,
    ):
        super().__init__()

        assert image_size == 288, "This model configuration is tuned for 288x288 images."
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult # <<< ADDED THIS
        self.num_res_blocks = num_res_blocks # <<< ADDED THIS
        
        time_emb_dim_input = model_channels # Input to the first Linear layer of time_mlp
        self.time_embedding = SinusoidalPositionalTimestepEmbedding(time_emb_dim_input)
        
        time_emb_dim_intermediate = model_channels * time_emb_dim_ratio
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim_input, time_emb_dim_intermediate),
            nn.SiLU(),
            nn.Linear(time_emb_dim_intermediate, time_emb_dim_intermediate) # This will be passed to ResNetBlocks
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # --- Downsampling Path (Encoder) ---
        current_channels = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlock(current_channels, out_ch, time_emb_dim=time_emb_dim_intermediate, groups=groups)
                )
                current_channels = out_ch
            if i != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(current_channels))
        
        # --- Bottleneck (Middle) ---
        self.mid_block1 = ResnetBlock(current_channels, current_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups)
        self.mid_block2 = ResnetBlock(current_channels, current_channels, time_emb_dim=time_emb_dim_intermediate, groups=groups)

        # --- Upsampling Path (Decoder) ---
        # current_channels is now the bottleneck channel count
        for i, mult in reversed(list(enumerate(channel_mult))):
            expected_in_ch_after_concat = model_channels * mult # Target channels for ResBlocks at this level (output)
            
            # The ResBlocks in up_blocks will take (upsampled_channels + skip_channels) as input
            # skip_channels will be channel_mult[i] * model_channels
            # upsampled_channels will be current_channels from previous up_block or bottleneck
            
            # Determine input channels for the ResBlocks in this upsampling stage
            # If this is the first upsampling stage (i == len(channel_mult) - 1),
            # then upsampled_channels is current_channels (from bottleneck).
            # Skip channel is also current_channels (output of last down ResBlock stage).
            # So, input to first ResBlock of up path is current_channels (from upsample op) + current_channels (from skip).
            
            # Let's consider the channel counts for ResBlocks in the up-path
            # Input to ResBlock: channels from previous upsample + channels from skip
            # Output of ResBlock: `model_channels * mult` (i.e., `expected_in_ch_after_concat`)
            
            # The channel count coming *into* the upsampling block (from the layer below it) is `current_channels`.
            # The skip connection for this level has `model_channels * mult` channels.
            res_block_in_channels = current_channels + (model_channels * mult)

            for _ in range(num_res_blocks): # num_res_blocks, not num_res_blocks + 1 here for symmetry with down_blocks
                self.up_blocks.append(
                    ResnetBlock(res_block_in_channels, expected_in_ch_after_concat, time_emb_dim=time_emb_dim_intermediate, groups=groups)
                )
                res_block_in_channels = expected_in_ch_after_concat # For next ResBlock in same level, if any
                current_channels = expected_in_ch_after_concat # This is the new current_channels for this level

            if i != 0: # Add Upsample to go to the next higher resolution, unless we are at the highest res
                self.up_blocks.append(Upsample(current_channels))
                # current_channels remains the output of Upsample for the next level's ResBlock input
        
        self.out_norm = nn.GroupNorm(groups, model_channels) # Output should be model_channels
        self.out_activation = nn.SiLU()
        self.conv_out = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

    def _debug_print_shapes(self, name, tensor):
        print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # self._debug_print_shapes("Initial x", x)
        # self._debug_print_shapes("Initial timesteps", timesteps)

        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_mlp(t_emb)
        # self._debug_print_shapes("Time embedding (t_emb)", t_emb)

        h = self.conv_in(x)
        # self._debug_print_shapes("After conv_in", h)
        
        encoder_skips = []
        down_block_module_idx = 0

        # --- Downsampling Path ---
        for i_level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                res_block = self.down_blocks[down_block_module_idx]
                h = res_block(h, t_emb)
                down_block_module_idx += 1
                # self._debug_print_shapes(f"Down ResBlock {i_level}-{_}", h)
            
            # Store skip *after* all ResBlocks for this level, *before* downsampling
            encoder_skips.append(h)
            
            if i_level < len(self.channel_mult) - 1: # If not the last level (bottleneck)
                downsampler = self.down_blocks[down_block_module_idx]
                h = downsampler(h)
                down_block_module_idx += 1
                # self._debug_print_shapes(f"Downsample {i_level}", h)
        
        # self._debug_print_shapes("After all down_blocks, input to mid", h)
        # print(f"Number of skip connections: {len(encoder_skips)}")
        # for i, s in enumerate(encoder_skips):
        #     self._debug_print_shapes(f"Encoder Skip {i}", s)

        h = self.mid_block1(h, t_emb)
        # self._debug_print_shapes("After mid_block1", h)
        h = self.mid_block2(h, t_emb)
        # self._debug_print_shapes("After mid_block2", h)

        up_block_module_idx = 0
        # --- Upsampling Path ---
        for i_level in range(len(self.channel_mult)):
            # Pop skip from corresponding encoder level (last skip is for highest res level of encoder)
            skip = encoder_skips.pop()
            # self._debug_print_shapes(f"Popped skip for up_level {len(self.channel_mult)-1-i_level}", skip)
            # self._debug_print_shapes(f"Current h before concat for up_level {len(self.channel_mult)-1-i_level}", h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                # self._debug_print_shapes(f"Resized h to match skip", h)

            h = torch.cat([h, skip], dim=1)
            # self._debug_print_shapes(f"After concat with skip", h)
            
            for _ in range(self.num_res_blocks):
                res_block = self.up_blocks[up_block_module_idx]
                h = res_block(h, t_emb)
                up_block_module_idx += 1
                # self._debug_print_shapes(f"Up ResBlock {len(self.channel_mult)-1-i_level}-{_}", h)

            if i_level < len(self.channel_mult) - 1: # If not the last upsampling stage (i.e., not outputting final res)
                upsampler = self.up_blocks[up_block_module_idx]
                h = upsampler(h)
                up_block_module_idx += 1
                # self._debug_print_shapes(f"Upsample {len(self.channel_mult)-1-i_level}", h)
        
        h = self.out_norm(h)
        h = self.out_activation(h)
        out = self.conv_out(h)
        # self._debug_print_shapes("Final output", out)
        
        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 288
    batch_size = 2
    
    print("\n--- Testing smaller configuration model ---")
    model = UNetDenoisingModel(
        image_size=img_size,
        in_channels=1,
        out_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 3, 4), # (32, 64, 96, 128)
        num_res_blocks=1,
        time_emb_dim_ratio=4,
        groups=16 # 32 % 16 == 0, 64 % 16 == 0, 96 % 16 == 0, 128 % 16 == 0. Ok.
                  # model_channels for out_norm is 32. 32 % 16 == 0. Ok.
    ).to(device)

    dummy_noisy_images = torch.randn(batch_size, 1, img_size, img_size).to(device)
    dummy_timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)

    print(f"Input image shape: {dummy_noisy_images.shape}")
    print(f"Timesteps shape: {dummy_timesteps.shape}")

    try:
        with torch.no_grad():
            predicted_noise = model(dummy_noisy_images, dummy_timesteps)
        print(f"Predicted noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == dummy_noisy_images.shape, "Output shape mismatch!"
        print("Model forward pass successful!")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"An error occurred during model test: {e}")
        import traceback
        traceback.print_exc()
        print("\n--- Debugging Model Initialization ---")
        print(f"model_channels: {model.model_channels}")
        if hasattr(model, 'channel_mult'): print(f"channel_mult: {model.channel_mult}")
        if hasattr(model, 'num_res_blocks'): print(f"num_res_blocks: {model.num_res_blocks}")
        # print(f"groups for GroupNorm (example from a down_block): {model.down_blocks[0].norm1.num_groups if model.down_blocks else 'N/A'}")

    print("\n--- Testing default parameters model ---")
    try:
        default_model = UNetDenoisingModel(image_size=img_size).to(device) # Uses default model_channels=64, num_res_blocks=2, etc.
        with torch.no_grad():
             predicted_noise_default = default_model(dummy_noisy_images, dummy_timesteps)
        print(f"Default Predicted noise shape: {predicted_noise_default.shape}")
        assert predicted_noise_default.shape == dummy_noisy_images.shape, "Output shape mismatch!"
        total_params_default = sum(p.numel() for p in default_model.parameters())
        print(f"Default Total parameters: {total_params_default:,}")
        print("Default Model forward pass successful!")
    except Exception as e:
        print(f"An error occurred during default model test: {e}")
        import traceback
        traceback.print_exc()