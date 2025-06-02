import os
import json
import yaml
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# 从你的 model.py 文件导入模型
from model import UNetDenoisingModel

# --- Diffusion Helper Functions ---

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """Extracts coefficients at specified timesteps and reshapes them to broadcast across images."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    Forward diffusion process: q(x_t | x_0)
    x_start: clean image (batch_size, channels, height, width)
    t: timesteps (batch_size,)
    noise: Optional noise tensor. If None, generates standard Gaussian noise.
    Returns noisy image x_t.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- Dataset Class ---

class RubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288)):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.base_noisy_dir = self.metadata['base_noisy_dir']
        self.base_clean_dir = self.metadata['base_clean_dir']
        self.image_pairs = self.metadata['image_pairs']
        
        # 图像通常需要归一化到 [-1, 1] 范围给扩散模型
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1), # 确保是单通道
            transforms.ToTensor(), # 转换为 [0, 1] 的张量
            transforms.Normalize((0.5,), (0.5,)) # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        
        # 我们只需要 clean image，noisy image 会在训练中动态生成
        clean_path = os.path.join(self.base_clean_dir, pair_info['clean_path_relative'])
        
        try:
            # 确保以 'L' 模式打开，即使元数据校验时是 'L'，以防万一
            clean_image = Image.open(clean_path).convert('L') 
            clean_image = self.transform(clean_image)
            return clean_image
        except Exception as e:
            print(f"Error loading image {clean_path}: {e}")
            # 返回一个占位符或者跳过，这里简单返回 None，DataLoader 的 collate_fn 可能需要处理
            # 更好的做法是在 create_metadata.py 中彻底过滤掉坏数据
            return None 

def custom_collate_fn(batch):
    """自定义collate_fn来过滤掉None值（加载失败的图像）"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # 或者引发错误，表示批次为空
    return torch.utils.data.dataloader.default_collate(batch)


# --- Visualization & Saving ---

def save_loss_plot(losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

@torch.no_grad()
def sample_denoised_images(model, diffusion_params, num_images, image_size, device, epoch, output_dir, fixed_noise_tensor=None):
    """
    Generates images from noise using the reverse diffusion process (DDPM sampling).
    """
    model.eval()
    
    timesteps = diffusion_params['num_timesteps']
    betas = diffusion_params['betas']
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Not used in basic DDPM sampling directly
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - torch.cat([torch.tensor([0.0], device=device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod) # Simplified for DDPM

    img_shape = (num_images, model.in_channels, image_size, image_size)

    if fixed_noise_tensor is not None and fixed_noise_tensor.shape == img_shape:
        img = fixed_noise_tensor.to(device)
    else:
        img = torch.randn(img_shape, device=device)

    imgs_sequence = [img.clone()] # Store initial noise

    for t_idx in tqdm(reversed(range(0, timesteps)), desc="Sampling", total=timesteps, leave=False):
        t = torch.full((num_images,), t_idx, device=device, dtype=torch.long)
        
        predicted_noise = model(img, t)
        
        # DDPM sampling step
        # x_t-1 = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta) + sqrt(beta_t) * z
        # where z is new noise if t > 0, else 0.
        
        model_mean = sqrt_recip_alphas[t_idx] * (
            img - (betas[t_idx] / sqrt_one_minus_alphas_cumprod[t_idx]) * predicted_noise
        )
        
        if t_idx == 0:
            img = model_mean # No noise added at the last step
        else:
            # For DDPM, posterior_variance_t is beta_t effectively when variance is fixed
            # Or more precisely, (1-alpha_cumprod_prev)/(1-alpha_cumprod) * beta_t
            # Here we use the simpler DDPM posterior variance for fixed small variance: beta_t
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance[t_idx]) * noise 
        imgs_sequence.append(img.clone())

    # Denormalize from [-1, 1] to [0, 1] for saving
    img_out = (img.clamp(-1, 1) + 1) / 2
    
    save_path = os.path.join(output_dir, f"epoch_{epoch:04d}_samples.png")
    save_image(img_out, save_path, nrow=int(num_images**0.5))
    print(f"Saved sampled images to {save_path}")
    
    # Optionally save a gif of the denoising process for one image
    # from matplotlib.animation import FuncAnimation, PillowWriter
    # fig_anim, ax_anim = plt.subplots()
    # def update(frame_idx):
    #     ax_anim.clear()
    #     img_to_show = (imgs_sequence[frame_idx* (len(imgs_sequence)//100)][0].cpu().clamp(-1,1) + 1)/2 # Show 1st image, 100 frames
    #     ax_anim.imshow(img_to_show.permute(1,2,0).squeeze(), cmap='gray')
    #     ax_anim.set_title(f"Denoising step {frame_idx * (len(imgs_sequence)//100)}")
    #     ax_anim.axis('off')
    # ani = FuncAnimation(fig_anim, update, frames=100, interval=50)
    # ani_save_path = os.path.join(output_dir, f"epoch_{epoch:04d}_denoising_process.gif")
    # ani.save(ani_save_path, writer=PillowWriter(fps=20))
    # plt.close(fig_anim)
    # print(f"Saved denoising animation to {ani_save_path}")

    model.train() # Set back to train mode


# --- Training Function ---

def train(config):
    # Setup
    output_dir = config['training_params']['output_dir']
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = RubbingsDataset(
        metadata_path=config['dataset']['metadata_path'],
        image_size=(config['model_params']['image_size'], config['model_params']['image_size'])
    )
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check metadata path and data.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True, 
        num_workers=config['dataset']['num_workers'],
        collate_fn=custom_collate_fn, # Handle potential None from dataset
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Dataset loaded: {len(dataset)} images.")

    # Model
    model = UNetDenoisingModel(
        image_size=config['model_params']['image_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        model_channels=config['model_params']['model_channels'],
        channel_mult=tuple(config['model_params']['channel_mult']), # Ensure it's a tuple
        num_res_blocks=config['model_params']['num_res_blocks'],
        time_emb_dim_ratio=config['model_params']['time_emb_dim_ratio'],
        groups=config['model_params']['groups']
    ).to(device)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Diffusion parameters
    num_timesteps = config['diffusion_params']['num_timesteps']
    betas = linear_beta_schedule(
        num_timesteps, 
        config['diffusion_params']['beta_start'], 
        config['diffusion_params']['beta_end']
    ).to(device)
    
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    diffusion_sampling_params = { # For visualization
        'num_timesteps': num_timesteps,
        'betas': betas,
        # Add other params needed by sample_denoised_images if any
    }


    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training_params']['lr'])
    criterion = nn.MSELoss() # Common loss for DDPMs (predict noise)

    losses_history = []
    
    # Fixed noise for consistent visualization of samples
    fixed_noise_for_vis = None
    if config['visualization_params']['fixed_noise_for_samples']:
        fixed_noise_for_vis = torch.randn(
            (config['visualization_params']['num_samples_to_visualize'],
             config['model_params']['in_channels'],
             config['model_params']['image_size'],
             config['model_params']['image_size']),
            device=device
        )


    # Training Loop
    print("Starting training...")
    for epoch in range(config['training_params']['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training_params']['epochs']}", leave=False)
        for batch_idx, clean_images in enumerate(progress_bar):
            if clean_images is None: # Skips batch if all images failed to load
                print(f"Skipping empty batch at epoch {epoch+1}, batch index {batch_idx}")
                continue
            
            clean_images = clean_images.to(device)
            batch_size = clean_images.shape[0]

            # Sample timesteps uniformly for each image in the batch
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            # Generate noise and create noisy images (forward process)
            noise = torch.randn_like(clean_images)
            noisy_images = q_sample(clean_images, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)
            
            # Predict noise using the model
            predicted_noise = model(noisy_images, t)
            
            # Calculate loss
            loss = criterion(predicted_noise, noise) # Target is the original noise
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{config['training_params']['epochs']}, Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config['training_params']['save_checkpoint_freq'] == 0 or (epoch + 1) == config['training_params']['epochs']:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': config # Save config for reproducibility
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Save visualizations
        if (epoch + 1) % config['training_params']['save_visualization_freq'] == 0 or (epoch + 1) == config['training_params']['epochs']:
            # Loss plot
            loss_plot_path = os.path.join(visualizations_dir, "loss_plot.png")
            save_loss_plot(losses_history, loss_plot_path)
            print(f"Saved loss plot to {loss_plot_path}")
            
            # Sampled images
            sample_denoised_images(
                model, 
                diffusion_sampling_params,
                num_images=config['visualization_params']['num_samples_to_visualize'],
                image_size=config['model_params']['image_size'],
                device=device,
                epoch=epoch+1,
                output_dir=visualizations_dir,
                fixed_noise_tensor=fixed_noise_for_vis
            )
            
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net Denoising Diffusion Model for Rubbings Inpainting.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)
    
    train(config_data)