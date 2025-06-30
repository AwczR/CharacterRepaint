# train.py (MAJOR REVISION - with PSNR/SSIM and Full Sampling)
"""
U-Net based Discrete Diffusion Probabilistic Model (D3PM) training script.
This version includes:
- PSNR/SSIM metrics for evaluating image restoration quality.
- A full denoising sampling loop during training for high-quality visualization.
"""
import os
import json
import yaml
import argparse
import math
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchmetrics
# <<<< NEW >>>> Import PSNR and SSIM metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import UNetDenoisingModel

# --- 1. DISCRETE DIFFUSION HELPER FUNCTIONS (Unchanged) ---
def cosine_betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    def alpha_bar_fn(t): return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1, t2 = i / num_diffusion_timesteps, (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def q_sample_discrete(x_start, t, q_bar_t):
    num_classes = q_bar_t.shape[-1]
    x_start_one_hot = F.one_hot(x_start.squeeze(1), num_classes=num_classes).float()
    b, h, w, c = x_start_one_hot.shape
    x_start_one_hot_flat = x_start_one_hot.view(b, h * w, c)
    xt_probs_flat = torch.bmm(x_start_one_hot_flat, q_bar_t)
    xt_probs_for_sampling = xt_probs_flat.view(b * h * w, c).clamp_(min=0)
    sampled_xt_flat = torch.multinomial(xt_probs_for_sampling, num_samples=1)
    return sampled_xt_flat.view(b, h, w).unsqueeze(1)

# --- 2. DATASET AND DATALOADER (Unchanged) ---
class RubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288), binarization_threshold=0.5):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f: self.metadata = json.load(f)
        except FileNotFoundError: raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
        self.base_clean_dir = self.metadata["base_clean_dir"]
        self.image_pairs = self.metadata["image_pairs"]
        if not os.path.isdir(self.base_clean_dir): print(f"Warning: Base directory '{self.base_clean_dir}' not found.")
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), lambda x: (x > binarization_threshold).long()])
    def __len__(self): return len(self.image_pairs)
    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        full_clean_path = os.path.join(self.base_clean_dir, pair_info["clean_path_relative"])
        try: return self.transform(Image.open(full_clean_path).convert("L"))
        except Exception as e: print(f"\n[Dataset Error] Skipping image {full_clean_path}. Reason: {e}"); return None

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- 3. PLOTTING FUNCTIONS (MODIFIED) ---
def save_plot(df, y_column, title, y_label, filename):
    plt.figure(figsize=(12, 6)); plt.plot(df["epoch"], df[y_column], label=y_label, color='royalblue', marker='.')
    plt.xlabel("Epoch"); plt.ylabel(y_label); plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(filename); plt.close()

def save_metrics_plot(df, output_dir):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Plot classification metrics on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Classification Metrics (0-1)', color='tab:blue')
    cls_metrics = {'accuracy': 'green', 'f1_score': 'red', 'precision': 'purple', 'recall': 'orange'}
    for metric, color in cls_metrics.items():
        if metric in df.columns: ax1.plot(df["epoch"], df[metric], label=metric.capitalize(), color=color, marker='o', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    
    # Create a second y-axis for PSNR
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='tab:red')
    if 'psnr' in df.columns: ax2.plot(df["epoch"], df['psnr'], label='PSNR', color='tab:red', marker='s', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title("Model Performance Metrics Over Epochs")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "metrics_plot.png"))
    plt.close()

# --- 4. FULL SAMPLING & VISUALIZATION (NEW/REFACTORED) ---

# train.py (FIX for RuntimeError in q_sample_discrete)
@torch.no_grad()
# train.py -> p_sample_loop_restoration (FINAL, VERIFIED FIX)
def p_sample_loop_restoration(model, device, config, q_mats, input_image, start_timestep):
    model.eval()

    # --- 1. Noise the input image to the start_timestep ---
    print("Applying controlled noise to the input image up to timestep t={}...".format(start_timestep))
    
    batch_size = input_image.shape[0]
    t_tensor = torch.full((batch_size,), start_timestep - 1, device=device, dtype=torch.long)
    
    q_bar_t_start_2d = q_mats['q_mats_cumprod'][start_timestep - 1]
    q_bar_t_start = q_bar_t_start_2d.expand(batch_size, -1, -1)
    
    img = q_sample_discrete(input_image, t_tensor, q_bar_t_start)
    print("Controlled noising complete. Starting denoising loop from this state.")
    
    # --- 2. Denoise from start_timestep back to 0 ---
    loop_range = reversed(range(start_timestep))
    
    for t in tqdm(loop_range, desc="Visualizing", total=start_timestep, leave=False):
        time = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
        pred_x0_logits = model(img.float(), time)
        
        if t == 0:
            img = torch.argmax(pred_x0_logits, dim=1).unsqueeze(1)
            break
        
        # ==============================================================================
        # <<<< THIS IS THE CRITICAL FIX INSIDE THE LOOP >>>>
        # We must expand the 2D matrices to match the batch size for every step.
        b, _, h, w = img.shape
        
        q_bar_t_minus_1_2d = q_mats['q_mats_cumprod'][t-1] if t > 0 else torch.eye(model.final_out_channels, device=device)
        q_bar_t_minus_1 = q_bar_t_minus_1_2d.expand(b, -1, -1)

        q_t_2d = q_mats['q_one_step_mats'][t]
        q_t = q_t_2d.expand(b, -1, -1)
        # =================================S=============================================
        
        x_t_one_hot_flat = F.one_hot(img.squeeze(1), num_classes=model.final_out_channels).float().view(b, h*w, -1)
        pred_x0_probs_flat = F.softmax(pred_x0_logits, dim=1).permute(0, 2, 3, 1).reshape(b, h*w, -1)
        
        # The bmm calls will now work correctly because all tensors have batch size `b`
        term1 = torch.bmm(x_t_one_hot_flat, q_t)
        term2 = torch.bmm(pred_x0_probs_flat, q_bar_t_minus_1)
        
        posterior_probs_log = torch.log(term1 + 1e-8) + torch.log(term2 + 1e-8)
        sampled_flat = torch.multinomial(F.softmax(posterior_probs_log.view(b*h*w, -1), dim=-1), num_samples=1)
        img = sampled_flat.view(b, 1, h, w)
    
    model.train()
    return img

# --- 5. SCHEDULER BUILDER (Unchanged) ---
def build_scheduler(optimizer, sched_cfg, total_epochs):
    params = sched_cfg.copy(); name = params.pop("name").lower()
    if name == "steplr": return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == "cosine":
        if "t_max" not in params: params["t_max"] = total_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else: raise ValueError(f"Unsupported scheduler: {name}")

# --- 6. MAIN TRAINING FUNCTION (HEAVILY MODIFIED) ---
# train.py -> train function (REVISED)

def train(config):
    # --- Setup ---
    output_dir = config["training_params"]["output_dir"]; ckpt_dir = os.path.join(output_dir, "checkpoints")
    vis_dir = os.path.join(output_dir, "visualizations"); os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(vis_dir, exist_ok=True)
    device = torch.device(config["training_params"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")

    # --- Data (Split into Train and a small Validation set) ---
    full_dataset = RubbingsDataset(metadata_path=config["dataset"]["metadata_path"], image_size=(config["model_params"]["image_size"], config["model_params"]["image_size"]))
    if len(full_dataset) < 10: raise ValueError("Dataset too small. Need at least 10 images for train/val split.")
    val_size = max(1, int(0.1 * len(full_dataset))) # Use 10% or at least 1 image for validation
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"], collate_fn=custom_collate_fn, pin_memory=True if device.type=='cuda' else False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["visualization_params"]["num_samples_to_visualize"], shuffle=False)
    fixed_val_batch = next(iter(val_dataloader)).to(device)
    print(f"[Init] Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation images.")

    # --- Model ---
    NUM_CLASSES = 2; model_kwargs = config["model_params"].copy(); model_kwargs['num_classes'] = NUM_CLASSES
    model = UNetDenoisingModel(**model_kwargs).to(device)
    print(f"[Init] Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Diffusion Constants ---
    num_timesteps = config["diffusion_params"]["num_timesteps"]; betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]; q_one_step_mats[i, 0, 0] = 1.0 - beta_t; q_one_step_mats[i, 1, 1] = 1.0 - beta_t
        q_one_step_mats[i, 0, 1] = beta_t; q_one_step_mats[i, 1, 0] = beta_t
    q_mats_cumprod = torch.zeros_like(q_one_step_mats); current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps): current_mat = torch.matmul(q_one_step_mats[i], current_mat); q_mats_cumprod[i] = current_mat
    q_mats = {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}
    
    # --- Optimizer, Scheduler, Loss, and Metrics ---
    optimizer = optim.AdamW(model.parameters(), lr=config["training_params"]["lr"])
    scheduler = build_scheduler(optimizer, config["training_params"].get("scheduler"), config["training_params"]["epochs"]) if config["training_params"].get("scheduler") else None
    criterion = nn.CrossEntropyLoss()
    cls_metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="binary"), 'precision': torchmetrics.Precision(task="binary"), 'recall': torchmetrics.Recall(task="binary"), 'f1_score': torchmetrics.F1Score(task="binary")}).to(device)
    img_metrics = torchmetrics.MetricCollection({'psnr': PeakSignalNoiseRatio(data_range=1.0), 'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)}).to(device)

    # --- Training Loop ---
    history = []
    print("\n[Train] Starting training...")
    for epoch in range(config["training_params"]["epochs"]):
        model.train(); cls_metrics.reset(); epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['training_params']['epochs']}", leave=False)
        for clean_images in pbar:
            if clean_images is None: continue
            clean_images = clean_images.to(device); bsz = clean_images.size(0); optimizer.zero_grad()
            t = torch.randint(0, num_timesteps, (bsz,), device=device).long()
            q_bar_t = q_mats_cumprod[t]; noisy_images = q_sample_discrete(clean_images, t, q_bar_t)
            predicted_x0_logits = model(noisy_images.float(), t); loss = criterion(predicted_x0_logits, clean_images.squeeze(1))
            loss.backward(); optimizer.step()
            epoch_loss += loss.item(); preds = torch.argmax(predicted_x0_logits, dim=1); cls_metrics.update(preds, clean_images.squeeze(1))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- End of Epoch Evaluation ---
        avg_loss = epoch_loss / len(train_dataloader); current_lr = optimizer.param_groups[0]["lr"]
        
        # <<<< FIX: Use .items() to iterate over the dictionary >>>>
        epoch_cls_metrics = {k: v.item() for k, v in cls_metrics.compute().items()}
        
        epoch_img_metrics = {} # Initialize as empty dict
        if (epoch + 1) % config["training_params"]["save_visualization_freq"] == 0 or (epoch + 1) == config["training_params"]["epochs"]:
            print("Running full sampling for visualization and image metrics...")
            restored_images = p_sample_loop_restoration(model, device, config, q_mats, fixed_val_batch, config["visualization_params"]["start_timestep"])
            
            img_metrics.update(restored_images.float(), fixed_val_batch.float())
            
            # <<<< FIX: Use .items() here as well >>>>
            epoch_img_metrics = {k: v.item() for k, v in img_metrics.compute().items()}
            
            vis_folder = os.path.join(vis_dir, "epoch_samples")
            os.makedirs(vis_folder, exist_ok=True)
            comparison_grid = torch.cat([fixed_val_batch, restored_images], dim=0)
            save_image(comparison_grid.float(), os.path.join(vis_folder, f"epoch_{epoch+1:04d}_comparison.png"), nrow=fixed_val_batch.shape[0])
        else:
            # For epochs where we don't calculate, we can fill with None or skip
            epoch_img_metrics = {'psnr': None, 'ssim': None} 

        # --- Logging ---
        log_entry = {"epoch": epoch + 1, "loss": avg_loss, "lr": current_lr, **epoch_cls_metrics, **epoch_img_metrics}
        history.append(log_entry)
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_cls_metrics.items()])
        if epoch_img_metrics.get('psnr') is not None: metrics_str += f" | psnr: {epoch_img_metrics['psnr']:.2f} | ssim: {epoch_img_metrics['ssim']:.4f}"
        print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | {metrics_str} | LR: {current_lr:.2e}")

        if scheduler: scheduler.step()
        
        # --- Save Checkpoints and Plots ---
        if (epoch + 1) % config["training_params"]["save_checkpoint_freq"] == 0 or (epoch + 1) == config["training_params"]["epochs"]:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:04d}.pth")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'config': config}, ckpt_path)
            print(f"    -> Checkpoint saved to {ckpt_path}")
        
        if (epoch + 1) % config["training_params"]["save_visualization_freq"] == 0 or (epoch + 1) == config["training_params"]["epochs"]:
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
            save_plot(history_df.dropna(subset=['loss']), "loss", "Training Loss Over Epochs", "Loss", os.path.join(vis_dir, "loss_plot.png"))
            save_metrics_plot(history_df.dropna(subset=['psnr', 'ssim']), vis_dir)

# --- 7. COMMAND-LINE INTERFACE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GRM using a Discrete Diffusion Model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the training configuration YAML file.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: raise FileNotFoundError(f"Config file not found at '{args.config}'.")
    output_dir = config["training_params"]["output_dir"]; os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config_snapshot.yaml'), 'w') as f: yaml.dump(config, f)
    train(config)