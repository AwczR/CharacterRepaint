# train.py (ROBUST EVALUATION & CLASS WEIGHTING VERSION)
"""
U-Net based Discrete Diffusion Probabilistic Model (D3PM) training script.
This version features:
- Robust evaluation on the entire validation set for accurate performance tracking.
- Automatic class weighting to handle data imbalance.
- Separate logging for training and validation metrics to detect overfitting.
"""
import os
import json
import yaml
import argparse
import math
from tqdm import tqdm

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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import UNetDenoisingModel

# --- 1. DISCRETE DIFFUSION HELPER FUNCTIONS (No changes) ---
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

# --- 2. DATASET AND DATALOADER (NO AUGMENTATION) ---
# <<<< 修改 >>>> 恢复为原始的、不带数据增强的Dataset
class RubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288), binarization_threshold=0.5):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f: self.metadata = json.load(f)
        except FileNotFoundError: raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
        self.base_clean_dir = self.metadata["base_clean_dir"]
        self.image_pairs = self.metadata["image_pairs"]
        if not os.path.isdir(self.base_clean_dir): print(f"Warning: Base directory '{self.base_clean_dir}' not found.")
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            lambda x: (x > binarization_threshold).long()
        ])
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

# --- 3. PLOTTING FUNCTIONS (MODIFIED FOR VAL METRICS) ---
def save_plot(df, y_column, title, y_label, filename):
    plt.figure(figsize=(12, 6))
    
    # 同样进行数据有效性检查
    plot_df = df.dropna(subset=[y_column])
    if not plot_df.empty:
        plt.plot(plot_df["epoch"], plot_df[y_column], label=y_label, color='royalblue', marker='.')
    
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def save_metrics_plot(df, output_dir):
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Classification Metrics (0-1)', color='tab:blue')
    
    cls_metrics = {'val_accuracy': 'green', 'val_f1_score': 'red', 'val_precision': 'purple', 'val_recall': 'orange'}
    for metric, color in cls_metrics.items():
        if metric in df.columns:
            # <<<< 关键修复：先筛选出包含有效指标的行，再进行绘图 >>>>
            plot_df = df.dropna(subset=[metric])
            if not plot_df.empty: # 确保筛选后还有数据可画
                label_name = metric.replace('val_', '').capitalize().replace('_', ' ')
                ax1.plot(plot_df["epoch"], plot_df[metric], label=label_name, color=color, marker='o', linestyle='--')
            
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # 保证图例不为空时才显示
    if ax1.get_legend_handles_labels()[1]:
        ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='tab:red')
    if 'val_psnr' in df.columns:
        # <<<< 关键修复：同样对PSNR进行筛选 >>>>
        plot_df = df.dropna(subset=['val_psnr'])
        if not plot_df.empty:
            ax2.plot(plot_df["epoch"], plot_df['val_psnr'], label='PSNR', color='tab:red', marker='s', linestyle=':')
    
    ax2.tick_params(axis='y', labelcolor='tab:red')
    # 保证图例不为空时才显示
    if ax2.get_legend_handles_labels()[1]:
        ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title("Model Performance Metrics Over Epochs (on Validation Set)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "validation_metrics_plot.png"))
    plt.close()

# --- 4. FULL SAMPLING & VISUALIZATION (No changes) ---

@torch.no_grad()
def p_sample_loop_restoration(model, device, config, q_mats, input_image, start_timestep):
    model.eval()
    batch_size = input_image.shape[0]
    t_tensor = torch.full((batch_size,), start_timestep - 1, device=device, dtype=torch.long)
    q_bar_t_start_2d = q_mats['q_mats_cumprod'][start_timestep - 1]
    q_bar_t_start = q_bar_t_start_2d.expand(batch_size, -1, -1)
    
    # <<<< 修改 1 >>>> 保存初始的噪声图像
    initial_noisy_img = q_sample_discrete(input_image, t_tensor, q_bar_t_start)
    img = initial_noisy_img.clone() # 使用克隆的副本进行去噪
    
    loop_range = reversed(range(start_timestep))
    
    for t in tqdm(loop_range, desc="Denoising", total=start_timestep, leave=False):
        time = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
        pred_x0_logits = model(img.float(), time)
        if t == 0:
            img = torch.argmax(pred_x0_logits, dim=1).unsqueeze(1)
            break
        b, _, h, w = img.shape
        q_bar_t_minus_1_2d = q_mats['q_mats_cumprod'][t-1] if t > 0 else torch.eye(model.final_out_channels, device=device)
        q_bar_t_minus_1 = q_bar_t_minus_1_2d.expand(b, -1, -1)
        q_t_2d = q_mats['q_one_step_mats'][t]
        q_t = q_t_2d.expand(b, -1, -1)
        x_t_one_hot_flat = F.one_hot(img.squeeze(1), num_classes=model.final_out_channels).float().view(b, h*w, -1)
        pred_x0_probs_flat = F.softmax(pred_x0_logits, dim=1).permute(0, 2, 3, 1).reshape(b, h*w, -1)
        term1 = torch.bmm(x_t_one_hot_flat, q_t)
        term2 = torch.bmm(pred_x0_probs_flat, q_bar_t_minus_1)
        posterior_probs_log = torch.log(term1 + 1e-8) + torch.log(term2 + 1e-8)
        sampled_flat = torch.multinomial(F.softmax(posterior_probs_log.view(b*h*w, -1), dim=-1), num_samples=1)
        img = sampled_flat.view(b, 1, h, w)
    
    model.train()
    # <<<< 修改 2 >>>> 返回两个值
    return img, initial_noisy_img

# --- 5. SCHEDULER BUILDER (No changes) ---
def build_scheduler(optimizer, sched_cfg, total_epochs):
    params = sched_cfg.copy(); name = params.pop("name").lower()
    if name == "steplr": return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == "cosine":
        if "t_max" not in params: params["t_max"] = total_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else: raise ValueError(f"Unsupported scheduler: {name}")

# --- 6. HELPER FUNCTIONS (NEW/MODIFIED) ---
# <<<< 新增 >>>> 计算类别权重的函数
def calculate_class_weights(dataset):
    print("Calculating class weights for loss function...")
    counts = torch.zeros(2) # [count_class_0, count_class_1]
    # 如果是Subset，需要访问其.dataset属性
    iterable_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    for i in tqdm(range(len(iterable_dataset)), desc="Analyzing dataset"):
        item = iterable_dataset[i]
        if item is not None:
            counts[0] += torch.sum(item == 0)
            counts[1] += torch.sum(item == 1)
    
    if counts[1] == 0 or counts[0] == 0: return None
    
    total_pixels = counts.sum()
    weights = total_pixels / (2 * counts) # Inverse frequency
    print(f"Class counts: Background(0)={int(counts[0])}, Foreground(1)={int(counts[1])}")
    print(f"Calculated weights for CrossEntropyLoss: {weights.tolist()}")
    return weights

# <<<< 新增 >>>> 在整个验证集上进行评估的函数
# 用下面的代码替换掉您现有的 evaluate 函数

@torch.no_grad()
def evaluate(model, dataloader, criterion, cls_metrics, img_metrics, device, config, q_mats):
    model.eval()
    cls_metrics.reset()
    img_metrics.reset()
    total_loss = 0.0
    
    # <<<< 修改 1 >>>> 用于保存第一个批次的可视化结果
    visualization_batch = None

    for i, clean_images in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        if clean_images is None: continue
        clean_images = clean_images.to(device)

        # 运行完整的采样修复过程
        restored_images, noisy_images = p_sample_loop_restoration(
            model, device, config, q_mats, clean_images, config["visualization_params"]["start_timestep"]
        )
        
        # <<<< 修改 2 >>>> 如果是第一个批次，保存它的所有图像用于可视化
        if i == 0:
            visualization_batch = (clean_images.cpu(), noisy_images.cpu(), restored_images.cpu())

        # 评估Loss
        t = torch.zeros(clean_images.size(0), device=device, dtype=torch.long)
        predicted_x0_logits = model(restored_images.float(), t)
        loss = criterion(predicted_x0_logits, clean_images.squeeze(1))
        total_loss += loss.item()
        
        # 更新指标
        cls_metrics.update(restored_images, clean_images)
        img_metrics.update(restored_images.float(), clean_images.float())

    avg_loss = total_loss / len(dataloader)
    epoch_cls_metrics = {f"val_{k}": v.item() for k, v in cls_metrics.compute().items()}
    epoch_img_metrics = {f"val_{k}": v.item() for k, v in img_metrics.compute().items()}
    
    all_val_metrics = {**epoch_cls_metrics, **epoch_img_metrics}
    
    # <<<< 修改 3 >>>> 返回三个值
    return avg_loss, all_val_metrics, visualization_batch

# --- 7. MAIN TRAINING FUNCTION (HEAVILY MODIFIED) ---
# 在您的 train.py 文件中，用下面的代码替换掉整个 train 函数。
# 文件的其他部分（导入、辅助函数等）保持不变。
# 用下面的代码替换掉您现有的 train 函数

def train(config):
    # --- Setup and Data loading (no changes here) ---
    output_dir = config["training_params"]["output_dir"]; ckpt_dir = os.path.join(output_dir, "checkpoints")
    vis_dir = os.path.join(output_dir, "visualizations"); os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(vis_dir, exist_ok=True)
    device = torch.device(config["training_params"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")
    full_dataset = RubbingsDataset(metadata_path=config["dataset"]["metadata_path"], image_size=(config["model_params"]["image_size"], config["model_params"]["image_size"]))
    if len(full_dataset) < 10: raise ValueError("Dataset too small.")
    val_size = max(4, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True, num_workers=config["dataset"]["num_workers"], collate_fn=custom_collate_fn, pin_memory=True if device.type=='cuda' else False, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["visualization_params"]["num_samples_to_visualize"], shuffle=False, collate_fn=custom_collate_fn)
    # fixed_val_batch is no longer needed for visualization, but can be kept for other debug purposes if you want.
    
    # --- Model, Diffusion, Optimizer (no changes here) ---
    NUM_CLASSES = 2; model_kwargs = config["model_params"].copy(); model_kwargs['num_classes'] = NUM_CLASSES
    model = UNetDenoisingModel(**model_kwargs).to(device)
    print(f"[Init] Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    num_timesteps = config["diffusion_params"]["num_timesteps"]; betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]; q_one_step_mats[i, 0, 0] = 1.0 - beta_t; q_one_step_mats[i, 1, 1] = 1.0 - beta_t; q_one_step_mats[i, 0, 1] = beta_t; q_one_step_mats[i, 1, 0] = beta_t
    q_mats_cumprod = torch.zeros_like(q_one_step_mats); current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps): current_mat = torch.matmul(q_one_step_mats[i], current_mat); q_mats_cumprod[i] = current_mat
    q_mats = {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}
    optimizer = optim.AdamW(model.parameters(), lr=config["training_params"]["lr"])
    scheduler = build_scheduler(optimizer, config["training_params"].get("scheduler"), config["training_params"]["epochs"]) if config["training_params"].get("scheduler") else None
    class_weights = calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    train_cls_metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="binary"), 'f1_score': torchmetrics.F1Score(task="binary")}).to(device)
    val_cls_metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="binary"), 'precision': torchmetrics.Precision(task="binary"), 'recall': torchmetrics.Recall(task="binary"), 'f1_score': torchmetrics.F1Score(task="binary")}).to(device)
    val_img_metrics = torchmetrics.MetricCollection({'psnr': PeakSignalNoiseRatio(data_range=1.0), 'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)}).to(device)

    # --- Training Loop ---
    history = []
    print("\n[Train] Starting training...")
    torch.cuda.empty_cache() # <<<< 安全措施：在开始训练前清理缓存
    for epoch in range(config["training_params"]["epochs"]):
        # --- Training part of the loop (no changes) ---
        model.train()
        train_cls_metrics.reset()
        epoch_train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['training_params']['epochs']}", leave=True) 
        for clean_images in pbar:
            if clean_images is None: continue
            clean_images = clean_images.to(device); bsz = clean_images.size(0); optimizer.zero_grad()
            t = torch.randint(0, num_timesteps, (bsz,), device=device).long()
            q_bar_t = q_mats_cumprod[t]; noisy_images = q_sample_discrete(clean_images, t, q_bar_t)
            predicted_x0_logits = model(noisy_images.float(), t); loss = criterion(predicted_x0_logits, clean_images.squeeze(1))
            loss.backward(); optimizer.step()
            epoch_train_loss += loss.item()
            preds = torch.argmax(predicted_x0_logits, dim=1)
            train_cls_metrics.update(preds, clean_images.squeeze(1))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        epoch_train_metrics = {f"train_{k}": v.item() for k, v in train_cls_metrics.compute().items()}
        current_lr = optimizer.param_groups[0]["lr"]
        log_entry = {"epoch": epoch + 1, "train_loss": avg_train_loss, "lr": current_lr, **epoch_train_metrics}

        if (epoch + 1) % config["training_params"]["save_visualization_freq"] == 0 or (epoch + 1) == config["training_params"]["epochs"]:
            # <<<< 修改 1 >>>> 接收 evaluate 返回的三个值
            avg_val_loss, epoch_val_metrics, vis_batch = evaluate(model, val_dataloader, criterion, val_cls_metrics, val_img_metrics, device, config, q_mats)
            
            log_entry['val_loss'] = avg_val_loss
            log_entry.update(epoch_val_metrics)
            print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {epoch_val_metrics.get('val_f1_score', 0):.4f} | Val PSNR: {epoch_val_metrics.get('val_psnr', 0):.2f} dB")
            
            # <<<< 修改 2 >>>> 直接使用 evaluate 返回的结果进行可视化
            if vis_batch:
                print("    -> 生成可视化结果...")
                clean_batch, noisy_batch, restored_batch = vis_batch
                comparison_grid = torch.cat([clean_batch, noisy_batch, restored_batch], dim=0)
                vis_folder = os.path.join(vis_dir, "epoch_samples")
                os.makedirs(vis_folder, exist_ok=True)
                save_image(comparison_grid.float(), os.path.join(vis_folder, f"epoch_{epoch+1:04d}_comparison.png"), nrow=clean_batch.shape[0])
            
            # --- Saving logs, plots, checkpoints (no changes here) ---
            print("    -> 保存训练日志和图表...")
            history.append(log_entry)
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
            save_plot(history_df, "train_loss", "Training Loss", "Loss", os.path.join(vis_dir, "loss_plot_train.png"))
            if 'val_loss' in history_df.columns:
                 save_plot(history_df, "val_loss", "Validation Loss", "Loss", os.path.join(vis_dir, "loss_plot_val.png"))
            save_metrics_plot(history_df, vis_dir)
            if (epoch + 1) % config["training_params"]["save_checkpoint_freq"] == 0 or (epoch + 1) == config["training_params"]["epochs"]:
                 ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:04d}.pth")
                 torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'config': config}, ckpt_path)
                 print(f"    -> 检查点已保存至 {ckpt_path}")
                 
            # <<<< 安全措施：在重量级操作后清理缓存 >>>>
            torch.cuda.empty_cache() 
        else:
            print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}")
            history.append(log_entry)

        if scheduler: scheduler.step()

# --- 8. COMMAND-LINE INTERFACE (No changes) ---
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