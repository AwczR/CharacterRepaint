# train.py (FINAL, BUG-FIXED, OPTIMIZED & ROBUST CHECKPOINTING WITH ERROR LOGGING)
"""
Discrete Diffusion Training Script - Final Version
- [CRITICAL BUG FIX] Correctly computes the cumulative transition matrix (q_cumprod).
- [OPTIMIZATION] Implements importance sampling for timesteps for more efficient training.
- [ROBUSTNESS] All known scope and logic errors have been fixed.
- [ROBUSTNESS] Checkpoint saving is wrapped in a try-except block that logs errors to a file and continues training.
- Implements EMA model for training stability.
- Includes class weighting, gradient clipping, and enhanced logging.
"""
import os, json, yaml, argparse, math
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.swa_utils import AveragedModel

# [新增导入] 为错误日志记录导入 datetime 和 traceback 库
import datetime
import traceback

# 导入您的模型定义
from model import AttentionUNetDenoisingModel

# ... (从这里到 train() 函数的所有代码都保持不变) ...
# ---------- 1. 噪声调度 ----------
def make_conservative_schedule(T, min_beta=1e-4, max_beta=0.15):
    """生成一个保守的线性beta调度。"""
    return torch.linspace(min_beta**0.5, max_beta**0.5, T) ** 2

# ---------- 2. 时间采样权重 (用于重要性采样) ----------
def get_time_weights(T, exp=1.0):
    """
    为重要性采样生成时间步权重。
    exp > 1.0: 倾向于采样更大的t (更难的任务)。
    exp = 1.0: 均匀采样。
    exp < 1.0: 倾向于采样更小的t (更容易的任务)。
    """
    w = torch.pow(torch.arange(1, T + 1, dtype=torch.float32), exp)
    return w / w.sum()

# ---------- 3. 扩散核心 ----------
def q_sample_discrete(x_start, t, q_bar_t):
    """根据给定的时间步t和累积转移矩阵q_bar_t，从x_start采样xt。"""
    B, _, H, W = x_start.shape
    x_start_onehot = F.one_hot(x_start.squeeze(1), 2).float()
    x_start_onehot = x_start_onehot.view(B, H * W, 2)
    xt_probs = torch.bmm(x_start_onehot, q_bar_t)
    xt_probs = xt_probs.view(-1, 2).clamp(min=1e-5)
    xt = torch.multinomial(xt_probs, 1).view(B, 1, H, W)
    return xt

@torch.no_grad()
def p_sample_loop_restoration(model, device, cfg, q_mats, x_start, start_t):
    """从t=start_t-1开始，执行反向去噪过程来恢复图像。"""
    model.eval()
    B, _, H, W = x_start.shape
    
    q_bar = q_mats['q_mats_cumprod'][start_t - 1].expand(B, -1, -1)
    noisy = q_sample_discrete(x_start, torch.full((B,), start_t - 1, device=device), q_bar)
    img = noisy.clone()

    for t in tqdm(reversed(range(start_t)), desc="Denoising", total=start_t, leave=False):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        logits = model(img.float(), t_tensor)

        q_btm1 = q_mats['q_mats_cumprod'][t - 1] if t > 0 else torch.eye(2, device=device)
        qt = q_mats['q_one_step_mats'][t]
        q_btm1, qt = q_btm1.expand(B, -1, -1), qt.expand(B, -1, -1)

        x_t_onehot = F.one_hot(img.squeeze(1), 2).float().view(B, H * W, 2)
        x0_prob = F.softmax(logits, 1).permute(0, 2, 3, 1).reshape(B, H * W, 2)
        
        term1 = torch.bmm(x_t_onehot, qt.transpose(1, 2))
        term2 = torch.bmm(x0_prob, q_btm1)
        
        post_log = torch.log(term1 + 1e-8) + torch.log(term2 + 1e-8)
        post_prob = F.softmax(post_log.view(-1, 2), dim=-1)
        
        img = torch.multinomial(post_prob, 1).view(B, 1, H, W)

    model.train()
    return img, noisy


# ---------- 4. 数据集 ----------
class RubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288), bin_th=0.5):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.base = meta["base_clean_dir"]
        self.pairs = meta["image_pairs"]
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            lambda x: (x > bin_th).long()
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        path = os.path.join(self.base, self.pairs[idx]["clean_path_relative"])
        try:
            return self.tf(Image.open(path).convert("L"))
        except Exception as e:
            print(f"Skipping image {path} due to error: {e}")
            return None

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# ---------- 5. 可视化 ----------
def save_plot(df, y_col, title, ylabel, fname):
    plt.figure(figsize=(12, 6))
    if not df.empty and y_col in df.columns:
        plt.plot(df["epoch"], df[y_col], marker='.')
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    plt.tight_layout(); plt.savefig(fname); plt.close()

def save_metrics_plot(df, out_dir):
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Classification Metrics', color='tab:blue')
    for col, colr in [('val_accuracy', 'green'), ('val_f1', 'red'),
                      ('val_precision', 'purple'), ('val_recall', 'orange')]:
        if col in df.columns and df[col].notna().any():
            ax1.plot(df["epoch"], df[col], label=col.replace('val_', '').capitalize(), color=colr, marker='o')
    if any(col in df.columns and df[col].notna().any() for col in ['val_accuracy', 'val_f1', 'val_precision', 'val_recall']):
        ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx(); ax2.set_ylabel('Image Quality Metrics', color='tab:red')
    if 'val_psnr' in df.columns and df['val_psnr'].notna().any():
        ax2.plot(df["epoch"], df['val_psnr'], label='PSNR', color='tab:red', marker='s')
    if 'val_ssim' in df.columns and df['val_ssim'].notna().any():
        ax2.plot(df["epoch"], df['val_ssim'], label='SSIM', color='darkred', marker='^')
    if any(col in df.columns and df[col].notna().any() for col in ['val_psnr', 'val_ssim']):
        ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, "validation_metrics_plot.png")); plt.close()

# ---------- 6. 训练主循环 ----------
def train(cfg):
    out_dir = cfg["training_params"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    device = torch.device(cfg["training_params"]["device"] if torch.cuda.is_available() else "cpu")

    # ... (数据加载, 扩散参数, 模型, 优化器等部分的定义保持不变) ...
    # 数据集
    dataset = RubbingsDataset(cfg["dataset"]["metadata_path"], (cfg["model_params"]["image_size"],) * 2)
    val_size = max(4, int(0.1 * len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, cfg["dataset"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"], collate_fn=custom_collate, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, cfg["visualization_params"]["num_samples_to_visualize"], shuffle=False, collate_fn=custom_collate, pin_memory=True)

    # 扩散参数
    T = cfg["diffusion_params"]["num_timesteps"]
    betas = make_conservative_schedule(T, max_beta=cfg["diffusion_params"]["beta_max"]).to(device)
    print(f"[Diffusion] Timesteps: {T}, β min={betas.min():.4f}, max={betas.max():.4f}")

    q_one_step = torch.zeros(T, 2, 2, device=device)
    q_one_step[:, 0, 0] = 1 - betas
    q_one_step[:, 1, 1] = 1 - betas
    q_one_step[:, 0, 1] = betas
    q_one_step[:, 1, 0] = betas
    
    q_cumprod = torch.zeros_like(q_one_step)
    current_prod = torch.eye(2, device=device)
    for i in range(T):
        current_prod = torch.matmul(q_one_step[i], current_prod)
        q_cumprod[i] = current_prod
    
    q_mats = {"q_one_step_mats": q_one_step, "q_mats_cumprod": q_cumprod}

    # 模型
    model = AttentionUNetDenoisingModel(**cfg["model_params"], num_classes=2).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    ema_model = AveragedModel(model, avg_fn=lambda avg, p, num: 0.999 * avg + 0.001 * p)
    ema_model.update_parameters(model)

    opt = optim.AdamW(model.parameters(), lr=float(cfg["training_params"]["lr"]), weight_decay=0.01)
    
    epochs = cfg["training_params"]["epochs"]
    scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    print(f"[Training] Using scheduler: CosineAnnealingLR(T_max={epochs})")
    
    time_exp = cfg["training_params"].get("time_exp", 1.5)
    t_weights = get_time_weights(T, exp=time_exp).to(device)
    print(f"[Training] Using importance sampling for timesteps with exponent: {time_exp}")
    
    class_weights = torch.tensor([1.0, 9.55], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    val_met = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary"),
        'precision': torchmetrics.Precision(task="binary"),
        'recall': torchmetrics.Recall(task="binary"),
        'f1': torchmetrics.F1Score(task="binary"),
        'psnr': PeakSignalNoiseRatio(data_range=1.0),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)
    }).to(device)


    history = []
    vis_freq = cfg["training_params"]["save_visualization_freq"]
    ckpt_freq = cfg["training_params"]["save_checkpoint_freq"]
    full_sample_freq = cfg["training_params"].get("full_sample_freq", 25)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for clean in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            if clean is None: continue
            clean = clean.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            t = torch.multinomial(t_weights, clean.shape[0], replacement=True).to(device)
            noisy = q_sample_discrete(clean, t, q_cumprod[t])
            logits = model(noisy.float(), t)
            
            loss = criterion(logits, clean.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_model.update_parameters(model)
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader) if train_loader else 0.0
        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f}")

        # --- 验证与日志记录循环 ---
        run_validation = (epoch + 1) % vis_freq == 0 or (epoch + 1) == epochs
        
        if run_validation:
            log_entry = {"epoch": epoch + 1, "train_loss": avg_train_loss}
            ema_model.eval()
            
            avg_val_loss = 0.0
            with torch.no_grad():
                for clean_val in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Fast Val]", leave=False):
                    if clean_val is None: continue
                    clean_val = clean_val.to(device, non_blocking=True)
                    logits_t0 = ema_model.module(clean_val.float(), torch.zeros(clean_val.shape[0], device=device, dtype=torch.long))
                    avg_val_loss += criterion(logits_t0, clean_val.squeeze(1)).item()
            avg_val_loss /= len(val_loader)
            log_entry["val_loss"] = avg_val_loss
            print(f"Epoch {epoch + 1:03d} | Fast Val Loss (L0): {avg_val_loss:.4f}")

            run_full_sampling = (epoch + 1) % full_sample_freq == 0 or (epoch + 1) == epochs
            if run_full_sampling:
                # ... (完整采样验证逻辑不变) ...
                vis_batch = None
                with torch.no_grad():
                    for clean_val in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Full Sample Val]", leave=False):
                        if clean_val is None: continue
                        clean_val = clean_val.to(device, non_blocking=True)
                        restored, noisy = p_sample_loop_restoration(ema_model.module, device, cfg, q_mats, clean_val, cfg["visualization_params"]["start_timestep"])
                        if vis_batch is None:
                            vis_batch = (clean_val.cpu(), noisy.cpu(), restored.cpu())
                        val_met.update(restored.float(), clean_val.float())
                
                metrics = {f"val_{k}": v.item() for k, v in val_met.compute().items()}
                val_met.reset()
                log_entry.update(metrics)
                print(f"Epoch {epoch + 1:03d} | Full Val | F1: {metrics.get('val_f1', 0):.4f} | PSNR: {metrics.get('val_psnr', 0):.2f} | SSIM: {metrics.get('val_ssim', 0):.4f}")

                if vis_batch:
                    clean_vis, noisy_vis, restored_vis = vis_batch
                    comp = torch.cat([clean_vis, noisy_vis, restored_vis], dim=0)
                    save_image(comp.float(), os.path.join(vis_dir, f"epoch_{epoch + 1:04d}_comparison.png"), nrow=clean_vis.shape[0])


            history.append(log_entry)
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
            save_plot(df, "train_loss", "Training Loss", "Loss", os.path.join(vis_dir, "loss_plot_train.png"))
            save_metrics_plot(df, vis_dir)

        # --- 检查点保存循环 ---
        run_checkpoint = (epoch + 1) % ckpt_freq == 0 or (epoch + 1) == epochs
        if run_checkpoint:
            save_dict = {
                "epoch": epoch + 1,
                "model": ema_model.module.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler_obj.state_dict(),
                "train_loss": avg_train_loss,
            }
            if run_validation and history:
                save_dict.update(history[-1])

            # [核心修复] 使用带有错误日志记录的健壮保存逻辑
            try:
                torch.save(save_dict, os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:04d}.pth"))
                print(f"Saved checkpoint for epoch {epoch + 1}")
            except Exception as e:
                # 1. 获取当前时间戳
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # 2. 定义错误日志文件名和路径 (保存在主输出目录)
                error_filename = f"error-{timestamp}.txt"
                error_filepath = os.path.join(out_dir, error_filename)
                
                # 3. 准备详细的错误信息
                error_message = (
                    f"Failed to save checkpoint at epoch {epoch + 1}.\n"
                    f"Timestamp: {timestamp}\n\n"
                    f"Error Type: {type(e).__name__}\n"
                    f"Error Message: {e}\n\n"
                    f"Full Traceback:\n"
                    f"-----------------\n"
                    f"{traceback.format_exc()}"
                )
                
                # 4. 将错误信息写入文件
                try:
                    with open(error_filepath, 'w', encoding='utf-8') as f:
                        f.write(error_message)
                    print(f"\n!!!!!!!! FAILED TO SAVE CHECKPOINT FOR EPOCH {epoch + 1} !!!!!!!!")
                    print(f"An error log has been saved to: {error_filepath}")
                except Exception as log_e:
                    print(f"\n!!!!!!!! FAILED TO SAVE CHECKPOINT AND ALSO FAILED TO WRITE LOG FILE !!!!!!!!")
                    print(f"Original saving error: {e}")
                    print(f"Logging error: {log_e}")

                # 5. 继续训练
                print("Continuing training without saving...\n")

        scheduler_obj.step()

# ---------- 7. CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Discrete Diffusion Model for Image Binarization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        train(config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()