# train.py (FINAL, RIGOROUSLY FIXED & OPTIMIZED with Full Self-Check)
"""
Discrete Diffusion Training Script - Final Version
- [CRITICAL BUG FIX] Rigorously fixed Bayesian inversion formula (dimension and indexing) in p_sample_loop_restoration.
- [CRITICAL BUG FIX] Corrected cumulative transition matrix (q_cumprod) calculation.
- [FIX] Ensured x_t_flat_idx and pred_x0_classes are correctly defined inside the denoising loop.
- [OPTIMIZATION] Implements importance sampling for timesteps for more efficient training.
- [ROBUSTNESS] All known scope and logic errors have been fixed.
- [ROBUSTNESS] Checkpoint saving is wrapped in a try-except block that logs errors to a file and continues training.
- [ROBUSTNESS] Added a comprehensive self-check function at startup to catch common errors early.
- Implements EMA model for training stability.
- Includes class weighting, gradient clipping, and enhanced logging.
"""
import os, json, yaml, argparse, math
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split # Added random_split
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.swa_utils import AveragedModel

import datetime
import traceback
import sys # For sys.exit()

from model import AttentionUNetDenoisingModel # Ensure this path is correct

# ---------- OpenBayesTool 安全导入 ----------
try:
    from openbayestool import log_param, log_metric, clear_metric
    print("OpenBayesTool modules loaded successfully.")
    _openbayestool_available = True
except ImportError:
    print("OpenBayesTool not found. Using dummy logging functions.")
    # 定义模拟函数，当 openbayestool 不可用时使用
    def log_param(key, value):
        pass # 或者可以 print(f"Dummy log_param: {key}={value}")
    def log_metric(key, value):
        pass # 或者可以 print(f"Dummy log_metric: {key}={value}")
    def clear_metric(key):
        pass # 或者可以 print(f"Dummy clear_metric: {key}")
    _openbayestool_available = False
# ---------------------------------------------

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

# ---------- 3. 扩散核心 - q_sample_discrete ----------
def q_sample_discrete(x_start, t, q_bar_t):
    """根据给定的时间步t和累积转移矩阵q_bar_t，从x_start采样xt。"""
    B, _, H, W = x_start.shape
    x_start_onehot = F.one_hot(x_start.squeeze(1), 2).float()
    x_start_onehot = x_start_onehot.view(B, H * W, 2)
    xt_probs = torch.bmm(x_start_onehot, q_bar_t)
    xt_probs = xt_probs.view(-1, 2).clamp(min=1e-5)
    xt = torch.multinomial(xt_probs, 1).view(B, 1, H, W)
    return xt

# ---------- 3. 扩散核心 - p_sample_loop_restoration (Rigorously Fixed Bayesian Inversion) ----------
@torch.no_grad()
def p_sample_loop_restoration(model, device, cfg, q_mats, x_start, start_t):
    """从t=start_t-1开始，执行反向去噪过程来恢复图像。"""
    model.eval()
    B, _, H, W = x_start.shape
    
    # 1. 生成起始的噪声图像
    q_bar = q_mats['q_mats_cumprod'][start_t - 1].expand(B, -1, -1)
    noisy = q_sample_discrete(x_start, torch.full((B,), start_t - 1, device=device), q_bar)
    img = noisy.clone() # img is x_t for the current step

    for t in tqdm(reversed(range(1, start_t)), desc="Denoising", total=start_t, leave=False):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        logits = model(img.float(), t_tensor) # Model predicts x_0 logits from x_t (img)

        # Get relevant transition matrices for this timestep t
        # q(x_{t-1}|x_0) -> q_btm1 (cumulative)
        q_btm1 = q_mats['q_mats_cumprod'][t - 1] if t > 0 else torch.eye(2, device=device) 
        # q(x_t|x_{t-1}) -> qt (one-step)
        qt = q_mats['q_one_step_mats'][t]
        
        # Expand matrices for batch operations (B, 2, 2)
        qt_b = qt.expand(B, -1, -1)
        q_btm1_b = q_btm1.expand(B, -1, -1)

        # Reshape current noisy image (x_t) and predicted x_0 probabilities for pixel-wise ops
        x_t_flat_idx = img.squeeze(1).view(B, H * W) # (B, H*W) where each element is 0 or 1 (the current pixel value of x_t)
        x0_prob_pred = F.softmax(logits, 1).permute(0, 2, 3, 1).reshape(B, H * W, 2) # (B, H*W, 2)
        pred_x0_classes = x0_prob_pred.argmax(dim=-1) # (B, H*W) - predicted hard class for x_0 for each pixel

        # Calculate log_posterior for each possible state of x_{t-1} (0 or 1)
        # log p(x_{t-1}=j | x_t, x_0_pred) propto log P(x_t | x_{t-1}=j) + log P(x_0_pred | x_{t-1}=j)
        log_posterior = torch.zeros(B, H * W, 2, device=device) 

        # For x_{t-1} = 0
        log_posterior[:, :, 0] = (
            torch.log(torch.gather(qt_b[:, 0, :], 1, x_t_flat_idx) + 1e-8) +
            torch.log(q_btm1_b[torch.arange(B, device=device).unsqueeze(1).expand(-1, H*W), pred_x0_classes, 0] + 1e-8)
        )

        # For x_{t-1} = 1
        log_posterior[:, :, 1] = (
            torch.log(torch.gather(qt_b[:, 1, :], 1, x_t_flat_idx) + 1e-8) +
            torch.log(q_btm1_b[torch.arange(B, device=device).unsqueeze(1).expand(-1, H*W), pred_x0_classes, 1] + 1e-8)
        )
        
        post_prob = F.softmax(log_posterior.view(-1, 2), dim=-1)
        img = torch.multinomial(post_prob, 1).view(B, 1, H, W)
    
    # 这个时候的 img 是从 x_1 采样得到的 x_0
    t_tensor = torch.full((B,), 1, device=device, dtype=torch.long)
    logits = model(img.float(), t_tensor) # 让模型对 x_0 的最终样貌做一次预测
    img = logits.argmax(dim=1, keepdim=True) # 直接取最可能的结果作为最终图像，不再随机采样
    
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
            print(f"Skipping image {path}: {e}")
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
    # --- [新增] 自检功能 ---
    self_check_pipeline(cfg)
    # --- 自检结束 ---

    out_dir = cfg["training_params"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    device = torch.device(cfg["training_params"]["device"] if torch.cuda.is_available() else "cpu")

    # 数据集
    dataset = RubbingsDataset(cfg["dataset"]["metadata_path"], (cfg["model_params"]["image_size"],) * 2)
    val_size = max(4, int(0.1 * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(42))
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
    
    # [CRITICAL BUG FIX] Correctly compute q_cumprod via cumulative matrix multiplication
    q_cumprod = torch.zeros_like(q_one_step)
    current_prod = torch.eye(2, device=device)
    for i in range(T):
        current_prod = torch.matmul(q_one_step[i], current_prod)
        q_cumprod[i] = current_prod
    
    q_mats = {"q_one_step_mats": q_one_step, "q_mats_cumprod": q_cumprod}

    # 模型
    model = AttentionUNetDenoisingModel(**cfg["model_params"], num_classes=2).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # EMA 模型
    ema_model = AveragedModel(model, avg_fn=lambda avg, p, num: 0.999 * avg + 0.001 * p)
    ema_model.update_parameters(model)

    # 优化器
    opt = optim.AdamW(model.parameters(), lr=float(cfg["training_params"]["lr"]), weight_decay=0.01)
    
    # [FIX] Define epochs before using it
    epochs = cfg["training_params"]["epochs"]
    
    # 学习率调度器
    scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    print(f"[Training] Using scheduler: CosineAnnealingLR(T_max={epochs})")
    
    # 时间采样权重 (重要性采样)
    time_exp = cfg["training_params"].get("time_exp", 1.5)
    t_weights = get_time_weights(T, exp=time_exp).to(device)
    print(f"[Training] Using importance sampling for timesteps with exponent: {time_exp}")
    
    # 类别权重 #
    class_weights = torch.tensor([1.0, 2.0], device=device) # NOTE: Replace with your dataset's calculated weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 评估指标
    val_met = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary"),
        'precision': torchmetrics.Precision(task="binary"),
        'recall': torchmetrics.Recall(task="binary"),
        'f1': torchmetrics.F1Score(task="binary"),
        'psnr': PeakSignalNoiseRatio(data_range=1.0),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)
    }).to(device)

    history = []
    
    # ---------- 读取新的频率参数 ----------
    fast_val_freq = cfg["training_params"].get("fast_val_freq", 1)
    val_loss_timestep = cfg["training_params"].get("val_loss_timestep", 0)
    full_sample_freq = cfg["training_params"].get("full_sample_freq", 25)
    ckpt_freq = cfg["training_params"]["save_checkpoint_freq"]
    # -------------------------------------

    # [新增] 在训练开始前清除旧的指标，确保每次运行都是新的图表
    if _openbayestool_available:
        clear_metric('train_loss')
        clear_metric('val_loss')
        clear_metric('PSNR')
        # 如果你还想可视化 F1 或 SSIM，也在这里清除
        # clear_metric('F1')
        # clear_metric('SSIM')

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
        # [新增] 记录 train_loss 到 OpenBayesTool
        log_metric('train_loss', avg_train_loss)


        # 为当前 epoch 初始化日志条目
        current_log_entry = {"epoch": epoch + 1, "train_loss": avg_train_loss}

        # --- 快速验证损失 (val_L_t) ---
        run_fast_val_loss = (epoch + 1) % fast_val_freq == 0 or (epoch + 1) == epochs
        if run_fast_val_loss:
            ema_model.eval()
            avg_val_loss = 0.0
            with torch.no_grad():
                for clean_val in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Fast Val]", leave=False):
                    if clean_val is None: continue
                    clean_val = clean_val.to(device, non_blocking=True)
                    
                    # 根据 val_loss_timestep 对 clean_val 添加噪声
                    if val_loss_timestep == 0:
                        noisy_input_for_val_loss = clean_val
                        t_for_model = torch.zeros(clean_val.shape[0], device=device, dtype=torch.long)
                    else:
                        t_for_model = torch.full((clean_val.shape[0],), val_loss_timestep, device=device, dtype=torch.long)
                        noisy_input_for_val_loss = q_sample_discrete(clean_val, t_for_model, q_cumprod[val_loss_timestep].expand(clean_val.shape[0], -1, -1))
                    
                    logits_val = ema_model.module(noisy_input_for_val_loss.float(), t_for_model)
                    avg_val_loss += criterion(logits_val, clean_val.squeeze(1)).item()
            avg_val_loss /= len(val_loader)
            current_log_entry["val_loss"] = avg_val_loss # 将其记录为 val_loss
            print(f"Epoch {epoch + 1:03d} | Fast Val Loss (L_t={val_loss_timestep}): {avg_val_loss:.4f}")
            # [新增] 记录 val_loss 到 OpenBayesTool
            log_metric('val_loss', avg_val_loss)


        # --- 完整采样、详细指标计算和图像保存 ---
        # 这个块执行昂贵的验证过程，包括完整的去噪和图像指标计算。
        run_full_sampling_and_viz = (epoch + 1) % full_sample_freq == 0 or (epoch + 1) == epochs
        if run_full_sampling_and_viz:
            ema_model.eval() # 确保模型处于评估模式
            vis_batch = None
            with torch.no_grad():
                for clean_val in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Full Sample Val]", leave=False):
                    if clean_val is None: continue
                    clean_val = clean_val.to(device, non_blocking=True)
                    # p_sample_loop_restoration 使用 visualization_params.start_timestep
                    restored, noisy = p_sample_loop_restoration(ema_model.module, device, cfg, q_mats, clean_val, cfg["visualization_params"]["start_timestep"])
                    
                    if vis_batch is None:
                        # 仅在第一次循环时保存批次，用于后续图像保存
                        vis_batch = (clean_val.cpu(), noisy.cpu(), restored.cpu())
                    
                    val_met.update(restored.float(), clean_val.float())
            
            # 计算并更新详细指标
            metrics = {f"val_{k}": v.item() for k, v in val_met.compute().items()}
            val_met.reset() # 重置指标以便下次计算
            current_log_entry.update(metrics) # 将这些指标添加到当前日志条目中
            print(f"Epoch {epoch + 1:03d} | Full Val | F1: {metrics.get('val_f1', 0):.4f} | PSNR: {metrics.get('val_psnr', 0):.2f} | SSIM: {metrics.get('val_ssim', 0):.4f}")
            
            # [新增] 记录 PSNR 到 OpenBayesTool
            log_metric('PSNR', metrics.get('val_psnr', 0))
            # 你也可以在这里记录其他你希望可视化的指标，例如 F1 和 SSIM
            # log_metric('F1', metrics.get('val_f1', 0))
            # log_metric('SSIM', metrics.get('val_ssim', 0))


            # 图像保存 (现在与 full_sample_freq 关联)
            if vis_batch:
                clean_vis, noisy_vis, restored_vis = vis_batch
                comp = torch.cat([clean_vis, noisy_vis, restored_vis], dim=0)
                save_image(comp.float(), os.path.join(vis_dir, f"epoch_{epoch + 1:04d}_comparison.png"), nrow=clean_vis.shape[0])

        # --- 历史记录、CSV 保存和绘图 ---
        # 始终将当前 epoch 的日志条目添加到历史记录中
        history.append(current_log_entry)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
        
        # 绘图（现在与 full_sample_freq 关联，因为绘图需要完整的指标）
        if run_full_sampling_and_viz: # 只有当运行了完整的采样和指标计算时才绘制
            save_plot(df, "train_loss", "Training Loss", "Loss", os.path.join(vis_dir, "loss_plot_train.png"))
            save_metrics_plot(df, vis_dir) # 这个图使用 PSNR, SSIM, F1 等指标

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
            # 更新保存字典，包含当前 epoch 的所有日志信息
            save_dict.update(current_log_entry)

            try:
                torch.save(save_dict, os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:04d}.pth"))
                print(f"Saved checkpoint for epoch {epoch + 1}")
            except Exception as e:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                error_filename = f"error-{timestamp}.txt"
                error_filepath = os.path.join(out_dir, error_filename)
                
                error_message = (
                    f"Failed to save checkpoint at epoch {epoch + 1}.\n"
                    f"Timestamp: {timestamp}\n\n"
                    f"Error Type: {type(e).__name__}\n"
                    f"Error Message: {e}\n\n"
                    f"Full Traceback:\n"
                    f"-----------------\n"
                    f"{traceback.format_exc()}"
                )
                
                try:
                    with open(error_filepath, 'w', encoding='utf-8') as f:
                        f.write(error_message)
                    print(f"\n!!!!!!!! FAILED TO SAVE CHECKPOINT FOR EPOCH {epoch + 1} !!!!!!!!")
                    print(f"An error log has been saved to: {error_filepath}")
                except Exception as log_e:
                    print(f"\n!!!!!!!! FAILED TO SAVE CHECKPOINT AND ALSO FAILED TO WRITE LOG FILE !!!!!!!!")
                    print(f"Original saving error: {e}")
                    print(f"Logging error: {log_e}")

                print("Continuing training without saving...\n")

        scheduler_obj.step()

# ---------- Pipeline Self-Check Function ----------
def self_check_pipeline(cfg):
    print("\n--- Running Training Pipeline Self-Check ---")
    
    # Check device availability
    device = torch.device(cfg["training_params"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"  [Check] Device: {device}")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print(f"[ERROR] CUDA requested but not available. Please check CUDA installation or set device to 'cpu'.")
            sys.exit(1)
        print(f"  [Check] CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [Check] CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Check output directories can be created/accessed
    try:
        os.makedirs(cfg["training_params"]["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(cfg["training_params"]["output_dir"], "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg["training_params"]["output_dir"], "visualizations"), exist_ok=True)
        print(f"  [Check] Output directories are accessible and/or created: {cfg['training_params']['output_dir']}")
    except Exception as e:
        print(f"[ERROR] Could not create/access output directories: {e}")
        sys.exit(1)

    # Check metadata file existence and basic content
    try:
        metadata_path = cfg["dataset"]["metadata_path"]
        if not os.path.exists(metadata_path):
            print(f"[ERROR] Dataset metadata file not found: {metadata_path}")
            sys.exit(1)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if "base_clean_dir" not in meta or "image_pairs" not in meta:
            print(f"[ERROR] Metadata file {metadata_path} is missing 'base_clean_dir' or 'image_pairs' keys.")
            sys.exit(1)
        if not os.path.exists(meta["base_clean_dir"]):
            print(f"[ERROR] Base clean image directory not found: {meta['base_clean_dir']}")
            sys.exit(1)
        if len(meta["image_pairs"]) == 0:
            print(f"[ERROR] 'image_pairs' list in metadata is empty. No data to train on.")
            sys.exit(1)
        
        # Try loading one image to check data loading pipeline
        temp_dataset = RubbingsDataset(metadata_path, (cfg["model_params"]["image_size"],) * 2)
        sample_image = None
        for i in range(min(5, len(temp_dataset))): # Try first 5 valid samples
            sample = temp_dataset[i]
            if sample is not None:
                sample_image = sample
                break
        if sample_image is None:
            print(f"[ERROR] Failed to load any valid image from dataset {metadata_path}. Check image paths/corruptions.")
            sys.exit(1)
        if sample_image.shape != (1, cfg["model_params"]["image_size"], cfg["model_params"]["image_size"]):
             print(f"[ERROR] Loaded image shape {sample_image.shape} does not match expected (1, {cfg['model_params']['image_size']}, {cfg['model_params']['image_size']}).")
             sys.exit(1)
        if not (sample_image.min() >= 0 and sample_image.max() <= 1): # Check if normalized
            print(f"[WARNING] Loaded image pixel values not within [0, 1] range. Min: {sample_image.min().item():.2f}, Max: {sample_image.max().item():.2f}. (Might be okay if values are 0 or 1 specifically).")
        if not (sample_image.dtype == torch.long or sample_image.dtype == torch.int): # Check if binarized to long
            print(f"[WARNING] Loaded image dtype is {sample_image.dtype}, expected torch.long or torch.int after binarization. (Might be float if not long converted).")
        if not (sample_image.unique().numel() <= 2):
            print(f"[WARNING] Loaded image has more than 2 unique pixel values ({sample_image.unique()}). Expected binary (0 or 1).")


        print(f"  [Check] Dataset metadata loaded and a sample image ({sample_image.shape}) loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Dataset/metadata check failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Check model parameters (basic sanity)
    if cfg["model_params"]["image_size"] <= 0 or not isinstance(cfg["model_params"]["image_size"], int):
        print(f"[ERROR] Invalid image_size: {cfg['model_params']['image_size']}")
        sys.exit(1)
    if cfg["model_params"]["in_channels"] != 1 or cfg["model_params"]["out_channels"] != 1:
        print(f"[WARNING] Model in_channels/out_channels are not 1. Ensure this matches your grayscale/binary images.")
    if cfg["model_params"]["num_res_blocks"] < 1:
        print(f"[ERROR] num_res_blocks should be at least 1.")
        sys.exit(1)
    print(f"  [Check] Model parameters basic sanity check passed.")

    # Check diffusion parameters (basic sanity)
    if cfg["diffusion_params"]["num_timesteps"] < 10 or not isinstance(cfg["diffusion_params"]["num_timesteps"], int):
        print(f"[ERROR] num_timesteps should be at least 10.")
        sys.exit(1)
    if not (0 < cfg["diffusion_params"]["beta_max"] <= 1.0):
        print(f"[WARNING] beta_max ({cfg['diffusion_params']['beta_max']}) is outside typical (0, 1] range. For binary, might be even smaller.")
    print(f"  [Check] Diffusion parameters basic sanity check passed.")

    # Check visualization_params
    if cfg["visualization_params"]["start_timestep"] >= cfg["diffusion_params"]["num_timesteps"]:
        print(f"[WARNING] visualization_params.start_timestep ({cfg['visualization_params']['start_timestep']}) is >= num_timesteps. It should be < num_timesteps.")
    print(f"  [Check] Visualization parameters basic sanity check passed.")

    print("--- Self-Check Completed. All critical checks passed. ---")
    print("If you see warnings, consider reviewing them, but training might proceed.")


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