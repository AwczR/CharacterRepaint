# test_timesteps.py (FINAL & FIXED for Random Sample Evaluation & Visualization)
"""
Evaluate a trained discrete diffusion model on a random subset of images.
Compatible with train.py & model.py (uses config.yaml directly).
"""
import os
import argparse
import json
import yaml
from tqdm import tqdm
import random
import numpy as np # [新增导入] 用于随机选择索引

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset # [新增导入] Subset
from torchvision.utils import save_image
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
from torchvision import transforms
 
# 导入模型定义
from model import AttentionUNetDenoisingModel
 
# 复制 train.py 中的核心扩散函数
# ---------- 1. 噪声调度 (从 train.py 复制) ----------
def make_conservative_schedule(T, min_beta=1e-4, max_beta=0.15):
    return torch.linspace(min_beta**0.5, max_beta**0.5, T) ** 2

# ---------- 2. 时间采样权重 (从 train.py 复制，在此脚本中不直接使用) ----------
def get_time_weights(T, exp=1.0):
    w = torch.pow(torch.arange(1, T + 1, dtype=torch.float32), exp)
    return w / w.sum()

# ---------- 3. 扩散核心 - q_sample_discrete (从 train.py 复制) ----------
def q_sample_discrete(x_start, t, q_bar_t):
    B, _, H, W = x_start.shape
    x_start_onehot = F.one_hot(x_start.squeeze(1), 2).float()
    x_start_onehot = x_start_onehot.view(B, H * W, 2)
    xt_probs = torch.bmm(x_start_onehot, q_bar_t)
    xt_probs = xt_probs.view(-1, 2).clamp(min=1e-5)
    xt = torch.multinomial(xt_probs, 1).view(B, 1, H, W)
    return xt

# ---------- 3. 扩散核心 - p_sample_loop_restoration_eval (修改以适应测试) ----------
@torch.no_grad()
def p_sample_loop_restoration_eval(model, device, cfg, q_mats, noisy_img_t_start, start_t):
    model.eval()
    B, _, H, W = noisy_img_t_start.shape
    img = noisy_img_t_start.clone()

    for t in tqdm(reversed(range(start_t + 1)), desc="Denoising", total=start_t + 1, leave=False):
        if t == 0:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            logits = model(img.float(), t_tensor)
            img = torch.argmax(logits, dim=1, keepdim=True).view(B, 1, H, W)
            break
            
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
    return img


# ---------- Dataset ----------
class PairedRubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288), bin_th=0.5):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.base_clean_dir = meta["base_clean_dir"]
        self.base_noisy_dir = meta.get("base_noisy_dir") # 使用 .get() 避免keyError
        self.image_pairs = meta["image_pairs"]
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            lambda x: (x > bin_th).long()
        ])
 
    def __len__(self):
        return len(self.image_pairs)
 
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        clean_path = os.path.join(self.base_clean_dir, pair["clean_path_relative"])
        
        try:
            clean = self.tf(Image.open(clean_path).convert("L"))
            noisy = None # 默认设置为None
            if self.base_noisy_dir and "noisy_path_relative" in pair:
                 full_noisy_path = os.path.join(self.base_noisy_dir, pair["noisy_path_relative"])
                 if os.path.exists(full_noisy_path): # [增强] 检查文件是否存在
                    noisy = self.tf(Image.open(full_noisy_path).convert("L"))
                 else:
                    print(f"[Dataset Warning] Noisy image not found: {full_noisy_path}. Using zero tensor as placeholder.")
                    noisy = torch.zeros_like(clean)
            else:
                 noisy = torch.zeros_like(clean) # 如果没有noisy_dir或noisy_path_relative，也用占位符
            
            return noisy, clean
        except Exception as e:
            print(f"[Dataset Error] Skip pair {clean_path}: {e}")
            return None, None

def custom_collate(batch):
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if not batch:
        return None, None
    noisy, clean = zip(*batch)
    return torch.stack(noisy), torch.stack(clean)
 
# ---------- Main ----------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Real-World Denoising Evaluation ---")
    print(f"Using device: {device}")
 
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
 
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model = AttentionUNetDenoisingModel(**cfg["model_params"], num_classes=2).to(device)
    
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model state dict from 'model' key in checkpoint.")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded model state dict directly from checkpoint (no 'model' key).")

    model.eval()
 
    T = cfg["diffusion_params"]["num_timesteps"]
    betas = make_conservative_schedule(
        T,
        max_beta=cfg["diffusion_params"].get("beta_max", 0.25)
    ).to(device)
 
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
    print(f"Diffusion matrices constructed with T={T}, beta_max={betas.max():.4f}")

    # 4. 数据集：加载完整数据集，但只评估一个子集
    full_dataset = PairedRubbingsDataset(
        args.metadata_path,
        (cfg["model_params"]["image_size"],) * 2
    )
    
    # [核心修改] 随机选择 num_eval_samples 数量的样本进行评估和可视化
    num_eval_samples = args.num_eval_samples # 从命令行获取
    if len(full_dataset) < num_eval_samples:
        print(f"[Warning] Dataset size ({len(full_dataset)}) is smaller than num_eval_samples ({num_eval_samples}). Using all available samples.")
        eval_indices = list(range(len(full_dataset)))
    else:
        # 使用 numpy 的 random.choice 更高效地选择唯一索引
        eval_indices = np.random.choice(len(full_dataset), num_eval_samples, replace=False).tolist()
    
    # 创建一个 Subset，只包含随机选择的样本
    eval_subset = Subset(full_dataset, eval_indices)
    
    # DataLoader 现在只加载这个 Subset 的数据
    dataloader = DataLoader(
        eval_subset, # [修改] 传入 Subset
        batch_size=args.batch_size,
        shuffle=False, # 子集已经随机选择，这里无需shuffle
        collate_fn=custom_collate,
        num_workers=args.num_workers
    )
    print(f"Selected {len(eval_subset)} random samples for evaluation.")
 
    # 5. 指标
    cls_metrics = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary"),
        'f1': torchmetrics.F1Score(task="binary"),
        'precision': torchmetrics.Precision(task="binary"),
        'recall': torchmetrics.Recall(task="binary")
    }).to(device)
    img_metrics = torchmetrics.MetricCollection({
        'psnr': PeakSignalNoiseRatio(data_range=1.0),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)
    }).to(device)
 
    # 6. 评估循环
    summary = []
    
    # 提前获取所有要评估和可视化的样本，避免在每个timestep循环中重复加载
    # 这里的 sample_noisy_imgs 和 sample_clean_imgs 实际上就是 eval_subset 的所有数据
    all_eval_noisy_imgs_list = []
    all_eval_clean_imgs_list = []
    for i in range(len(eval_subset)):
        noisy_sample, clean_sample = eval_subset[i]
        if noisy_sample is not None and clean_sample is not None:
            all_eval_noisy_imgs_list.append(noisy_sample)
            all_eval_clean_imgs_list.append(clean_sample)
            
    if all_eval_noisy_imgs_list:
        sample_noisy_imgs_tensor = torch.stack(all_eval_noisy_imgs_list).to(device)
        sample_clean_imgs_tensor = torch.stack(all_eval_clean_imgs_list).to(device)
    else:
        sample_noisy_imgs_tensor, sample_clean_imgs_tensor = None, None
        print("[Warning] No valid samples found for evaluation or visualization in the selected subset.")


    for start_t in args.timesteps:
        print(f"\n--- Evaluating Denoising from Timestep (t_start): {start_t} ---")
        out_dir_t = os.path.join(args.output_dir, f"T_{start_t}")
        os.makedirs(out_dir_t, exist_ok=True)
        cls_metrics.reset()
        img_metrics.reset()
 
        # 现在 dataloader 已经只包含您选择的随机样本了，直接遍历它即可
        for i, (noisy_inputs, clean_targets) in enumerate(tqdm(dataloader, desc=f"Processing T={start_t}")):
            if noisy_inputs is None or clean_targets is None:
                continue
            noisy_inputs = noisy_inputs.to(device)
            clean_targets = clean_targets.to(device)

            restored = p_sample_loop_restoration_eval(
                model, device, cfg, q_mats, noisy_inputs, start_t
            )
            
            cls_metrics.update(restored.squeeze(1), clean_targets.squeeze(1))
            img_metrics.update(restored.float(), clean_targets.float())
 
        # [可视化] 直接使用 sample_noisy_imgs_tensor 和 sample_clean_imgs_tensor
        if sample_noisy_imgs_tensor is not None and sample_clean_imgs_tensor is not None:
            with torch.no_grad():
                sample_restored = p_sample_loop_restoration_eval(
                    model, device, cfg, q_mats, sample_noisy_imgs_tensor, start_t
                )
            
            grid = torch.cat([
                sample_noisy_imgs_tensor.cpu(),  # 原始带噪声的输入
                sample_restored.cpu(),    # 模型的去噪输出
                sample_clean_imgs_tensor.cpu()   # 真实干净的目标
            ], dim=0)
            save_image(
                grid.float(),
                os.path.join(out_dir_t, f"comparison_T_{start_t}.png"),
                nrow=len(eval_subset) if len(eval_subset) > 0 else 1 # 每行显示所有可视化样本
            )
            print(f"  Saved comparison image for T={start_t}")
 
        res = {k: v.item() for k, v in cls_metrics.compute().items()}
        res.update({k: v.item() for k, v in img_metrics.compute().items()})
        res["timestep"] = start_t
        summary.append(res)
        print(f"  Results for T={start_t}: PSNR:{res['psnr']:.3f} SSIM:{res['ssim']:.3f} F1:{res['f1']:.3f} Accuracy:{res['accuracy']:.3f}")
 
    # 7. 结果汇总
    if summary:
        df = pd.DataFrame(summary).sort_values("psnr", ascending=False)
        print("\n--- Summary Report ---")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(args.output_dir, "summary_report.csv"), index=False)
 
# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate discrete diffusion on real noisy images")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml used during training")
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True,
                        help="Path to the .pth checkpoint")
    parser.add_argument("-m", "--metadata_path", type=str, required=True,
                        help="Path to evaluation metadata JSON")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("-t", "--timesteps", type=int, nargs='+', required=True,
                        help="List of start timesteps for denoising (e.g. 50 100 150)")
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-nw", "--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    # [新增参数] 评估样本数量
    parser.add_argument("--num_eval_samples", type=int, default=4,
                        help="Number of random samples to evaluate and visualize.")
    args = parser.parse_args()
    main(args)