# test_timesteps.py (REAL-WORLD DENOISING VERSION)
"""
Script to test and evaluate the performance of a trained D3PM model
on real-world noisy images.

This script loads pairs of (noisy_image, clean_ground_truth), applies a
small amount of controlled diffusion noise to the noisy_image, and then
runs the full denoising process. The final restored image is compared
against the clean_ground_truth to calculate performance metrics.
"""
import os
import argparse
import json
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
from torchvision import transforms

# --- Import necessary components from your project files ---
from model import AttentionUNetDenoisingModel

# <<<< 新增/修改 >>>> 这是一个专为本测试脚本设计的Dataset，用于加载图像对
class PairedRubbingsDataset(Dataset):
    """Loads pairs of (noisy_image, clean_image) for evaluation."""
    def __init__(self, metadata_path, image_size=(288, 288), binarization_threshold=0.5):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")

        self.base_clean_dir = self.metadata["base_clean_dir"]
        self.base_noisy_dir = self.metadata["base_noisy_dir"] # 新增
        self.image_pairs = self.metadata["image_pairs"]

        if not os.path.isdir(self.base_clean_dir):
            print(f"Warning: Clean directory '{self.base_clean_dir}' not found.")
        if not os.path.isdir(self.base_noisy_dir):
            print(f"Warning: Noisy directory '{self.base_noisy_dir}' not found.")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            lambda x: (x > binarization_threshold).long()
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        
        # 路径拼接
        clean_path = os.path.join(self.base_clean_dir, pair_info["clean_path_relative"])
        noisy_path = os.path.join(self.base_noisy_dir, pair_info["noisy_path_relative"])

        try:
            clean_image = self.transform(Image.open(clean_path).convert("L"))
            noisy_image = self.transform(Image.open(noisy_path).convert("L"))
            return noisy_image, clean_image
        except Exception as e:
            print(f"\n[Dataset Error] Skipping pair for clean: {clean_path}. Reason: {e}")
            return None

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None
    noisy_images, clean_images = zip(*batch)
    return torch.stack(noisy_images), torch.stack(clean_images)

# --- Helper functions (copied from train.py) ---
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

# <<<< 修改 >>>> 函数重命名并修改逻辑，以反映其真实作用
@torch.no_grad()
def denoise_from_real_noise(model, device, q_mats, real_noisy_image, start_timestep):
    """
    Takes a real-world noisy image, applies controlled noise up to start_timestep,
    and then performs the full reverse diffusion process.
    """
    model.eval()
    batch_size = real_noisy_image.shape[0]
    num_classes = model.final_out_channels

    # <<<< 核心逻辑修改 >>>>
    # 1. 对已经有噪声的图像 `real_noisy_image`，再施加少量扩散噪声，得到 x_t
    if start_timestep > 0:
        t_tensor = torch.full((batch_size,), start_timestep - 1, device=device, dtype=torch.long)
        q_bar_t_start_2d = q_mats['q_mats_cumprod'][start_timestep - 1]
        q_bar_t_start = q_bar_t_start_2d.expand(batch_size, -1, -1)
        # The input to q_sample_discrete is our real noisy image
        img = q_sample_discrete(real_noisy_image, t_tensor, q_bar_t_start)
    else: # 如果 t=0, 不加任何额外噪声，直接开始“去噪”（实际上是模型的一次前向传播）
        img = real_noisy_image.clone()
    
    # 保存这个作为起点的、被二次加噪的图像，用于可视化
    initial_starting_point_for_denoising = img.clone()

    # 2. Denoising loop (from t down to 0)
    loop_range = reversed(range(start_timestep))
    for t in tqdm(loop_range, desc=f"Denoising from T={start_timestep}", total=start_timestep, leave=False):
        time = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
        pred_x0_logits = model(img.float(), time)
        
        if t == 0:
            img = torch.argmax(pred_x0_logits, dim=1).unsqueeze(1)
            break

        b, _, h, w = img.shape
        q_bar_t_minus_1_2d = q_mats['q_mats_cumprod'][t - 1]
        q_bar_t_minus_1 = q_bar_t_minus_1_2d.expand(b, -1, -1)
        q_t_2d = q_mats['q_one_step_mats'][t]
        q_t = q_t_2d.expand(b, -1, -1)
        
        x_t_one_hot_flat = F.one_hot(img.squeeze(1), num_classes=num_classes).float().view(b, h * w, -1)
        pred_x0_probs_flat = F.softmax(pred_x0_logits, dim=1).permute(0, 2, 3, 1).reshape(b, h * w, -1)
        
        term1 = torch.bmm(x_t_one_hot_flat, q_t)
        term2 = torch.bmm(pred_x0_probs_flat, q_bar_t_minus_1)
        
        posterior_probs_log = torch.log(term1.clamp(min=1e-20)) + torch.log(term2.clamp(min=1e-20))
        posterior_probs = F.softmax(posterior_probs_log.view(b * h * w, -1), dim=-1)
        
        sampled_flat = torch.multinomial(posterior_probs, num_samples=1)
        img = sampled_flat.view(b, 1, h, w)
        
    return img, initial_starting_point_for_denoising

def main(args):
    """Main function to run the timestep evaluation."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Real-World Denoising Evaluation ---")
    print(f"Using device: {device}")

    # Load Model from Checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = AttentionUNetDenoisingModel(**config['model_params'], num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {args.checkpoint_path}")

    # Build Diffusion Schedule
    num_timesteps = config["diffusion_params"]["num_timesteps"]
    betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    NUM_CLASSES = 2
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]; q_one_step_mats[i, 0, 0] = 1.0 - beta_t; q_one_step_mats[i, 1, 1] = 1.0 - beta_t; q_one_step_mats[i, 0, 1] = beta_t; q_one_step_mats[i, 1, 0] = beta_t
    q_mats_cumprod = torch.zeros_like(q_one_step_mats); current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps): current_mat = torch.matmul(q_one_step_mats[i], current_mat); q_mats_cumprod[i] = current_mat
    q_mats = {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}

    # <<<< 修改 >>>> 使用新的 PairedRubbingsDataset
    dataset = PairedRubbingsDataset(
        metadata_path=args.metadata_path,
        image_size=(config["model_params"]["image_size"], config["model_params"]["image_size"])
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=4
    )
    print(f"Paired dataset loaded with {len(dataset)} image pairs.")

    # Initialize Metrics
    cls_metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="binary"), 'f1_score': torchmetrics.F1Score(task="binary")}).to(device)
    img_metrics = torchmetrics.MetricCollection({'psnr': PeakSignalNoiseRatio(data_range=1.0), 'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)}).to(device)

    # Run Evaluation Loop
    results_summary = []
    for start_t in args.timesteps:
        print(f"\n--- Evaluating Start Timestep (Re-noise level): {start_t} ---")
        vis_dir = os.path.join(args.output_dir, f"T_{start_t}"); os.makedirs(vis_dir, exist_ok=True)
        cls_metrics.reset(); img_metrics.reset()
        
        # <<<< 修改 >>>> 循环现在获取 (noisy_images, clean_images) 对
        for i, (noisy_images, clean_images) in enumerate(tqdm(dataloader, desc="Processing batches")):
            if noisy_images is None or clean_images is None: continue
            
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            # <<<< 修改 >>>> 将 `noisy_images` 传入去噪函数
            restored_images, start_denoise_images = denoise_from_real_noise(
                model, device, q_mats, noisy_images, start_t
            )

            # Update metrics by comparing RESTORED vs CLEAN
            cls_metrics.update(restored_images, clean_images)
            img_metrics.update(restored_images.float(), clean_images.float())

            # Save visualization for the first batch
            if i == 0:
                # 可视化内容：原始噪声图 -> 二次加噪图 -> 修复图 -> 干净真值图
                comparison_grid = torch.cat([noisy_images.cpu(), start_denoise_images.cpu(), restored_images.cpu(), clean_images.cpu()], dim=0)
                save_image(
                    comparison_grid.float(),
                    os.path.join(vis_dir, "batch_0_comparison.png"),
                    nrow=noisy_images.shape[0],
                    normalize=False
                )
        
        # Compute and store final metrics
        cls_results = cls_metrics.compute(); img_results = img_metrics.compute()
        final_metrics = {k: v.item() for k, v in cls_results.items()}
        final_metrics.update({k: v.item() for k, v in img_results.items()})
        final_metrics['timestep'] = start_t
        results_summary.append(final_metrics)
        print(f"Results for T={start_t}: PSNR: {final_metrics['psnr']:.4f}, SSIM: {final_metrics['ssim']:.4f}, F1: {final_metrics['f1_score']:.4f}")

    # Final Report
    if results_summary:
        report_df = pd.DataFrame(results_summary)
        report_df = report_df[['timestep', 'psnr', 'ssim', 'f1_score', 'accuracy']].sort_values(by='psnr', ascending=False)
        print("\n\n--- Timestep Evaluation Summary ---"); print(report_df.to_string(index=False))
        report_path = os.path.join(args.output_dir, "summary_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\nSummary report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate D3PM model on real-world noisy images.")
    # 参数不变
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint .pth file.")
    parser.add_argument("-m", "--metadata_path", type=str, required=True, help="Path to the PAIRED dataset metadata JSON file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("-t", "--timesteps", type=int, nargs='+', required=True, help="A list of start timesteps (re-noise levels) to test (e.g., 50 100 150).")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use for computation.")

    args = parser.parse_args()
    main(args)