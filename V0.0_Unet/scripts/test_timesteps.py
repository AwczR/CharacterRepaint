# test_timesteps.py (FINAL VERSION - BASED ON REAL NOISY IMAGES)

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image

# --- 依赖导入 ---
# 确保 model.py 和 train.py 在正确的路径下
try:
    from model import UNetDenoisingModel
    # 我们需要从 train.py 导入的函数是通用的，所以保持不变
    from train import cosine_betas_for_alpha_bar, q_sample_discrete
except ImportError as e:
    print(f"Error importing from project files: {e}")
    print("Please ensure 'model.py' and 'train.py' are in the same directory or Python path.")
    exit(1)

# --- 数据集定义 (直接包含在此文件中，方便使用) ---
class PairedRubbingsDataset(Dataset):
    def __init__(self, metadata_path, image_size=(288, 288), binarization_threshold=0.5):
        with open(metadata_path, "r", encoding="utf-8") as f: self.metadata = json.load(f)
        self.base_noisy_dir = self.metadata["base_noisy_dir"]
        self.base_clean_dir = self.metadata["base_clean_dir"]
        self.image_pairs = self.metadata["image_pairs"]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            lambda x: (x > binarization_threshold).long()
        ])
    def __len__(self): return len(self.image_pairs)
    def __getitem__(self, idx):
        pair_info = self.image_pairs[idx]
        noisy_path = os.path.join(self.base_noisy_dir, pair_info["noisy_path_relative"])
        clean_path = os.path.join(self.base_clean_dir, pair_info["clean_path_relative"])
        try:
            noisy_image = self.transform(Image.open(noisy_path).convert("L"))
            clean_image = self.transform(Image.open(clean_path).convert("L"))
            return noisy_image, clean_image
        except Exception: return None

def paired_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None
    noisy, clean = zip(*batch)
    return torch.stack(noisy), torch.stack(clean)

# --- 重新实现的 p_sample_loop_restoration ---
# train.py中的版本是为从干净图像开始设计的。为了清晰，我们在这里重写一个。
@torch.no_grad()
def p_sample_loop_from_noisy(model, device, config, q_mats, input_noisy_image, start_timestep):
    """
    修改版的采样循环：从一个已有的噪声图像开始，再加噪，然后完全去噪。
    """
    model.eval()
    
    # 1. 对已有的噪声图像，再施加 t-step 的噪声
    t_tensor = torch.full((input_noisy_image.shape[0],), start_timestep - 1, device=device, dtype=torch.long)
    q_bar_t_start_2d = q_mats['q_mats_cumprod'][start_timestep - 1]
    q_bar_t_start = q_bar_t_start_2d.expand(input_noisy_image.shape[0], -1, -1)
    
    # q_sample_discrete 的输入必须是二值化的。我们的dataset已经做好了。
    img = q_sample_discrete(input_noisy_image, t_tensor, q_bar_t_start)
    
    # 2. 从 t 开始，执行完整的去噪循环
    loop_range = reversed(range(start_timestep))
    for t in loop_range:
        time = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
        pred_x0_logits = model(img.float(), time)
        
        if t == 0:
            img = torch.argmax(pred_x0_logits, dim=1).unsqueeze(1)
            break

        b, _, h, w = img.shape
        NUM_CLASSES = model.final_out_channels
        q_bar_t_minus_1_2d = q_mats['q_mats_cumprod'][t-1] if t > 0 else torch.eye(NUM_CLASSES, device=device)
        q_bar_t_minus_1 = q_bar_t_minus_1_2d.expand(b, -1, -1)
        q_t_2d = q_mats['q_one_step_mats'][t]
        q_t = q_t_2d.expand(b, -1, -1)
        x_t_one_hot_flat = F.one_hot(img.squeeze(1), num_classes=NUM_CLASSES).float().view(b, h*w, -1)
        pred_x0_probs_flat = F.softmax(pred_x0_logits, dim=1).permute(0, 2, 3, 1).reshape(b, h*w, -1)
        term1 = torch.bmm(x_t_one_hot_flat, q_t)
        term2 = torch.bmm(pred_x0_probs_flat, q_bar_t_minus_1)
        posterior_probs_log = torch.log(term1 + 1e-8) + torch.log(term2 + 1e-8)
        sampled_flat = torch.multinomial(F.softmax(posterior_probs_log.view(b*h*w, -1), dim=-1), num_samples=1)
        img = sampled_flat.view(b, 1, h, w)
    
    model.train() # 恢复模型到训练模式
    return img

# --- 辅助函数 (加载模型、绘图等，保持不变) ---
def load_model_and_config(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model_kwargs = config["model_params"].copy(); model_kwargs['num_classes'] = 2
    model = UNetDenoisingModel(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    num_timesteps = config["diffusion_params"]["num_timesteps"]; NUM_CLASSES = 2
    betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]; q_one_step_mats[i, 0, 0] = 1.0 - beta_t; q_one_step_mats[i, 1, 1] = 1.0 - beta_t; q_one_step_mats[i, 0, 1] = beta_t; q_one_step_mats[i, 1, 0] = beta_t
    q_mats_cumprod = torch.zeros_like(q_one_step_mats); current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps): current_mat = torch.matmul(q_one_step_mats[i], current_mat); q_mats_cumprod[i] = current_mat
    q_mats = {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}
    return model, config, q_mats

def prepare_dataloader(metadata_path, config, num_samples):
    full_dataset = PairedRubbingsDataset(metadata_path=metadata_path, image_size=(config["model_params"]["image_size"], config["model_params"]["image_size"]))
    if num_samples > len(full_dataset): num_samples = len(full_dataset)
    subset_indices = list(range(num_samples))
    test_subset = Subset(full_dataset, subset_indices)
    dataloader = DataLoader(test_subset, batch_size=num_samples, shuffle=False, collate_fn=paired_collate_fn)
    print(f"Using {len(test_subset)} image pairs for testing.")
    return dataloader

def save_performance_plots(df, output_dir):
    metrics = [col for col in df.columns if col != 'timestep']
    for metric in metrics:
        plt.figure(figsize=(10, 6)); plt.plot(df['timestep'], df[metric], marker='o', linestyle='-', label=metric.upper())
        plt.title(f'{metric.upper()} vs. Start Timestep'); plt.xlabel('Start Timestep (Additional Noise Level)'); plt.ylabel(metric.upper())
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'plot_{metric}_vs_timestep.png')); plt.close()
    print(f"Performance plots saved in '{output_dir}'")

# --- 主测试函数 (已修改) ---
@torch.no_grad()
def test_timesteps(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config, q_mats = load_model_and_config(args.checkpoint, device)
    dataloader = prepare_dataloader(args.metadata_path, config, args.batch_size)
    timesteps_to_test = sorted(args.timesteps)
    print(f"Will test the following start_timesteps on existing noisy images: {timesteps_to_test}")

    img_metrics = torchmetrics.MetricCollection({'psnr': PeakSignalNoiseRatio(data_range=1.0), 'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)}).to(device)
    cls_metrics = torchmetrics.MetricCollection({'accuracy': torchmetrics.Accuracy(task="binary"), 'f1_score': torchmetrics.F1Score(task="binary")}).to(device)

    results_log = []
    final_vis_grid_list = []
    
    noisy_images_real, clean_images_gt = next(iter(dataloader))
    noisy_images_real, clean_images_gt = noisy_images_real.to(device), clean_images_gt.to(device)

    # 生成可视化网格
    print("Generating images for visualization grid...")
    for i in tqdm(range(noisy_images_real.shape[0]), desc="Processing samples"):
        noisy_image_single = noisy_images_real[i:i+1]
        clean_image_single_gt = clean_images_gt[i:i+1]
        
        # 可视化列表第一列：Ground Truth Clean Image
        final_vis_grid_list.append(clean_image_single_gt.cpu())
        # 可视化列表第二列：Original Noisy Image
        final_vis_grid_list.append(noisy_image_single.cpu())
        
        for start_timestep in timesteps_to_test:
            restored_image = p_sample_loop_from_noisy(model, device, config, q_mats, noisy_image_single, start_timestep)
            final_vis_grid_list.append(restored_image.cpu())

    # 计算整体指标
    print("\nCalculating metrics for each timestep...")
    for start_timestep in tqdm(timesteps_to_test, desc="Aggregating Metrics"):
        img_metrics.reset(); cls_metrics.reset()
        restored_images = p_sample_loop_from_noisy(model, device, config, q_mats, noisy_images_real, start_timestep)
        img_metrics.update(restored_images.float(), clean_images_gt.float())
        cls_metrics.update(restored_images, clean_images_gt)
        current_results = {'timestep': start_timestep, **{k: v.item() for k, v in img_metrics.compute().items()}, **{k: v.item() for k, v in cls_metrics.compute().items()}}
        results_log.append(current_results)

    # 保存最终的可视化大图
    print("\nGenerating final comparison grid...")
    final_grid_tensor = torch.cat(final_vis_grid_list, dim=0)
    num_cols = 2 + len(timesteps_to_test) # 2 (Clean, Noisy) + N (restored)
    vis_path = os.path.join(args.output_dir, 'final_comparison_grid.png')
    save_image(final_grid_tensor.float(), vis_path, nrow=num_cols)
    print(f"Final comparison grid saved to: {vis_path}")
    print(f"Grid layout: {args.batch_size} rows, {num_cols} columns.")
    print(f"Columns: [Ground Truth Clean], [Original Noisy], [Restored from t={timesteps_to_test[0]}], ...")

    # 保存量化结果和性能图
    results_df = pd.DataFrame(results_log); csv_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nQuantitative results saved to: {csv_path}\n{results_df.to_string()}")
    save_performance_plots(results_df, args.output_dir)
    print("\nTesting complete.")

# --- 命令行接口 (保持不变) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test D3PM model performance by applying additional noise to existing noisy images.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth) file.")
    parser.add_argument("-m", "--metadata_path", type=str, required=True, help="Path to the dataset metadata JSON file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save results (visualizations, plots, csv).")
    parser.add_argument("-t", "--timesteps", type=int, nargs='+', required=True, help="A list of start_timesteps (additional noise levels) to test.")
    parser.add_argument("-bs", "--batch_size", type=int, default=5, help="Number of image pairs to sample from the dataset for testing.")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use for computation ('cuda' or 'cpu').")
    args = parser.parse_args()
    test_timesteps(args)