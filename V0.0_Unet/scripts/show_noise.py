# visualize_forward_process.py

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

# --- 依赖导入 ---
# 从之前的脚本中复用必要的函数
try:
    from train import cosine_betas_for_alpha_bar, q_sample_discrete
except ImportError as e:
    print(f"Error importing from project files: {e}")
    print("Please ensure 'train.py' is in the same directory or Python path.")
    exit(1)

# --- 数据集定义 (与 test_timesteps.py 中的相同) ---
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

def add_text_to_image(image_tensor, text):
    """在 PyTorch 张量图像上添加文字"""
    # 将张量转换为 PIL Image
    image = transforms.ToPILImage()(image_tensor.cpu().float())
    
    # 准备绘图
    draw = ImageDraw.Draw(image)
    try:
        # 尝试加载一个常见的字体，如果找不到，使用默认字体
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()
        
    # 获取文本尺寸
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 计算文本位置（左上角）
    pos = (5, 5)
    
    # 绘制黑色背景框
    bg_pos = (pos[0], pos[1], pos[0] + text_width + 4, pos[1] + text_height + 4)
    draw.rectangle(bg_pos, fill="black")
    
    # 绘制白色文字
    draw.text(pos, text, font=font, fill="white")
    
    # 将 PIL Image 转换回张量
    return transforms.ToTensor()(image)


def build_diffusion_scheduler(num_timesteps, device):
    """构建扩散过程所需的Q矩阵"""
    NUM_CLASSES = 2
    betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]
        q_one_step_mats[i, 0, 0] = 1.0 - beta_t
        q_one_step_mats[i, 1, 1] = 1.0 - beta_t
        q_one_step_mats[i, 0, 1] = beta_t
        q_one_step_mats[i, 1, 0] = beta_t
    
    q_mats_cumprod = torch.zeros_like(q_one_step_mats)
    current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        current_mat = torch.matmul(q_one_step_mats[i], current_mat)
        q_mats_cumprod[i] = current_mat
        
    return {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}


def visualize_forward_process(args):
    """主可视化函数"""
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 准备数据和扩散调度器 ---
    # 使用 PairedRubbingsDataset
    full_dataset = PairedRubbingsDataset(
        metadata_path=args.metadata_path, 
        image_size=(args.image_size, args.image_size)
    )
    
    # 随机选择样本
    if args.num_samples > len(full_dataset):
        print(f"Warning: Requested {args.num_samples} samples, but dataset only has {len(full_dataset)}. Using all.")
        args.num_samples = len(full_dataset)
    
    random_indices = np.random.choice(len(full_dataset), args.num_samples, replace=False)
    subset = Subset(full_dataset, random_indices)
    dataloader = DataLoader(subset, batch_size=args.num_samples, shuffle=False, collate_fn=paired_collate_fn)
    
    noisy_images_real, clean_images_gt = next(iter(dataloader))
    noisy_images_real, clean_images_gt = noisy_images_real.to(device), clean_images_gt.to(device)
    
    print(f"Selected {len(random_indices)} random samples for visualization.")
    
    # 构建扩散调度器
    q_mats = build_diffusion_scheduler(args.total_timesteps, device)
    timesteps_to_show = sorted(args.timesteps)
    print(f"Will visualize noise at timesteps: {timesteps_to_show}")

    # --- 2. 生成图像并添加到列表 ---
    final_vis_grid_list = []

    for i in range(args.num_samples):
        # 获取单个图像对
        noisy_single = noisy_images_real[i:i+1]
        clean_single = clean_images_gt[i:i+1]

        # 第一行：原始图像
        final_vis_grid_list.append(add_text_to_image(clean_single.squeeze(0), "Original Clean"))
        final_vis_grid_list.append(add_text_to_image(noisy_single.squeeze(0), "Original Noisy"))

        # 后续行：不同timestep下的加噪结果
        for t in timesteps_to_show:
            t_tensor = torch.full((1,), t - 1, device=device, dtype=torch.long)
            q_bar_t = q_mats['q_mats_cumprod'][t_tensor]

            # 对干净图像加噪
            noised_clean = q_sample_discrete(clean_single, t_tensor, q_bar_t)
            final_vis_grid_list.append(add_text_to_image(noised_clean.squeeze(0), f"Clean | t={t}"))

            # 对噪声图像加噪
            noised_noisy = q_sample_discrete(noisy_single, t_tensor, q_bar_t)
            final_vis_grid_list.append(add_text_to_image(noised_noisy.squeeze(0), f"Noisy | t={t}"))
            
    # --- 3. 保存最终的可视化大图 ---
    print("\nGenerating final visualization grid...")
    final_grid_tensor = torch.stack(final_vis_grid_list)
    
    # 每行显示一对 (clean, noisy) 的加噪结果
    num_cols = 2 
    
    # 调整排列方式，使得同一timestep的 clean/noisy 对显示在一起
    # 原始排列: [C0, N0, C0_t1, N0_t1, C0_t2, N0_t2, ..., C1, N1, C1_t1, N1_t1, ...]
    # 我们希望的排列是按行排列
    
    vis_path = os.path.join(args.output_dir, 'forward_process_visualization.png')
    
    # 每行的图像数量是 2 (原始对) + 2 * len(timesteps_to_show) (加噪对)
    # 不，这样太宽了。我们让每行只显示一对(clean, noisy)，垂直向下是时间步
    # 新的排列:
    # [Orig_C_0, Orig_N_0], [Orig_C_1, Orig_N_1], ...
    # [C_0_t1,   N_0_t1],   [C_1_t1,   N_1_t1], ...
    # [C_0_t2,   N_0_t2],   [C_1_t2,   N_1_t2], ...
    
    reordered_list = []
    num_timesteps_shown = len(timesteps_to_show)
    num_images_per_sample = 2 * (1 + num_timesteps_shown)
    
    for row in range(1 + num_timesteps_shown): # 遍历时间行 (0=原始, 1=t1, 2=t2...)
        for col in range(args.num_samples): # 遍历样本列
            # Clean image
            reordered_list.append(final_vis_grid_list[col * num_images_per_sample + row * 2])
            # Noisy image
            reordered_list.append(final_vis_grid_list[col * num_images_per_sample + row * 2 + 1])

    final_grid_tensor = torch.stack(reordered_list)
    num_cols = args.num_samples * 2

    save_image(final_grid_tensor.float(), vis_path, nrow=num_cols)
    print(f"Final visualization grid saved to: {vis_path}")
    print(f"Grid layout: {1 + num_timesteps_shown} rows, {args.num_samples} pairs of columns.")
    print("Each pair of columns represents a random sample.")
    print("Rows represent increasing noise levels (Original, t1, t2, ...).")
    print("\nVisualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the D3PM forward process on clean and noisy images.")
    parser.add_argument("-m", "--metadata_path", type=str, required=True,
                        help="Path to the dataset metadata JSON file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save the visualization grid.")
    parser.add_argument("-t", "--timesteps", type=int, nargs='+', required=True,
                        help="A list of timesteps to visualize (e.g., 50 150 250 500).")
    parser.add_argument("-n", "--num_samples", type=int, default=4,
                        help="Number of random image pairs to sample from the dataset.")
    parser.add_argument("--total_timesteps", type=int, default=1000,
                        help="The total number of timesteps in the diffusion process (must match training).")
    parser.add_argument("--image_size", type=int, default=288,
                        help="Image size to resize to.")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device to use for computation ('cuda' or 'cpu').")
    
    args = parser.parse_args()
    visualize_forward_process(args)