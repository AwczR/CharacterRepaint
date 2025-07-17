# debug_pipeline.py (FINAL & FULLY-FIXED for Debugging)
"""
Diagnostic script to quickly identify issues in the discrete diffusion training pipeline.
Tests data generation, diffusion steps, model forward pass, loss computation,
optimizer step, and CRITICALLY, noise addition on a real image from the dataset,
and also tests the full p_sample_loop_restoration logic for one step.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import sys
import argparse
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 从您的 train.py 导入核心函数 (确保这些函数是最新的，包含所有修复)
from train import make_conservative_schedule, q_sample_discrete, RubbingsDataset, custom_collate
# 导入您的模型定义
from model import AttentionUNetDenoisingModel

# --- Configuration (Hardcoded for Debugging, now with metadata_path) ---
DEBUG_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": 288, # 使用实际图像尺寸
    "num_timesteps": 100, # 使用较少时间步
    "model_channels": 16, # 使用极小模型
    "channel_mult": [1, 2], # 只有两层
    "num_res_blocks": 1,
    "groups": 4,
    "attention_resolutions": [], # 完全禁用注意力
    "lr": 0.001, # 较高的学习率，确保有明显更新
    "beta_max": 0.02, # 降低 beta_max，确保噪声增长平缓
    "time_exp": 1.0, # 均匀采样
    "output_debug_dir": "debug_output",
    "metadata_path": None # 占位符，由命令行参数传入
}

os.makedirs(DEBUG_PARAMS["output_debug_dir"], exist_ok=True)
print(f"--- Starting Debug Pipeline Test ---")
print(f"Using device: {DEBUG_PARAMS['device']}")
print(f"Outputting debug images to: {DEBUG_PARAMS['output_debug_dir']}")

# --- 核心：复制 train.py 中 p_sample_loop_restoration 的最新修复逻辑 ---
@torch.no_grad()
def p_sample_loop_restoration_debug(model, device, cfg, q_mats, x_start_input, start_t, is_real_image_test=False):
    """
    一个用于调试的 p_sample_loop_restoration 版本，可以根据需要从干净图像或噪声图像开始。
    如果 is_real_image_test 为 True，则 x_start_input 是噪声图像，我们直接从它去噪。
    否则，x_start_input 是干净图像，我们先对其加噪声。
    """
    model.eval()
    B, _, H, W = x_start_input.shape
    
    noisy_img_at_start_t = None
    if is_real_image_test:
        img = x_start_input.clone()
        noisy_img_at_start_t = x_start_input.clone()
    else:
        q_bar = q_mats['q_mats_cumprod'][start_t - 1].expand(B, -1, -1)
        noisy_img_at_start_t = q_sample_discrete(x_start_input, torch.full((B,), start_t - 1, device=device), q_bar)
        img = noisy_img_at_start_t.clone()

    for t in tqdm(reversed(range(start_t)), desc="Denoising", total=start_t, leave=False):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        logits = model(img.float(), t_tensor)

        q_btm1 = q_mats['q_mats_cumprod'][t - 1] if t > 0 else torch.eye(2, device=device)
        qt = q_mats['q_one_step_mats'][t]
        
        qt_b = qt.expand(B, -1, -1)
        q_btm1_b = q_btm1.expand(B, -1, -1)

        x0_prob_pred = F.softmax(logits, 1).permute(0, 2, 3, 1).reshape(B, H * W, 2)
        x_t_flat_idx = img.squeeze(1).view(B, H * W)
        pred_x0_classes = x0_prob_pred.argmax(dim=-1)

        log_posterior = torch.zeros(B, H * W, 2, device=device) 

        # For x_{t-1} = 0
        log_posterior[:, :, 0] = (
            torch.log(torch.gather(qt_b[:, 0, :], 1, x_t_flat_idx) + 1e-8) +
            torch.log(q_btm1_b[torch.arange(B).unsqueeze(1).expand(-1, H*W), pred_x0_classes, 0] + 1e-8)
        )

        # For x_{t-1} = 1
        log_posterior[:, :, 1] = (
            torch.log(torch.gather(qt_b[:, 1, :], 1, x_t_flat_idx) + 1e-8) +
            torch.log(q_btm1_b[torch.arange(B).unsqueeze(1).expand(-1, H*W), pred_x0_classes, 1] + 1e-8)
        )
        
        post_prob = F.softmax(log_posterior.view(-1, 2), dim=-1)
        img = torch.multinomial(post_prob, 1).view(B, 1, H, W)

    model.train()
    return img, noisy_img_at_start_t

def run_debug_test_pipeline(metadata_path):
    DEBUG_PARAMS["metadata_path"] = metadata_path

    # --- Step 0: Load a real image from dataset ---
    print("\n[Step 0/8] Loading a real image from dataset...")
    try:
        dataset = RubbingsDataset(DEBUG_PARAMS["metadata_path"], (DEBUG_PARAMS["image_size"],) * 2)
        if len(dataset) == 0:
            print("[ERROR] Dataset is empty! Cannot load real image.")
            sys.exit(1)
        
        x_start_real = None
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample is not None:
                x_start_real = sample.unsqueeze(0).to(DEBUG_PARAMS["device"])
                break
        if x_start_real is None:
            print("[ERROR] Failed to load any valid image from dataset.")
            sys.exit(1)
        
        save_image(x_start_real.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "00_x_start_real.png"))
        print(f"  Loaded real x_start (clean image) of shape {x_start_real.shape}. Saved to 00_x_start_real.png.")
    except Exception as e:
        print(f"  [ERROR] Failed to load real image from dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    B, C, H, W = x_start_real.shape


    # --- Step 1: Data Generation (using dummy data for internal consistency checks) ---
    print("\n[Step 1/8] Testing Dummy Data Generation (for internal consistency checks)...")
    x_start_dummy = torch.zeros(1, 1, DEBUG_PARAMS["image_size"], DEBUG_PARAMS["image_size"], dtype=torch.long, device=DEBUG_PARAMS["device"])
    block_size_dummy = DEBUG_PARAMS["image_size"] // 4
    start_h_dummy = DEBUG_PARAMS["image_size"] // 2 - block_size_dummy // 2
    end_h_dummy = DEBUG_PARAMS["image_size"] // 2 + block_size_dummy // 2
    start_w_dummy = DEBUG_PARAMS["image_size"] // 2 - block_size_dummy // 2
    end_w_dummy = DEBUG_PARAMS["image_size"] // 2 + block_size_dummy // 2
    x_start_dummy[:, :, start_h_dummy:end_h_dummy, start_w_dummy:end_w_dummy] = 1
    save_image(x_start_dummy.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "01_x_start_dummy.png"))
    print(f"  Generated dummy x_start of shape {x_start_dummy.shape}. Saved to 01_x_start_dummy.png.")


    # --- Step 2: Diffusion Parameters (betas, q_one_step, q_cumprod) ---
    print("\n[Step 2/8] Testing Diffusion Parameters Calculation...")
    T = DEBUG_PARAMS["num_timesteps"]
    betas = make_conservative_schedule(T, max_beta=DEBUG_PARAMS["beta_max"]).to(DEBUG_PARAMS["device"])

    q_one_step = torch.zeros(T, 2, 2, device=DEBUG_PARAMS["device"])
    q_one_step[:, 0, 0] = 1 - betas
    q_one_step[:, 1, 1] = 1 - betas
    q_one_step[:, 0, 1] = betas
    q_one_step[:, 1, 0] = betas

    q_cumprod = torch.zeros_like(q_one_step)
    current_prod = torch.eye(2, device=DEBUG_PARAMS["device"])
    for i in range(T):
        current_prod = torch.matmul(q_one_step[i], current_prod)
        q_cumprod[i] = current_prod

    q_mats = {"q_one_step_mats": q_one_step, "q_mats_cumprod": q_cumprod}

    print(f"  Betas (min/max): {betas.min().item():.4f}/{betas.max().item():.4f}")
    print(f"  q_one_step[0]:\n{q_one_step[0].cpu().numpy()}")
    print(f"  q_cumprod[0] (should be q_one_step[0]):\n{q_cumprod[0].cpu().numpy()}")
    print(f"  q_cumprod[T-1] (should approach [[0.5, 0.5], [0.5, 0.5]]):\n{q_cumprod[T-1].cpu().numpy()}")
    if not torch.allclose(q_cumprod[0], q_one_step[0]):
        print("  [ERROR] q_cumprod[0] does not match q_one_step[0]!")
        sys.exit(1)
    if not torch.allclose(q_cumprod[T-1], torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=DEBUG_PARAMS["device"]), atol=0.05):
        print("  [WARNING] q_cumprod[T-1] does not approach 0.5, check beta schedule or T.")
    print("  Diffusion parameters calculation successful.")


    # --- Step 3: q_sample_discrete (Noise Addition on REAL image) ---
    print("\n[Step 3/8] Testing q_sample_discrete (Noise Addition) on REAL image...")
    t_test_low = torch.tensor([0], device=DEBUG_PARAMS["device"], dtype=torch.long)
    t_test_mid = torch.tensor([T // 2], device=DEBUG_PARAMS["device"], dtype=torch.long)
    t_test_high = torch.tensor([T - 1], device=DEBUG_PARAMS["device"], dtype=torch.long)

    noisy_low_t_real = q_sample_discrete(x_start_real, t_test_low, q_cumprod[t_test_low[0]].unsqueeze(0))
    noisy_mid_t_real = q_sample_discrete(x_start_real, t_test_mid, q_cumprod[t_test_mid[0]].unsqueeze(0))
    noisy_high_t_real = q_sample_discrete(x_start_real, t_test_high, q_cumprod[t_test_high[0]].unsqueeze(0))

    save_image(noisy_low_t_real.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "02_noisy_low_t_real.png"))
    save_image(noisy_mid_t_real.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "03_noisy_mid_t_real.png"))
    save_image(noisy_high_t_real.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "04_noisy_high_t_real.png"))

    low_noise_diff_real = (noisy_low_t_real != x_start_real).float().mean().item()
    mid_noise_diff_real = (noisy_mid_t_real != x_start_real).float().mean().item()
    high_noise_diff_real = (noisy_high_t_real != x_start_real).float().mean().item()

    print(f"  Real image Pixel diff from x_start at t=0: {low_noise_diff_real:.4f} (should be very low)")
    print(f"  Real image Pixel diff from x_start at t={T//2}: {mid_noise_diff_real:.4f} (should be around 0.2-0.4)")
    print(f"  Real image Pixel diff from x_start at t={T-1}: {high_noise_diff_real:.4f} (should be around 0.5)")

    if low_noise_diff_real > 0.05:
        print("  [WARNING] q_sample_discrete at t=0 on real image seems to add too much noise.")
    if high_noise_diff_real < 0.45 or high_noise_diff_real > 0.55:
        print("  [WARNING] q_sample_discrete at max t on real image does not seem to be near random (0.5).")
    print("  q_sample_discrete test on real image successful. Check images for visual confirmation.")


    # --- Step 4: Model Initialization and Forward Pass ---
    print("\n[Step 4/8] Testing Model Initialization and Forward Pass...")
    model = AttentionUNetDenoisingModel(
        image_size=DEBUG_PARAMS["image_size"],
        in_channels=1,
        num_classes=2,
        model_channels=DEBUG_PARAMS["model_channels"],
        channel_mult=tuple(DEBUG_PARAMS["channel_mult"]),
        num_res_blocks=DEBUG_PARAMS["num_res_blocks"],
        groups=DEBUG_PARAMS["groups"],
        attention_resolutions=tuple(DEBUG_PARAMS["attention_resolutions"]),
        attention_head_dim=8
    ).to(DEBUG_PARAMS["device"])

    model_input_img = noisy_mid_t_real.float()
    model_input_t = t_test_mid

    try:
        logits = model(model_input_img, model_input_t)
        print(f"  Model forward pass successful. Output logits shape: {logits.shape}")
        if logits.shape != (B, 2, H, W):
            print(f"  [ERROR] Model output shape mismatch! Expected {(B, 2, H, W)}, got {logits.shape}")
            sys.exit(1)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("  [ERROR] Model output contains NaN or Inf values!")
            sys.exit(1)
        
        logits_std = logits.std().item()
        if logits_std < 0.01:
            print(f"  [WARNING] Model logits have very low standard deviation ({logits_std:.4f}), might be collapsing to constant output.")

        pred_probs = F.softmax(logits, dim=1)[:, 1:, :, :]
        pred_binary = (pred_probs > 0.5).long()
        save_image(pred_probs.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "05_model_pred_probs_real.png"))
        save_image(pred_binary.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], "06_model_pred_binary_real.png"))
        print("  Model output probabilities and binary prediction saved for real image.")

    except Exception as e:
        print(f"  [ERROR] Model initialization or forward pass failed: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Step 5: Loss Function and Backprop (Single Step) ---
    print("\n[Step 5/8] Testing Loss Function and Backpropagation (single step)...")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.55], device=DEBUG_PARAMS["device"]))

    target_label = x_start_real.squeeze(1)

    try:
        loss = criterion(logits, target_label)
        print(f"  Loss calculated: {loss.item():.4f}")
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("  [ERROR] Loss is NaN or Inf!")
            sys.exit(1)
        
        initial_param = model.model.conv_in.weight.clone()

        model.zero_grad()
        loss.backward()
        
        gradient_exists = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 1e-6:
                gradient_exists = True
                break
        
        if not gradient_exists:
            print("  [ERROR] No valid gradients found after backward pass! Check loss calculation or model design.")
            sys.exit(1)
        else:
            print("  Gradients successfully computed.")

        # --- Step 6: Optimizer Step ---
        print("\n[Step 6/8] Testing Optimizer Step...")
        optimizer = optim.AdamW(model.parameters(), lr=DEBUG_PARAMS["lr"], weight_decay=0.01)
        optimizer.step()

        final_param = model.model.conv_in.weight.clone()
        if torch.equal(initial_param, final_param):
            print("  [ERROR] Model parameters did not update after optimizer step! Check optimizer, learning rate, or gradients.")
            sys.exit(1)
        else:
            print("  Model parameters updated successfully.")

    except Exception as e:
        print(f"  [ERROR] Loss computation or backpropagation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Step 7: Testing Full Denoising Chain (p_sample_loop_restoration_debug) ---
    print("\n[Step 7/8] Testing Full Denoising Chain (p_sample_loop_restoration_debug) on REAL image...")
    # 从真实图片加噪到高噪声，然后用模型去噪
    start_t_denoising_test = DEBUG_PARAMS["num_timesteps"] - 1 # 从最高噪声开始
    
    try:
        # x_start_real is the clean image. We set is_real_image_test=False to simulate training noise addition.
        restored_from_high_noise, initial_noisy_for_denoising_test = p_sample_loop_restoration_debug(
            model, DEBUG_PARAMS["device"], DEBUG_PARAMS, q_mats, x_start_real, start_t_denoising_test, is_real_image_test=False
        )
        
        save_image(initial_noisy_for_denoising_test.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], f"08_noisy_for_denoising_T_{start_t_denoising_test}_real.png"))
        save_image(restored_from_high_noise.float(), os.path.join(DEBUG_PARAMS["output_debug_dir"], f"09_restored_from_high_noise_T_{start_t_denoising_test}_real.png"))
        
        print(f"  Full denoising chain test successful from t={start_t_denoising_test}. Images saved.")

        from torchmetrics.image import PeakSignalNoiseRatio
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEBUG_PARAMS["device"])
        psnr_val = psnr_metric(restored_from_high_noise.float(), x_start_real.float()).item()
        print(f"  PSNR for full denoising chain (T={start_t_denoising_test}): {psnr_val:.3f}")
        if psnr_val < 3.0:
             print("  [WARNING] PSNR is extremely low. Full denoising chain might be generating pure noise.")


    except Exception as e:
        print(f"  [ERROR] Full denoising chain test failed: {e}")
        traceback.print_exc()
        sys.exit(1)


    print("\n--- Debug Pipeline Test Completed Successfully! ---")
    print("If all steps reported success, your core training pipeline logic is likely correct.")
    print("The problem might then be related to data characteristics, full model complexity, or scale.")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug discrete diffusion pipeline components.")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to your dataset metadata JSON (e.g., CIRI_syth_metadata.json)")
    args = parser.parse_args()
    
    run_debug_test_pipeline(args.metadata_path)