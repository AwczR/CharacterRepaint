# inference.py (FINAL - Correct Implementation for Image Restoration)
"""
Performs a controlled denoising process (SDEdit-like) on a given input image
by first noising it to a specified timestep and then denoising it back.
"""
import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from PIL import Image

# Assuming model.py and train.py are in the same directory or accessible
from model import UNetDenoisingModel
from train import cosine_betas_for_alpha_bar, q_sample_discrete

@torch.no_grad()
def p_sample_loop_restoration(model, device, config, q_mats, input_image, start_timestep, num_vis_steps=10):
    """
    The full reverse diffusion sampling loop for image restoration.
    """
    model.eval()
    
    # --- 1. Noise the input image to the start_timestep ---
    print(f"Applying controlled noise to the input image up to timestep t={start_timestep}...")
    shape = input_image.shape
    t_tensor = torch.full((shape[0],), start_timestep - 1, device=device, dtype=torch.long) # t is 0-indexed
    
    # Get the cumulative transition matrix for the starting step
    q_bar_t_start = q_mats['q_mats_cumprod'][start_timestep - 1].unsqueeze(0)
    
    # Start the process from the noised version of the input image
    img = q_sample_discrete(input_image, t_tensor, q_bar_t_start)
    print("Controlled noising complete. Starting denoising loop from this state.")
    
    # --- 2. Denoise from start_timestep back to 0 ---
    vis_steps = []
    vis_schedule = torch.linspace(start_timestep - 1, 0, num_vis_steps, dtype=torch.long)
    
    # The loop now starts from start_timestep, not from the maximum
    loop_range = reversed(range(start_timestep))
    
    print("Starting controlled denoising loop...")
    for t in tqdm(loop_range, desc="Restoring", total=start_timestep):
        time = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        if t == start_timestep - 1 or t in vis_schedule:
            vis_steps.append(img.clone())

        pred_x0_logits = model(img.float(), time)
        
        if t == 0:
            pred_x0 = torch.argmax(pred_x0_logits, dim=1).unsqueeze(1)
            img = pred_x0
            break

        q_bar_t_minus_1 = (q_mats['q_mats_cumprod'][t-1] if t > 0 else torch.eye(model.final_out_channels, device=device)).unsqueeze(0)
        q_t = q_mats['q_one_step_mats'][t].unsqueeze(0)
        
        b, c, h, w = img.shape
        x_t_one_hot_flat = F.one_hot(img.squeeze(1), num_classes=model.final_out_channels).float().view(b, h*w, model.final_out_channels)
        pred_x0_probs_flat = F.softmax(pred_x0_logits, dim=1).permute(0, 2, 3, 1).reshape(b, h*w, model.final_out_channels)
        
        term1 = torch.bmm(x_t_one_hot_flat, q_t)
        term2 = torch.bmm(pred_x0_probs_flat, q_bar_t_minus_1)
        
        posterior_probs_log = torch.log(term1 + 1e-8) + torch.log(term2 + 1e-8)
        
        posterior_probs_reshaped = posterior_probs_log.view(b * h * w, model.final_out_channels)
        sampled_xt_minus_1_flat = torch.multinomial(F.softmax(posterior_probs_reshaped, dim=-1), num_samples=1)
        
        img = sampled_xt_minus_1_flat.view(b, h, w).unsqueeze(1)
            
    vis_steps.append(img.clone())

    # --- 3. Create and Save the Visualization Grid ---
    if vis_steps:
        print("Saving restoration process as a grid image...")
        # Add the original input image to the start of the visualization
        vis_steps.insert(0, input_image.clone())
        
        grid = make_grid(
            torch.cat(vis_steps, dim=0),
            nrow=len(vis_steps),
            padding=5,
            pad_value=0.5
        )
        grid_path = os.path.join(os.path.dirname(config['output_path']), 'restoration_process_grid.png')
        save_image(grid.float(), grid_path)
        print(f"Process visualization grid saved to {grid_path}")

    return img


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint_path}")
    
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt['config']
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
        
    config['output_path'] = args.output_path

    NUM_CLASSES = 2
    model_kwargs = config["model_params"].copy()
    model_kwargs['num_classes'] = NUM_CLASSES
    model = UNetDenoisingModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model loaded from epoch {ckpt['epoch']}.")

    num_timesteps = config["diffusion_params"]["num_timesteps"]
    betas = cosine_betas_for_alpha_bar(num_timesteps).to(device)
    q_one_step_mats = torch.zeros(num_timesteps, NUM_CLASSES, NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        beta_t = betas[i]
        q_one_step_mats[i, 0, 0] = 1.0 - beta_t; q_one_step_mats[i, 1, 1] = 1.0 - beta_t
        q_one_step_mats[i, 0, 1] = beta_t; q_one_step_mats[i, 1, 0] = beta_t
    
    q_mats_cumprod = torch.zeros_like(q_one_step_mats)
    current_mat = torch.eye(NUM_CLASSES, device=device)
    for i in range(num_timesteps):
        current_mat = torch.matmul(q_one_step_mats[i], current_mat)
        q_mats_cumprod[i] = current_mat
        
    q_mats = {"q_one_step_mats": q_one_step_mats, "q_mats_cumprod": q_mats_cumprod}

    image_size = (config["model_params"]["image_size"], config["model_params"]["image_size"])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        lambda x: (x > 0.5).long()
    ])
    
    if not args.input_path or not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input image not found or not specified. Please provide a valid path with -i.")
        
    print(f"Loading input image from: {args.input_path}")
    pil_image = Image.open(args.input_path).convert("L")
    input_image = transform(pil_image).unsqueeze(0).to(device)
    
    # <<<< CORE LOGIC CHANGE: We now call the new restoration loop >>>>
    restored_image = p_sample_loop_restoration(
        model, device, config, q_mats, input_image, 
        start_timestep=args.start_timestep,
        num_vis_steps=args.num_vis_steps
    )
    
    save_image(restored_image.float(), args.output_path)
    print(f"Final restored image saved to: {args.output_path}")

# inference.py (FIX for AttributeError)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore a degraded image using a trained D3PM model.")
    
    parser.add_argument("-i", "--input_path", type=str, required=True, 
                        help="Path to the input degraded image.")
                        
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True, 
                        help="Path to the trained model checkpoint (.pth file).")
                        
    parser.add_argument("-o", "--output_path", type=str, default="output/restored_image.png", 
                        help="Path to save the final restored output image.")
    
    # The crucial parameter to control restoration strength
    parser.add_argument("-t", "--start_timestep", type=int, default=250, 
                        help="The timestep to noise the input image to before starting the denoising process. "
                             "Higher values mean more noise and creative freedom (e.g., 500). "
                             "Lower values mean more fidelity to the original (e.g., 150).")
    
    parser.add_argument("--num_vis_steps", type=int, default=10, 
                        help="Number of intermediate steps to visualize in the grid.")
                        
    # <<<< THIS IS THE CORRECTED LINE >>>>
    parser.add_argument("-d", "--device", type=str, default="cuda", 
                        help="Device to use ('cuda' or 'cpu').")
    
    args = parser.parse_args()
    main(args)