import os
import sys
import glob
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Hybrid Images using IP2P")
    parser.add_argument("--data_dir", default="/tmp2/martinlin/Instruct-4DGS/data/dynerf/time0_coffee_martini")
    parser.add_argument("--prompt_fg", default="Turn him into a Van Gogh painting")
    parser.add_argument("--prompt_bg", default="none")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Paths
    img_pattern = os.path.join(args.data_dir, "original_time0_*.png")
    img_paths = sorted(glob.glob(img_pattern), key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1)))
    
    if not img_paths:
        print(f"[ERROR] No images found at {img_pattern}")
        return

    mask_dir = os.path.join(args.data_dir, "masks", "binary")
    out_dir = os.path.join(args.data_dir, "hybrid")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading InstructPix2Pix on {args.device}...")
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None,
        cache_dir="/tmp2/martinlin/.cache/huggingface_new"
    ).to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"\nConfiguration:")
    print(f"  Foreground Prompt: '{args.prompt_fg}'")
    if args.prompt_bg and args.prompt_bg.lower() != "none":
        print(f"  Background Prompt: '{args.prompt_bg}'")
    else:
        print(f"  Background: kept original")
    print(f"  Output Dir: {out_dir}")
    print(f"  Total Images: {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="Processing images"):
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, base_name)
        
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found for {base_name}, skipping.")
            continue

        # Load original and mask
        orig_img = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Convert mask to float numpy array [0.0, 1.0]
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1) # (H, W, 1)

        # Generate Foreground
        fg_edited = pipe(
            args.prompt_fg, 
            image=orig_img, 
            num_inference_steps=20, 
            image_guidance_scale=args.image_guidance_scale,
            guidance_scale=args.guidance_scale
        ).images[0]
        fg_edited = fg_edited.resize(orig_img.size, Image.Resampling.LANCZOS)
        
        # Generate Background if prompt is provided and not 'none'
        if args.prompt_bg and args.prompt_bg.lower() != "none":
            bg_edited = pipe(
                args.prompt_bg,
                image=orig_img,
                num_inference_steps=20,
                image_guidance_scale=args.image_guidance_scale,
                guidance_scale=args.guidance_scale
            ).images[0]
            bg_edited = bg_edited.resize(orig_img.size, Image.Resampling.LANCZOS)
            bg_np = np.array(bg_edited, dtype=np.float32)
        else:
            bg_np = np.array(orig_img, dtype=np.float32)

        # Composite
        fg_np = np.array(fg_edited, dtype=np.float32)

        # I_hybrid = I_fg * mask + I_bg * (1 - mask)
        hybrid_np = fg_np * mask_np + bg_np * (1.0 - mask_np)
        hybrid_np = hybrid_np.clip(0, 255).astype(np.uint8)

        # Save
        out_path = os.path.join(out_dir, base_name)
        Image.fromarray(hybrid_np).save(out_path)

    print("\n✅ Hybrid images generation complete!")

if __name__ == "__main__":
    main()
