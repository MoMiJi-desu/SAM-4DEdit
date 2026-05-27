import os
import torch
import cv2
import sys
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

def render_and_blend():
    parser = ArgumentParser(description="Render and Blend")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--configs", type=str)
    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    hidden = hp.extract(args)
    
    # We load iteration 99999
    gaussians = GaussianModel(dataset.sh_degree, hidden)
    scene = Scene(dataset, gaussians, load_iteration=99999, shuffle=False)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cams = scene.getTrainCameras()
    
    # Find a camera with mask (which means time0)
    target_cam = None
    for cam in train_cams:
        if hasattr(cam, 'mask') and cam.mask is not None:
            target_cam = cam
            break
            
    if target_cam is None:
        target_cam = train_cams[0]
        
    print(f"Rendering camera {target_cam.image_name}")
    
    with torch.no_grad():
        render_pkg = render(target_cam, gaussians, pipe, background, cam_type=scene.dataset_type)
        rendered_image = render_pkg["render"]
        
    # Convert to 8-bit BGR for OpenCV
    rendered_img_np = rendered_image.cpu().numpy()
    rendered_img_np = (np.clip(rendered_img_np, 0, 1) * 255).astype(np.uint8)
    rendered_img_bgr = np.transpose(rendered_img_np, (1, 2, 0))[:, :, ::-1] # RGB to BGR
    
    # Original image
    if scene.dataset_type != "PanopticSports":
        orig_img = target_cam.original_image[:3, :, :]
    else:
        orig_img = target_cam['image']
        
    orig_img_np = orig_img.cpu().numpy()
    orig_img_np = (np.clip(orig_img_np, 0, 1) * 255).astype(np.uint8)
    orig_img_bgr = np.transpose(orig_img_np, (1, 2, 0))[:, :, ::-1]
    
    # Blend
    b, g, r = cv2.split(rendered_img_bgr)
    # The mask foreground was set to red (r > b)
    is_foreground = (r.astype(np.int32) > b.astype(np.int32) + 20).astype(np.float32)
    
    alpha = 0.6 * is_foreground[:, :, np.newaxis]
    
    blended = (orig_img_bgr * (1 - alpha) + rendered_img_bgr * alpha).astype(np.uint8)
    
    out_path = "/tmp2/martinlin/Instruct-4DGS/overlay_visualization.png"
    cv2.imwrite(out_path, blended)
    print(f"Saved overlay visualization to {out_path}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    render_and_blend()
