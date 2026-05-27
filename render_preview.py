import os
import torch
import cv2
import sys
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

def render_preview():
    parser = ArgumentParser(description="Render Preview")
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
    
    gaussians = GaussianModel(dataset.sh_degree, hidden)
    scene = Scene(dataset, gaussians, load_iteration=14000, shuffle=False)
    
    # Load the 3D-edited point cloud!
    edit_ply_path = "./output/dynerf/coffee_martini_masked/point_cloud_3dedit/van gogh person fauvism background/iteration_1000/point_cloud.ply"
    gaussians.load_ply(edit_ply_path)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cams = scene.getTrainCameras()
    target_cam = train_cams[0]
        
    print(f"Rendering camera {target_cam.image_name}")
    
    with torch.no_grad():
        render_pkg = render(target_cam, gaussians, pipe, background, cam_type=scene.dataset_type)
        rendered_image = render_pkg["render"]
        
    # Convert to 8-bit BGR for OpenCV
    rendered_img_np = rendered_image.cpu().numpy()
    rendered_img_np = (np.clip(rendered_img_np, 0, 1) * 255).astype(np.uint8)
    rendered_img_bgr = np.transpose(rendered_img_np, (1, 2, 0))[:, :, ::-1] # RGB to BGR
    
    out_path = "/tmp2/martinlin/Instruct-4DGS/edit_3d_preview.png"
    cv2.imwrite(out_path, rendered_img_bgr)
    print(f"Saved preview visualization to {out_path}")

if __name__ == "__main__":
    render_preview()
