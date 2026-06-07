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
    parser.add_argument("--ply_path", type=str, default="")
    parser.add_argument("--out", type=str, default="preview_rendered.png")
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

    # Load colmap scene / background
    # We don't need to load the full scene, but we need the cameras
    # Actually, initializing Scene loads all cameras.
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)

    # Load the specific PLY
    edit_ply_path = args.ply_path
    if edit_ply_path:
        print(f"Loading specifically edited PLY: {edit_ply_path}")
        gaussians.load_ply(edit_ply_path)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_cam = scene.getTrainCameras().copy()[0] # Take first training camera
    
    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_img = render_pkg["render"]
    
    # Save image
    import cv2
    import numpy as np
    
    rendered_img_np = rendered_img.detach().cpu().numpy().transpose(1, 2, 0)
    rendered_img_np = (rendered_img_np * 255).astype(np.uint8)
    rendered_img_bgr = cv2.cvtColor(rendered_img_np, cv2.COLOR_RGB2BGR)
    
    out_path = args.out
    cv2.imwrite(out_path, rendered_img_bgr)
    print(f"Saved preview visualization to {out_path}")

if __name__ == "__main__":
    render_preview()
