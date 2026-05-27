import os
import torch
import sys
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from tqdm import tqdm

def train_mask_quick(dataset, opt, pipe, hp, ply_path=None):
    first_iter = 0
    tb_writer = None
    
    if ply_path:
        dataset.ply_path = ply_path
        
    # We will load the Fauvism edited point cloud so we don't lose the color!
    gaussians = GaussianModel(dataset.sh_degree, hp)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    
    # Setup Optimizer ONLY for the mask
    l = [
        {'params': [gaussians._mask], 'lr': 0.1, "name": "mask"},
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Collect ONLY time0 cameras that have masks
    train_cams = scene.getTrainCameras()
    mask_cams = [cam for cam in train_cams if hasattr(cam, 'mask') and cam.mask is not None]
    print(f"Found {len(mask_cams)} cameras with masks!")
    
    if len(mask_cams) == 0:
        print("Error: No masks found! Check dataset.py parsing.")
        return

    progress_bar = tqdm(range(first_iter, 1000), desc="Training Mask")
    
    for iteration in range(first_iter + 1, 1001):
        iter_start.record()
        
        # Pick a random mask camera
        viewpoint_cam = mask_cams[randint(0, len(mask_cams)-1)]
        
        # Render Mask
        mask_val = torch.sigmoid(gaussians._mask) # (N, 1)
        override_color = mask_val.repeat(1, 3) # (N, 3)
        
        # We must use black background to avoid mask bleeding
        black_background = torch.zeros_like(background)
        render_mask_pkg = render(viewpoint_cam, gaussians, pipe, black_background, override_color=override_color, stage="fine", cam_type=scene.dataset_type)
        rendered_mask = render_mask_pkg["render"][0] # Take first channel
        
        gt_mask = viewpoint_cam.mask.cuda() # [H, W]
        
        loss = l1_loss(rendered_mask, gt_mask)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        iter_end.record()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)

    # Save the updated point cloud back to the same path
    print(f"Saving updated mask to: {dataset.ply_path}")
    gaussians.save_ply(dataset.ply_path)
    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training mask script")
    from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str)
    args = parser.parse_args(sys.argv[1:])
    
    if args.ply_path:
        lp.ply_path = args.ply_path
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    safe_state(False)
    
    train_mask_quick(lp.extract(args), op.extract(args), pp.extract(args), hp.extract(args), ply_path=args.ply_path)
