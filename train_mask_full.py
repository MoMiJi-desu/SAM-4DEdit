import os
import torch
import sys
import numpy as np
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

def fix_all_masks():
    parser = ArgumentParser(description="Fix all masks")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the edit_3d output PLY to inject mask into")
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
    
    edit_ply_path = args.ply_path
    print(f"Injecting mask into: {edit_ply_path}")

    
    # Load the colored point cloud
    gaussians.load_ply(edit_ply_path)
    
    # Default everything to Background (sigmoid(-10) = 0)
    new_mask = torch.full_like(gaussians._mask, -10.0, device="cuda")
    
    train_cams = scene.getTrainCameras()
    
    # Collect all cameras that have a mask
    masked_cams = [cam for cam in train_cams if hasattr(cam, 'mask') and cam.mask is not None]
    print(f"Found {len(masked_cams)} cameras with masks!")
    
    with torch.no_grad():
        for cam in masked_cams:
            # Get camera properties
            W, H = cam.image_width, cam.image_height
            mask_2d = cam.mask.cuda() # [1, H, W]
            
            # Project all 3D points to this camera's 2D plane
            xyz = gaussians.get_xyz
            
            # Transform to camera space
            # cam.world_view_transform is [4, 4]
            xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
            xyz_cam = torch.matmul(xyz_hom, cam.world_view_transform.cuda())
            
            # Filter points behind camera
            valid_depth = xyz_cam[:, 2] > 0
            
            # Transform to NDC space
            xyz_ndc = torch.matmul(xyz_hom, cam.full_proj_transform.cuda())
            
            # Perspective divide
            xyz_ndc = xyz_ndc[:, :3] / (xyz_ndc[:, 3:] + 1e-7)
            
            # NDC to pixel coordinates
            u = ((xyz_ndc[:, 0] + 1) * W - 1) * 0.5
            v = ((xyz_ndc[:, 1] + 1) * H - 1) * 0.5
            
            u = torch.round(u).long()
            v = torch.round(v).long()
            
            # Filter points outside image
            valid_u = (u >= 0) & (u < W)
            valid_v = (v >= 0) & (v < H)
            
            valid_points = valid_depth & valid_u & valid_v
            
            # For valid points, check if they fall in the foreground mask
            valid_indices = torch.where(valid_points)[0]
            
            u_valid = u[valid_indices]
            v_valid = v[valid_indices]
            
            # mask_2d shape is [1, H, W]
            mask_vals = mask_2d[0, v_valid, u_valid]
            
            # If a point is foreground in THIS camera, set its mask to 10.0 (Foreground)
            foreground_indices = valid_indices[mask_vals > 0.5]
            new_mask[foreground_indices] = 10.0
            
            print(f"Camera {cam.image_name}: Found {len(foreground_indices)} foreground points")
            
    gaussians._mask = new_mask
    
    # Save back to the edit path!
    gaussians.save_ply(edit_ply_path)
    print(f"Successfully injected perfect multi-camera mask into {edit_ply_path}")

if __name__ == "__main__":
    fix_all_masks()
