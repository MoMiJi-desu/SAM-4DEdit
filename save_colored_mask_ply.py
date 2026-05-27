import torch
from plyfile import PlyData, PlyElement
import numpy as np
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser
import sys

# To easily color the PLY, we can just edit the f_dc parameters and set them to specific SH coefficients.
# The SH DC term is related to RGB by: RGB = SH_C0 * f_dc + 0.5
# SH_C0 = 0.28209479177387814

def create_colored_ply():
    ply_path = "/tmp2/martinlin/Instruct-4DGS/output/dynerf/coffee_martini_masked/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000/point_cloud.ply"
    out_path = "/tmp2/martinlin/Instruct-4DGS/output/dynerf/coffee_martini_masked/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000/point_cloud_mask_visualized.ply"
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    mask = np.array(vertex['mask'])
    # sigmoid
    mask_sigmoid = 1 / (1 + np.exp(-mask))
    
    # Let's make background (mask < 0.5) BLUE, and foreground (mask > 0.5) RED
    # DC = (RGB - 0.5) / 0.28209
    
    # Red: (1, 0, 0)
    red_dc = (np.array([1.0, 0.0, 0.0]) - 0.5) / 0.28209479177387814
    # Blue: (0, 0, 1)
    blue_dc = (np.array([0.0, 0.0, 1.0]) - 0.5) / 0.28209479177387814
    
    f_dc_0 = np.where(mask_sigmoid > 0.5, red_dc[0], blue_dc[0])
    f_dc_1 = np.where(mask_sigmoid > 0.5, red_dc[1], blue_dc[1])
    f_dc_2 = np.where(mask_sigmoid > 0.5, red_dc[2], blue_dc[2])
    
    # Update the data
    vertex.data['f_dc_0'] = f_dc_0
    vertex.data['f_dc_1'] = f_dc_1
    vertex.data['f_dc_2'] = f_dc_2
    
    # Zero out the rest SH to make the colors flat
    for i in range(45):
        vertex.data[f'f_rest_{i}'] = np.zeros_like(f_dc_0)
        
    PlyData([vertex]).write(out_path)
    print(f"Saved colored mask PLY to {out_path}")
    print(f"Foreground points (Red): {(mask_sigmoid > 0.5).sum()}")
    print(f"Background points (Blue): {(mask_sigmoid <= 0.5).sum()}")

if __name__ == "__main__":
    create_colored_ply()
