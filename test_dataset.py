import torch
from argparse import ArgumentParser
from arguments import ModelParams
from scene import Scene
import sys

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
model.source_path = "./data/dynerf/coffee_martini"
model.model_path = "./output/dynerf/coffee_martini_masked"
model.eval = True
args = parser.parse_args([])
args = model.extract(args)

scene = Scene(args, gaussians=None, load_iteration=-1, shuffle=False)
train_cams = scene.getTrainCameras()

mask_count = 0
for cam in train_cams:
    if hasattr(cam, 'mask') and cam.mask is not None:
        mask_count += 1
        
print("Number of cameras with mask:", mask_count)
