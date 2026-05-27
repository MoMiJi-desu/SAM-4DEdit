from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                # dynerf branch uses Neural3D_NDC_Dataset which returns a tuple
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
                hybrid_image = None
                
                # Dynamic mask/hybrid loading for dynerf
                if self.dataset_type == "dynerf" and hasattr(self.dataset, 'image_paths') and hasattr(self.dataset, 'root_dir'):
                    import os
                    from PIL import Image
                    from pathlib import Path
                    
                    img_path = self.dataset.image_paths[index]
                    cam_dir = Path(img_path).parts[-3] # e.g. 'cam00'
                    frame_name = Path(img_path).stem # e.g. '0000'
                    
                    if frame_name == "0000" and cam_dir.startswith("cam"):
                        cam_idx = int(cam_dir.replace("cam", ""))
                        mask_path = os.path.join(self.dataset.root_dir, "../time0_coffee_martini/masks/binary", f"original_time0_{cam_idx}.png")
                        hybrid_path = os.path.join(self.dataset.root_dir, "../time0_coffee_martini/hybrid", f"original_time0_{cam_idx}.png")
                        
                        if os.path.exists(mask_path):
                            mask_pil = Image.open(mask_path).convert("L")
                            mask_tensor = PILtoTorch(mask_pil, None)
                            mask = (mask_tensor > 0.5).float() # [1, H, W]
                            
                        if os.path.exists(hybrid_path):
                            hybrid_pil = Image.open(hybrid_path).convert("RGB")
                            hybrid_image = PILtoTorch(hybrid_pil, None)[:3, :, :]
                            
            except:
                # Other datasets (e.g. colmap) pass a list of CameraInfo
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                
                mask = caminfo.mask
                hybrid_image = getattr(caminfo, 'hybrid_image', None)
                
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, hybrid_image=hybrid_image)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
