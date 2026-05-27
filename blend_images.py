import cv2
import numpy as np

def blend():
    # The rendered red/blue mask image from camera 04
    mask_render_path = "/tmp2/martinlin/Instruct-4DGS/output/dynerf/coffee_martini_masked/point_cloud/iteration_99999/renders/iteration_99999/train/renders/00000/cam04.png"
    # The original image
    orig_path = "/tmp2/martinlin/Instruct-4DGS/data/dynerf/coffee_martini/cam04/images/0000.png"
    
    out_path = "/tmp2/martinlin/Instruct-4DGS/overlay_visualization.png"
    
    # Read images
    mask_img = cv2.imread(mask_render_path)
    orig_img = cv2.imread(orig_path)
    
    if mask_img is None:
        print(f"Error: Could not read {mask_render_path}")
        return
        
    if orig_img is None:
        print(f"Error: Could not read {orig_path}")
        return
        
    # Resize mask image to original image size if they differ
    if mask_img.shape != orig_img.shape:
        mask_img = cv2.resize(mask_img, (orig_img.shape[1], orig_img.shape[0]))
        
    # We want to overlay the RED part heavily, and maybe keep the blue part transparent
    # Let's create an alpha mask based on red vs blue
    # mask_img is in BGR format
    b, g, r = cv2.split(mask_img)
    
    # Where red is dominant over blue
    is_foreground = (r > b + 20).astype(np.float32)
    
    # Overlay with 60% opacity on foreground, 0% on background
    alpha = 0.6 * is_foreground[:, :, np.newaxis]
    
    blended = (orig_img * (1 - alpha) + mask_img * alpha).astype(np.uint8)
    
    cv2.imwrite(out_path, blended)
    print(f"Saved overlay visualization to {out_path}")

if __name__ == "__main__":
    blend()
