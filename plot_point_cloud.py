import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData
import numpy as np

def visualize_point_cloud():
    ply_path = "/tmp2/martinlin/Instruct-4DGS/output/dynerf/coffee_martini_masked/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000/point_cloud.ply"
    out_img = "/tmp2/martinlin/Instruct-4DGS/mask_visualization.png"
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    mask = np.array(vertex['mask'])
    
    # Sigmoid mask
    mask_sigmoid = 1 / (1 + np.exp(-mask))
    
    # Separate foreground and background
    fg_indices = np.where(mask_sigmoid > 0.5)[0]
    bg_indices = np.where(mask_sigmoid <= 0.5)[0]
    
    # Subsample background so it doesn't clutter the plot
    bg_sample = np.random.choice(bg_indices, size=10000, replace=False)
    
    # Also subsample foreground if needed (4600 is fine to plot fully)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Background (Blue)
    ax.scatter(x[bg_sample], y[bg_sample], z[bg_sample], c='blue', s=0.1, alpha=0.1, label='Background')
    
    # Foreground (Red)
    ax.scatter(x[fg_indices], y[fg_indices], z[fg_indices], c='red', s=2.0, alpha=1.0, label='Foreground (Masked)')
    
    # Adjust view angle for better visualization (front view typically)
    ax.view_init(elev=0, azim=180)
    
    ax.set_title("3D Point Cloud Mask Visualization")
    ax.legend()
    
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_img}")

if __name__ == "__main__":
    visualize_point_cloud()
