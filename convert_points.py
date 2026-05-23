import numpy as np
import open3d as o3d

# 設定檔案路徑
npy_path = "/tmp2/martinlin/Instruct-4DGS/data/dycheck/mochi-high-five/points.npy"
ply_path = "/tmp2/martinlin/Instruct-4DGS/data/dycheck/mochi-high-five/points3D_downsample2.ply"

print(f"正在讀取 {npy_path} ...")
points_data = np.load(npy_path)

# 1. 取出 XYZ 座標
xyz = points_data[:, :3]
num_points = xyz.shape[0]

# 建立 Open3D 點雲物件
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# 2. 處理顏色 (Red, Green, Blue)
if points_data.shape[1] >= 6:
    print("偵測到原始顏色資料，正在套用...")
    colors = points_data[:, 3:6]
    if colors.max() > 1.0:
        colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
else:
    print("未偵測到顏色，正在填補預設灰色...")
    default_colors = np.ones((num_points, 3)) * 0.5
    pcd.colors = o3d.utility.Vector3dVector(default_colors)

# 3. 關鍵修復：處理法向量 (nx, ny, nz)
print("正在填補預設法向量 (nx, ny, nz)...")
# 建立一個全部朝上的假法向量 [0.0, 1.0, 0.0] 來騙過 Dataloader
default_normals = np.zeros((num_points, 3))
default_normals[:, 1] = 1.0 
pcd.normals = o3d.utility.Vector3dVector(default_normals)

# 儲存檔案
o3d.io.write_point_cloud(ply_path, pcd)
print(f"轉換成功！包含 XYZ、顏色、法向量的完美點雲已儲存至：{ply_path}")