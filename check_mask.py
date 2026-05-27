import torch
from plyfile import PlyData

plydata = PlyData.read("/tmp2/martinlin/Instruct-4DGS/output/dynerf/coffee_martini_masked/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000/point_cloud.ply")

print(plydata.elements[0].data.dtype.names)
if 'mask' in plydata.elements[0].data.dtype.names:
    mask = torch.tensor(plydata.elements[0]['mask'])
    print("Mask max:", mask.max())
    print("Mask min:", mask.min())
    print("Mask mean:", mask.mean())
    
    mask_sigmoid = torch.sigmoid(mask)
    print("Mask sigmoid max:", mask_sigmoid.max())
    print("Mask sigmoid min:", mask_sigmoid.min())
    print("Mask sigmoid mean:", mask_sigmoid.mean())
    print("Num points > 0.5:", (mask_sigmoid > 0.5).sum().item())
else:
    print("No mask in PLY!")
