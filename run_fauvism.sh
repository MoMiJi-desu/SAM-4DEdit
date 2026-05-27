#!/bin/bash
set -e

echo "[1/3] Generating Fauvism Hybrid Images..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/generate_hybrid_images.py --prompt_fg "Make it look like a fauvism painting"

echo "[2/3] Running edit_3d.py..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python edit_3d.py \
    --configs "./arguments/dynerf/coffee_martini.py" \
    --ply_path "./output/dynerf/coffee_martini/point_cloud/iteration_14000/point_cloud.ply" \
    -s "./data/dynerf/coffee_martini" \
    --model_path "./output/dynerf/coffee_martini" \
    --dataset "dynerf" \
    --scene "coffee_martini" \
    --prompt "Make it look like a fauvism painting"

echo "[3/3] Running fully_edit_sds.py..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python fully_edit_sds.py \
    --configs "./arguments/dynerf/coffee_martini.py" \
    --ply_path "./output/dynerf/coffee_martini/point_cloud_3dedit/Make it look like a fauvism painting/iteration_1000/point_cloud.ply" \
    -s "./data/dynerf/coffee_martini" \
    --model_path "./output/dynerf/coffee_martini" \
    --prompt_fg "Make it look like a fauvism painting" \
    --prompt_bg "none" \
    --guidance_scale 10.5 \
    --image_guidance_scale 1.2

echo "🎉 All Done!"
