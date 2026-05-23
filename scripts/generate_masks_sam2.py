#!/usr/bin/env python3
"""
generate_masks_sam2.py
======================
使用 SAM2 (Segment Anything 2) 對 dynerf coffee_martini 的多視角 t=0 影像
進行多物件分割，產生以下輸出：

  data/dynerf/time0_coffee_martini/masks/
      binary/        人體二值遮罩  (人=255, 背景=0)  .png
      instance/      實例遮罩      (背景=0, 人=1, 酒瓶=2, 杯子=3...) .png
      preview/       可視化預覽 RGB  .png

物件 ID 定義：
  BACKGROUND = 0
  PERSON     = 1
  BOTTLE     = 2
  GLASS      = 3

使用方式 (在 sam2_env 環境執行):
  python scripts/generate_masks_sam2.py \
      --data_dir data/dynerf/time0_coffee_martini \
      --sam2_dir /tmp2/martinlin/sam2 \
      --checkpoint /tmp2/martinlin/sam2/checkpoints/sam2.1_hiera_large.pt \
      --model_cfg sam2.1_hiera_l.yaml \
      --device cuda:0

互動模式說明：
  - 每張影像會彈出視窗，顯示影像後等待輸入 prompt 點
  - 對 headless server，改用 --auto_prompt 自動用中心點
"""

import os
import sys
import argparse
import glob
import re
import json
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch

# ── 物件 ID 對應 ────────────────────────────────────────────────────────────
OBJECT_IDS = {
    "background": 0,
    "person":     1,
    "bottle":     2,
    "glass":      3,
}

# 每個物件的顯示顏色 (BGR for OpenCV)
PALETTE = {
    0: (0,   0,   0),    # background - black
    1: (0,   200, 80),   # person     - green
    2: (200, 80,  0),    # bottle     - blue
    3: (80,  0,   200),  # glass      - purple
}


def load_sam2_predictor(sam2_dir, checkpoint, model_cfg, device):
    """載入 SAM2 ImagePredictor"""
    sys.path.insert(0, sam2_dir)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # model_cfg 路徑：優先用絕對路徑，否則讓 hydra 從 sam2/configs 找
    cfg_candidates = [
        os.path.join(sam2_dir, "sam2", f"{model_cfg}"),
        os.path.join(sam2_dir, "sam2", f"configs/{model_cfg}"),
        model_cfg,
    ]
    cfg_path = None
    for c in cfg_candidates:
        if os.path.exists(c):
            cfg_path = c
            break
    if cfg_path is None:
        cfg_path = model_cfg  # fallback: hydra 自動搜尋

    print(f"[SAM2] building model from checkpoint: {checkpoint}")
    sam2_model = build_sam2(cfg_path, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"[SAM2] predictor loaded on {device}")
    return predictor


# ── Prompt 定義 ──────────────────────────────────────────────────────────────
# 每個物件預設的 point prompts (歸一化座標, 0~1)
# 針對 coffee_martini 場景的大致位置，可在 interactive mode 修正
DEFAULT_PROMPTS = {
    # 物件 ID → [(norm_x, norm_y, label)]  label: 1=foreground, 0=background
    "person": [
        (0.40, 0.40, 1),   # 人身體中心
        (0.40, 0.20, 1),   # 人頭部
        (0.10, 0.50, 0),   # 排除左側背景
        (0.90, 0.50, 0),   # 排除右側背景
    ],
    "bottle": [
        (0.80, 0.65, 1),   # 酒瓶群中心
        (0.85, 0.55, 1),   # 酒瓶上方
    ],
    "glass": [
        (0.47, 0.72, 1),   # 馬丁尼杯
        (0.47, 0.65, 1),   # 杯身
    ],
}


def get_point_prompts_for_image(img_np, object_name, norm_prompts):
    """
    將歸一化 prompt 轉為像素座標
    img_np: H×W×3 uint8
    norm_prompts: list of (norm_x, norm_y, label)
    returns: points [N,2], labels [N]
    """
    H, W = img_np.shape[:2]
    points, labels = [], []
    for nx, ny, lb in norm_prompts:
        points.append([nx * W, ny * H])
        labels.append(lb)
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def predict_mask_for_object(predictor, img_np, points, labels):
    """
    用 SAM2 ImagePredictor 預測單物件 mask
    returns: best_mask [H, W] bool
    """
    predictor.set_image(img_np)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    best_idx = np.argmax(scores)
    return masks[best_idx]   # bool H×W


def build_instance_mask(binary_masks_dict, img_shape):
    """
    合併多個二值遮罩為實例遮罩
    binary_masks_dict: {object_id: bool H×W}
    returns: instance_mask [H, W] uint8
    """
    H, W = img_shape[:2]
    instance = np.zeros((H, W), dtype=np.uint8)
    # 按優先順序塗色（後塗的會覆蓋前者，所以 person 最後塗以保證優先）
    priority_order = [
        (OBJECT_IDS["bottle"], "bottle"),
        (OBJECT_IDS["glass"],  "glass"),
        (OBJECT_IDS["person"], "person"),
    ]
    for obj_id, _ in priority_order:
        if obj_id in binary_masks_dict:
            instance[binary_masks_dict[obj_id]] = obj_id
    return instance


def colorize_instance_mask(instance_mask):
    """將實例遮罩轉為 RGB 可視化"""
    H, W = instance_mask.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for obj_id, bgr in PALETTE.items():
        rgb = (bgr[2], bgr[1], bgr[0])  # BGR→RGB
        vis[instance_mask == obj_id] = rgb
    return vis


def process_image(predictor, img_path, out_dir, device, auto_prompt=True,
                  custom_prompts=None):
    """
    處理單張影像，產生遮罩並儲存
    returns: dict with output paths
    """
    img_name = Path(img_path).stem      # e.g. "original_time0_0"
    img_pil  = Image.open(img_path).convert("RGB")
    img_np   = np.array(img_pil)        # H×W×3 uint8

    binary_masks = {}

    objects_to_segment = {
        "person": OBJECT_IDS["person"],
        "bottle": OBJECT_IDS["bottle"],
        "glass":  OBJECT_IDS["glass"],
    }

    for obj_name, obj_id in objects_to_segment.items():
        prompts = (custom_prompts or {}).get(obj_name, DEFAULT_PROMPTS.get(obj_name, []))
        if not prompts:
            print(f"  [skip] no prompts for {obj_name}")
            continue

        points, labels = get_point_prompts_for_image(img_np, obj_name, prompts)
        print(f"  [{obj_name}] predicting with {len(points)} points ...")
        mask = predict_mask_for_object(predictor, img_np, points, labels)
        binary_masks[obj_id] = mask

    # ── 儲存二值人體遮罩 ──
    bin_dir = os.path.join(out_dir, "binary")
    os.makedirs(bin_dir, exist_ok=True)
    if OBJECT_IDS["person"] in binary_masks:
        person_mask_uint8 = (binary_masks[OBJECT_IDS["person"]] * 255).astype(np.uint8)
        bin_path = os.path.join(bin_dir, f"{img_name}.png")
        Image.fromarray(person_mask_uint8, mode="L").save(bin_path)

    # ── 儲存實例遮罩 ──
    inst_dir = os.path.join(out_dir, "instance")
    os.makedirs(inst_dir, exist_ok=True)
    instance_mask = build_instance_mask(binary_masks, img_np.shape)
    inst_path = os.path.join(inst_dir, f"{img_name}.png")
    Image.fromarray(instance_mask, mode="L").save(inst_path)

    # ── 儲存可視化預覽 ──
    prev_dir = os.path.join(out_dir, "preview")
    os.makedirs(prev_dir, exist_ok=True)
    vis_mask = colorize_instance_mask(instance_mask)
    # overlay on original image
    alpha = 0.55
    preview = (img_np * (1 - alpha) + vis_mask * alpha).astype(np.uint8)
    prev_path = os.path.join(prev_dir, f"{img_name}.png")
    Image.fromarray(preview).save(prev_path)

    print(f"  ✓ saved: instance={inst_path}")
    return {
        "binary":   bin_path if OBJECT_IDS["person"] in binary_masks else None,
        "instance": inst_path,
        "preview":  prev_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate SAM2 masks for dynerf coffee_martini")
    parser.add_argument("--data_dir",   default="data/dynerf/time0_coffee_martini",
                        help="Path to the time0 multi-view data directory")
    parser.add_argument("--sam2_dir",   default="/tmp2/martinlin/sam2",
                        help="Path to cloned SAM2 repository")
    parser.add_argument("--checkpoint", default="/tmp2/martinlin/sam2/checkpoints/sam2.1_hiera_large.pt",
                        help="SAM2 checkpoint .pt file")
    parser.add_argument("--model_cfg",  default="sam2.1_hiera_l.yaml",
                        help="SAM2 model config yaml filename")
    parser.add_argument("--device",     default="cuda:0",
                        help="Torch device (cuda:0, cpu, ...)")
    parser.add_argument("--auto_prompt", action="store_true", default=True,
                        help="Use pre-defined default prompts (no GUI)")
    parser.add_argument("--prompts_json", default=None,
                        help="Optional JSON file with custom prompts per camera")
    parser.add_argument("--output_subdir", default="masks",
                        help="Sub-directory name under data_dir for mask output")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = os.path.join(data_dir, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # ── 找所有影像 ──
    pattern = os.path.join(data_dir, "original_time0_*.png")
    img_paths = sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1)))
    if not img_paths:
        print(f"[ERROR] no images found at {pattern}")
        sys.exit(1)
    print(f"Found {len(img_paths)} images in {data_dir}")

    # ── 載入 SAM2 ──
    predictor = load_sam2_predictor(
        args.sam2_dir, args.checkpoint, args.model_cfg, args.device
    )

    # ── 載入自定義 prompts (可選) ──
    custom_prompts_per_cam = {}
    if args.prompts_json and os.path.exists(args.prompts_json):
        with open(args.prompts_json) as f:
            custom_prompts_per_cam = json.load(f)
        print(f"Loaded custom prompts from {args.prompts_json}")

    # ── 逐張處理 ──
    results = {}
    for img_path in img_paths:
        cam_id = re.search(r"_(\d+)\.png$", img_path).group(1)
        print(f"\n[{cam_id}] Processing: {img_path}")
        cam_prompts = custom_prompts_per_cam.get(cam_id, None)
        result = process_image(predictor, img_path, out_dir, args.device,
                               auto_prompt=args.auto_prompt,
                               custom_prompts=cam_prompts)
        results[cam_id] = result

    # ── 儲存結果摘要 ──
    summary_path = os.path.join(out_dir, "mask_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Done! Masks saved to: {out_dir}")
    print(f"   Summary: {summary_path}")
    print(f"\nInstance ID mapping:")
    for name, id_ in OBJECT_IDS.items():
        print(f"   {name:12s} = {id_}")


if __name__ == "__main__":
    main()
