#!/usr/bin/env python3
"""
generate_masks_sam.py
=====================
使用 SAM (Segment Anything Model v1) 對 dynerf coffee_martini
多視角 t=0 影像分割「人」，產生二值遮罩：

  人   = 255 (白)
  背景 = 0   (黑)

策略：
  1. SamAutomaticMaskGenerator 自動找所有物件
  2. 從所有候選 mask 中，選出「人」—— 條件：
     - 面積介於 total_pixels 的 5%~50%
     - mask 中心 x 落在 [0.2, 0.8] 範圍內（排除側邊物件）
     - mask 中心 y 落在 [0.2, 0.85] 範圍內（排除天花板/桌面）
     - 取上述候選中面積最大者
  3. 若自動選取失敗，fallback 到 point-prompt 模式

輸出目錄：
  {data_dir}/masks/
      binary/    {image_name}.png  (uint8 L, 0 or 255)
      preview/   {image_name}.png  (RGB 疊圖預覽)

環境：conda activate Gaussians4D
  pip install segment-anything opencv-python

執行：
  CUDA_VISIBLE_DEVICES=0 \\
  /tmp2/martinlin/miniconda3/envs/Gaussians4D/bin/python \\
      scripts/generate_masks_sam.py \\
      --data_dir data/dynerf/time0_coffee_martini \\
      --checkpoint /tmp2/martinlin/sam_checkpoints/sam_vit_h_4b8939.pth \\
      --device cuda:0
"""

import os, sys, argparse, glob, re, json
from pathlib import Path

import numpy as np
from PIL import Image

# ── Automatic mask 選人標準 ───────────────────────────────────────────────────
AREA_MIN_RATIO = 0.04   # 人至少佔畫面 4%
AREA_MAX_RATIO = 0.50   # 人最多佔畫面 50%
CENTER_X_MIN   = 0.15   # 人中心 x 不能太靠邊
CENTER_X_MAX   = 0.85
CENTER_Y_MIN   = 0.20   # 不能在天花板
CENTER_Y_MAX   = 0.88   # 不能在桌面最前方

# ── Point-prompt fallback（歸一化 0~1） ───────────────────────────────────────
FALLBACK_PROMPTS = [
    (0.40, 0.57, 1),   # 人身體（深色圍裙中央）
    (0.43, 0.65, 1),   # 人腰部
    (0.40, 0.37, 1),   # 人臉 / 頸部
    (0.30, 0.10, 0),   # 窗簾左上
    (0.55, 0.08, 0),   # 窗簾中上
    (0.05, 0.50, 0),   # 左牆
    (0.93, 0.50, 0),   # 右窗
    (0.50, 0.92, 0),   # 桌面
]


def load_auto_predictor(checkpoint: str, device: str):
    """載入 SAM AutomaticMaskGenerator"""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    ckpt = os.path.basename(checkpoint)
    model_type = "vit_h" if "vit_h" in ckpt else "vit_l" if "vit_l" in ckpt else "vit_b"
    print(f"[SAM] Loading {model_type} ...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    auto_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=500,
    )
    predictor = SamPredictor(sam)
    print(f"[SAM] Ready on {device}")
    return auto_gen, predictor


def select_person_mask(masks_data: list, H: int, W: int):
    """
    從 SamAutomaticMaskGenerator 輸出中選出最可能是「人」的 mask
    每個 mask_data: {'segmentation': ndarray bool, 'area': int, 'bbox': [x,y,w,h], ...}
    回傳 bool mask (H,W) 或 None
    """
    total_px = H * W
    candidates = []
    for m in masks_data:
        seg  = m['segmentation']   # bool H×W
        area = seg.sum()

        # 面積過濾
        ratio = area / total_px
        if ratio < AREA_MIN_RATIO or ratio > AREA_MAX_RATIO:
            continue

        # 中心座標過濾
        ys, xs = np.where(seg)
        cx = xs.mean() / W
        cy = ys.mean() / H
        if not (CENTER_X_MIN <= cx <= CENTER_X_MAX):
            continue
        if not (CENTER_Y_MIN <= cy <= CENTER_Y_MAX):
            continue

        candidates.append((area, seg, cx, cy))

    if not candidates:
        return None

    # 取面積最大的
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_area, best_seg, cx, cy = candidates[0]
    coverage = best_area / total_px * 100
    print(f" [auto] selected: area={coverage:.1f}%, center=({cx:.2f},{cy:.2f})", end="")
    return best_seg


def predict_person_fallback(predictor, img_np: np.ndarray) -> np.ndarray:
    """Point-prompt fallback，回傳 bool mask (H,W)"""
    H, W = img_np.shape[:2]
    pts    = np.array([[nx * W, ny * H] for nx, ny, _ in FALLBACK_PROMPTS], dtype=np.float32)
    labels = np.array([lb for _, _, lb in FALLBACK_PROMPTS], dtype=np.int32)
    predictor.set_image(img_np)
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=True,
    )
    return masks[np.argmax(scores)]


def process_image(auto_gen, predictor, img_path: str, out_dir: str,
                  norm_prompts=None) -> dict:
    img_name = Path(img_path).stem
    img_pil  = Image.open(img_path).convert("RGB")
    img_np   = np.array(img_pil, dtype=np.uint8)
    H, W     = img_np.shape[:2]

    if norm_prompts:   # 「標注工具提供的点」模式
        pts    = np.array([[nx * W, ny * H] for nx, ny, _ in norm_prompts], dtype=np.float32)
        labels = np.array([int(lb) for _, _, lb in norm_prompts], dtype=np.int32)
        print(f"  point-prompt ({len(pts)} pts fg={labels.sum()}) ...", end="", flush=True)
        predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True,
        )
        mask_bool = masks[np.argmax(scores)]
    else:              # 自動模式
        print(f"  generating masks ...", end="", flush=True)
        all_masks = auto_gen.generate(img_np)
        print(f" {len(all_masks)} candidates,", end="", flush=True)
        mask_bool = select_person_mask(all_masks, H, W)
        if mask_bool is None:
            print(f" [WARN] auto failed, using fallback ...", end="", flush=True)
            mask_bool = predict_person_fallback(predictor, img_np)

    coverage = mask_bool.mean() * 100
    print(f" coverage={coverage:.1f}%")

    # ── binary mask ──────────────────────────────────────────────────────────
    bin_dir  = os.path.join(out_dir, "binary")
    os.makedirs(bin_dir, exist_ok=True)
    bin_path = os.path.join(bin_dir, f"{img_name}.png")
    Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L").save(bin_path)

    # ── preview ───────────────────────────────────────────────────────────────
    prev_dir  = os.path.join(out_dir, "preview")
    os.makedirs(prev_dir, exist_ok=True)
    overlay   = np.zeros_like(img_np)
    overlay[mask_bool] = (80, 200, 80)
    preview   = (img_np * 0.55 + overlay * 0.45).clip(0, 255).astype(np.uint8)
    prev_path = os.path.join(prev_dir, f"{img_name}.png")
    Image.fromarray(preview).save(prev_path)

    return {"binary": bin_path, "preview": prev_path, "coverage": float(coverage)}


def main():
    parser = argparse.ArgumentParser(description="SAM automatic person/background mask")
    parser.add_argument("--data_dir",
                        default="/tmp2/martinlin/Instruct-4DGS/data/dynerf/time0_coffee_martini")
    parser.add_argument("--checkpoint",
                        default="/tmp2/martinlin/sam_checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prompts_json", default=None,
                        help="JSON produced by annotate_server.py: "
                             "{cam_id: [[nx,ny,label], ...]}")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "masks")
    os.makedirs(out_dir, exist_ok=True)

    pattern   = os.path.join(args.data_dir, "original_time0_*.png")
    img_paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1))
    )
    if not img_paths:
        print(f"[ERROR] No images at {pattern}"); sys.exit(1)
    print(f"Found {len(img_paths)} images")

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}"); sys.exit(1)

    auto_gen, predictor = load_auto_predictor(args.checkpoint, args.device)

    results = {}
    # 載入標注 prompts
    custom_prompts = {}
    if args.prompts_json and os.path.exists(args.prompts_json):
        with open(args.prompts_json) as f:
            custom_prompts = json.load(f)
        print(f"Loaded prompts for {len(custom_prompts)} cameras from {args.prompts_json}")

    for img_path in img_paths:
        cam_id = re.search(r"_(\d+)\.png$", img_path).group(1)

        if args.skip_existing:
            p = os.path.join(out_dir, "binary", f"{Path(img_path).stem}.png")
            if os.path.exists(p):
                print(f"[cam {cam_id}] skip"); results[cam_id] = {"binary": p}; continue

        print(f"\n[cam {cam_id}] {os.path.basename(img_path)}")
        # 取得此相機的標注点（格式: [[nx, ny, label], ...]）
        raw = custom_prompts.get(cam_id)
        if raw:
            norm_prompts = [tuple(p) for p in raw]
        else:
            norm_prompts = None

        result = process_image(auto_gen, predictor, img_path, out_dir, norm_prompts)
        results[cam_id] = result

    with open(os.path.join(out_dir, "mask_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    coverages = [v["coverage"] for v in results.values() if "coverage" in v]
    print(f"\n{'='*55}")
    print(f"✅  Done!  {len(results)} masks saved to: {out_dir}")
    if coverages:
        print(f"    Avg coverage: {np.mean(coverages):.1f}%  "
              f"(min={np.min(coverages):.1f}%, max={np.max(coverages):.1f}%)")
    print(f"    Mask: person=255, background=0  (uint8 L PNG)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
