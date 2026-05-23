#!/usr/bin/env python3
"""
tune_sam2_prompts.py
====================
互動式 GUI 工具，用於可視化並調整 SAM2 prompt 點位，讓用戶
在每張相機影像上點選前景/背景點來精確分割每個物件。

使用方式：
  python scripts/tune_sam2_prompts.py \
      --data_dir data/dynerf/time0_coffee_martini \
      --sam2_dir /tmp2/martinlin/sam2 \
      --checkpoint /tmp2/martinlin/sam2/checkpoints/sam2.1_hiera_large.pt \
      --model_cfg sam2.1_hiera_l.yaml \
      --device cuda:0

操作說明：
  - 滑鼠左鍵：新增前景點 (綠色)
  - 滑鼠右鍵：新增背景點 (紅色)
  - 按 n/m    ：切換物件 (person / bottle / glass)
  - 按 Enter  ：執行分割並顯示結果
  - 按 s      ：儲存此相機的 prompts 到 JSON
  - 按 q      ：跳至下一張影像
  - 按 ESC    ：結束並儲存所有 prompts

輸出：
  data/dynerf/time0_coffee_martini/masks/custom_prompts.json
"""
import os, sys, json, glob, re, argparse
import numpy as np
from PIL import Image

OBJECT_NAMES = ["person", "bottle", "glass"]
OBJECT_COLORS = {
    "person": (0, 200, 80),
    "bottle": (200, 80, 0),
    "glass":  (80, 0, 200),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/dynerf/time0_coffee_martini")
    parser.add_argument("--sam2_dir",   default="/tmp2/martinlin/sam2")
    parser.add_argument("--checkpoint", default="/tmp2/martinlin/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg",  default="sam2.1_hiera_l.yaml")
    parser.add_argument("--device",     default="cuda:0")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python not installed. Run: pip install opencv-python")
        sys.exit(1)

    sys.path.insert(0, args.sam2_dir)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_path = args.model_cfg
    for candidate in [
        os.path.join(args.sam2_dir, "sam2", args.model_cfg),
        os.path.join(args.sam2_dir, "sam2", "configs", args.model_cfg),
    ]:
        if os.path.exists(candidate):
            cfg_path = candidate
            break

    sam2_model = build_sam2(cfg_path, args.checkpoint, device=args.device)
    predictor  = SAM2ImagePredictor(sam2_model)

    pattern   = os.path.join(args.data_dir, "original_time0_*.png")
    img_paths = sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1)))

    out_json = os.path.join(args.data_dir, "masks", "custom_prompts.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    all_prompts = {}
    current_obj_idx = 0
    points_per_obj  = {name: [] for name in OBJECT_NAMES}

    for img_path in img_paths:
        cam_id  = re.search(r"_(\d+)\.png$", img_path).group(1)
        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil)
        img_bgr = img_np[:, :, ::-1].copy()
        H, W    = img_np.shape[:2]
        points_per_obj = {name: [] for name in OBJECT_NAMES}
        current_obj_idx = 0
        current_mask = None

        def draw_canvas():
            canvas = img_bgr.copy()
            # draw all points
            for oi, oname in enumerate(OBJECT_NAMES):
                color = OBJECT_COLORS[oname]
                bgr_fg = (color[2], color[1], color[0])
                bgr_bg = (50, 50, 200)
                for (px, py, lb) in points_per_obj[oname]:
                    c = bgr_fg if lb == 1 else bgr_bg
                    cv2.circle(canvas, (int(px), int(py)), 8, c, -1)
                    cv2.circle(canvas, (int(px), int(py)), 8, (255,255,255), 2)
            # draw mask overlay if available
            if current_mask is not None:
                overlay = np.zeros_like(canvas)
                obj_name = OBJECT_NAMES[current_obj_idx]
                color = OBJECT_COLORS[obj_name]
                overlay[current_mask] = (color[2], color[1], color[0])
                canvas = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)
            # status text
            obj_name = OBJECT_NAMES[current_obj_idx]
            cv2.putText(canvas,
                f"CAM:{cam_id}  Object:[{obj_name}]  LClick=fg RClick=bg  Enter=predict  s=save  n/m=obj  q=next",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            return canvas

        def mouse_cb(event, x, y, flags, param):
            nonlocal current_mask
            obj_name = OBJECT_NAMES[current_obj_idx]
            if event == cv2.EVENT_LBUTTONDOWN:
                points_per_obj[obj_name].append((x, y, 1))
                current_mask = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                points_per_obj[obj_name].append((x, y, 0))
                current_mask = None
            cv2.imshow("SAM2 Tuner", draw_canvas())

        cv2.namedWindow("SAM2 Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SAM2 Tuner", min(W, 1280), min(H, 720))
        cv2.setMouseCallback("SAM2 Tuner", mouse_cb)
        predictor.set_image(img_np)

        while True:
            cv2.imshow("SAM2 Tuner", draw_canvas())
            key = cv2.waitKey(20) & 0xFF

            if key == ord('n'):
                current_obj_idx = (current_obj_idx + 1) % len(OBJECT_NAMES)
                current_mask = None
            elif key == ord('m'):
                current_obj_idx = (current_obj_idx - 1) % len(OBJECT_NAMES)
                current_mask = None
            elif key == 13:  # Enter
                obj_name = OBJECT_NAMES[current_obj_idx]
                pts = points_per_obj[obj_name]
                if pts:
                    coords = np.array([[p, q] for p, q, _ in pts], dtype=np.float32)
                    labels = np.array([lb for _, _, lb in pts], dtype=np.int32)
                    masks, scores, _ = predictor.predict(
                        point_coords=coords, point_labels=labels, multimask_output=True)
                    current_mask = masks[np.argmax(scores)]
                    print(f"  [{obj_name}] best score: {np.max(scores):.3f}")
            elif key == ord('s'):
                cam_prompts = {}
                for oname in OBJECT_NAMES:
                    if points_per_obj[oname]:
                        cam_prompts[oname] = [
                            (px / W, py / H, lb)
                            for px, py, lb in points_per_obj[oname]
                        ]
                all_prompts[cam_id] = cam_prompts
                with open(out_json, "w") as f:
                    json.dump(all_prompts, f, indent=2)
                print(f"  ✓ Saved prompts for cam {cam_id}")
            elif key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()
        if key == 27:
            break

    with open(out_json, "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"\n✅ All prompts saved to {out_json}")
    print("Run generate_masks_sam2.py with --prompts_json to apply them.")


if __name__ == "__main__":
    main()
