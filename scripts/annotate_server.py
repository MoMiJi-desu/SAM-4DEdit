#!/usr/bin/env python3
"""
annotate_server.py
==================
輕量標注 Web 伺服器，讓你在瀏覽器中對每張相機影像點選 SAM prompt 點。

使用方式：
  /tmp2/martinlin/miniconda3/envs/Gaussians4D/bin/python \\
      scripts/annotate_server.py \\
      --data_dir data/dynerf/time0_coffee_martini \\
      --port 8765

在瀏覽器打開：http://localhost:8765
  或若從遠端連 server：http://<server_ip>:8765

操作：
  左鍵點擊  → 新增前景點（綠色 ●）= 人體
  右鍵點擊  → 新增背景點（紅色 ●）= 背景
  [Undo]    → 刪除最後一個點
  [Clear]   → 清除此張影像所有點
  [Save & Next] → 儲存並跳下一張（所有張都存完後自動關閉）
  [Skip]    → 不儲存，跳下一張

完成後輸出：
  {data_dir}/masks/custom_prompts.json

接著執行：
  CUDA_VISIBLE_DEVICES=0 \\
  /tmp2/martinlin/miniconda3/envs/Gaussians4D/bin/python \\
      scripts/generate_masks_sam.py \\
      --prompts_json data/dynerf/time0_coffee_martini/masks/custom_prompts.json
"""

import os, sys, glob, re, json, base64, argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── 全域狀態 ──────────────────────────────────────────────────────────────────
state = {
    "img_paths": [],
    "cam_ids":   [],
    "current":   0,
    "prompts":   {},   # {cam_id: [[nx, ny, label], ...]}
    "out_json":  "",
    "done":      False,
}


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>SAM Point Annotator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }
  .header { background: #16213e; padding: 14px 24px; display: flex;
            justify-content: space-between; align-items: center;
            border-bottom: 2px solid #0f3460; }
  .header h1 { font-size: 18px; color: #e94560; }
  .status { font-size: 14px; color: #a0a0b0; }
  .legend { display: flex; gap: 16px; font-size: 13px; align-items: center; }
  .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
  .fg { background: #4caf50; }
  .bg { background: #f44336; }
  .canvas-wrap { position: relative; display: inline-block; margin: 20px auto;
                 display: block; text-align: center; }
  canvas { cursor: crosshair; border: 2px solid #0f3460; border-radius: 4px;
           max-width: 95vw; }
  .controls { text-align: center; padding: 12px; display: flex;
              gap: 12px; justify-content: center; }
  button { padding: 10px 22px; border: none; border-radius: 6px;
           font-size: 14px; cursor: pointer; font-weight: 600;
           transition: transform .1s, opacity .2s; }
  button:hover { transform: translateY(-1px); opacity: 0.9; }
  #btn-save { background: #4caf50; color: #fff; }
  #btn-skip { background: #555; color: #ccc; }
  #btn-undo { background: #ff9800; color: #fff; }
  #btn-clear { background: #f44336; color: #fff; }
  .info { text-align: center; font-size: 13px; color: #888;
          padding-bottom: 8px; }
  .done-banner { text-align: center; padding: 80px 20px; }
  .done-banner h2 { color: #4caf50; font-size: 28px; margin-bottom: 16px; }
  .done-banner p  { color: #aaa; font-size: 15px; }
  .pt-count { font-size: 13px; color: #7ecff4; margin-left: 8px; }
</style>
</head>
<body>
<div class="header">
  <h1>🎯 SAM Point Annotator</h1>
  <div class="status">Camera <b id="cam-label">__CAM__</b>
    &nbsp;|&nbsp; __IDX__ / __TOTAL__
    <span class="pt-count" id="pt-count">0 點</span>
  </div>
  <div class="legend">
    <span><span class="dot fg"></span> 左鍵 = 前景（人）</span>
    <span><span class="dot bg"></span> 右鍵 = 背景</span>
  </div>
</div>

<div class="info" style="padding-top:10px">
  點選人體各部位（臉、胸、腰、手臂），右鍵點選明確的背景區域</div>

<div class="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<div class="controls">
  <button id="btn-undo">↩ Undo</button>
  <button id="btn-clear">🗑 Clear</button>
  <button id="btn-skip">Skip →</button>
  <button id="btn-save">✅ Save & Next</button>
</div>

<script>
const CAM_ID  = "__CAM_ID__";
const IMG_W   = __IMG_W__;
const IMG_H   = __IMG_H__;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
let points   = [];   // [{x, y, label}]  x,y in pixel coords on original img

// ── 自適應縮放 ─────────────────────────────────────────────────────────────
const MAX_W  = Math.min(window.innerWidth * 0.95, 1200);
const SCALE  = Math.min(MAX_W / IMG_W, 1.0);
canvas.width  = Math.round(IMG_W * SCALE);
canvas.height = Math.round(IMG_H * SCALE);

const img = new Image();
img.src = "data:image/png;base64,__B64__";
img.onload = () => draw();

function draw() {
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  points.forEach(p => {
    const cx = p.x * SCALE;
    const cy = p.y * SCALE;
    ctx.beginPath();
    ctx.arc(cx, cy, 8, 0, Math.PI*2);
    ctx.fillStyle   = p.label === 1 ? 'rgba(76,175,80,0.85)' : 'rgba(244,67,54,0.85)';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
    // label number
    ctx.fillStyle   = '#fff';
    ctx.font        = 'bold 11px sans-serif';
    ctx.textAlign   = 'center';
    ctx.textBaseline= 'middle';
    ctx.fillText(p.label === 1 ? '●' : '×', cx, cy);
  });
  document.getElementById('pt-count').textContent =
    points.length + ' 點 (' +
    points.filter(p=>p.label===1).length + ' fg / ' +
    points.filter(p=>p.label===0).length + ' bg)';
}

canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => {
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  // pixel on original image
  const px = cx / SCALE;
  const py = cy / SCALE;
  const label = (e.button === 0) ? 1 : 0;
  points.push({x: px, y: py, label});
  draw();
});

document.getElementById('btn-undo').onclick = () => { points.pop(); draw(); };
document.getElementById('btn-clear').onclick = () => { points = []; draw(); };

document.getElementById('btn-save').onclick = () => {
  const norm = points.map(p => [p.x/IMG_W, p.y/IMG_H, p.label]);
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cam_id: CAM_ID, points: norm})
  }).then(r => r.json()).then(d => {
    if (d.done) {
      document.body.innerHTML = `<div class="done-banner">
        <h2>✅ 全部標注完成！</h2>
        <p>Prompts 已儲存至：<code>${d.json_path}</code></p>
        <br><p>現在執行：</p>
        <pre style="background:#111;padding:16px;border-radius:8px;
                    text-align:left;display:inline-block;color:#7ecff4">
CUDA_VISIBLE_DEVICES=0 \\
/tmp2/martinlin/miniconda3/envs/Gaussians4D/bin/python \\
    scripts/generate_masks_sam.py \\
    --prompts_json ${d.json_path}</pre></div>`;
    } else {
      window.location.reload();
    }
  });
};

document.getElementById('btn-skip').onclick = () => {
  fetch('/skip', {method:'POST'}).then(r=>r.json()).then(d => {
    if (d.done) { document.body.innerHTML='<div class="done-banner"><h2>Done</h2></div>'; }
    else { window.location.reload(); }
  });
};
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access logs

    def do_GET(self):
        if state["done"]:
            self.send_html("<h2 style='color:green;padding:40px'>All done! Check terminal.</h2>")
            return
        idx = state["current"]
        if idx >= len(state["img_paths"]):
            state["done"] = True
            self._save_json()
            self.send_html("<h2 style='padding:40px;color:green'>✅ Done! JSON saved.</h2>")
            return
        img_path = state["img_paths"][idx]
        cam_id   = state["cam_ids"][idx]
        from PIL import Image as PILImg
        img_pil  = PILImg.open(img_path).convert("RGB")
        W, H     = img_pil.size
        b64      = img_to_base64(img_path)
        html = HTML_TEMPLATE \
            .replace("__CAM__",   str(cam_id)) \
            .replace("__CAM_ID__",str(cam_id)) \
            .replace("__IDX__",   str(idx + 1)) \
            .replace("__TOTAL__", str(len(state["img_paths"]))) \
            .replace("__IMG_W__", str(W)) \
            .replace("__IMG_H__", str(H)) \
            .replace("__B64__",   b64)
        self.send_html(html)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        parsed  = urlparse(self.path)

        if parsed.path == "/save":
            data    = json.loads(body)
            cam_id  = data["cam_id"]
            pts     = data["points"]
            if pts:
                state["prompts"][cam_id] = pts
            state["current"] += 1
            self._save_json()
            done = state["current"] >= len(state["img_paths"])
            if done: state["done"] = True
            self.send_json({"done": done, "json_path": state["out_json"]})

        elif parsed.path == "/skip":
            state["current"] += 1
            done = state["current"] >= len(state["img_paths"])
            if done: state["done"] = True
            self.send_json({"done": done})

    def send_html(self, html):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _save_json(self):
        os.makedirs(os.path.dirname(state["out_json"]), exist_ok=True)
        with open(state["out_json"], "w") as f:
            json.dump(state["prompts"], f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="/tmp2/martinlin/Instruct-4DGS/data/dynerf/time0_coffee_martini")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    pattern   = os.path.join(args.data_dir, "original_time0_*.png")
    img_paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1))
    )
    if not img_paths:
        print(f"[ERROR] No images at {pattern}"); sys.exit(1)

    state["img_paths"] = img_paths
    state["cam_ids"]   = [re.search(r"_(\d+)\.png$", p).group(1) for p in img_paths]
    state["out_json"]  = os.path.join(args.data_dir, "masks", "custom_prompts.json")

    # Load existing prompts if any
    if os.path.exists(state["out_json"]):
        with open(state["out_json"]) as f:
            state["prompts"] = json.load(f)
        print(f"Loaded existing prompts for {len(state['prompts'])} cameras")

    print(f"\n{'='*55}")
    print(f"  SAM Point Annotator")
    print(f"  {len(img_paths)} images to annotate")
    print(f"{'='*55}")
    print(f"\n  Open in browser:  http://localhost:{args.port}")
    print(f"  (or via SSH tunnel: ssh -L {args.port}:localhost:{args.port} <user>@<server>)")
    print(f"\n  Left click  = foreground (person)")
    print(f"  Right click = background")
    print(f"  [Save & Next] to proceed")
    print(f"\n  Output: {state['out_json']}")
    print(f"{'='*55}\n")
    print("  Press Ctrl+C to stop\n")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nServer stopped. Prompts saved to: {state['out_json']}")


if __name__ == "__main__":
    main()
