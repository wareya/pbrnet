"""
infer_gui.py
────────────
Web GUI for PBRNet inference. Opens a browser interface where you can
drag-and-drop or select an image, pick output mode, run inference,
and view/download the results.

Usage
─────
    uv run python infer_gui.py
    uv run python infer_gui.py --model pbr_net.pt --port 7860
"""

import argparse
import base64
import io
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, send_file

# ── device ────────────────────────────────────────────────────────────────────

def pick_device(override):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using device: MPS")
        return torch.device("mps")
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Using device: CPU ({torch.get_num_threads()} threads)")
    return torch.device("cpu")

# ── network ───────────────────────────────────────────────────────────────────

INPUT_DIM = 420

class PBRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128),       nn.LeakyReLU(0.01),
            nn.Linear(128, 64),        nn.LeakyReLU(0.01),
            nn.Linear(64, 3),          nn.Sigmoid(),
            nn.Linear(3, 3),
        )
        nn.init.eye_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# ── image helpers ─────────────────────────────────────────────────────────────

SCALE_FACTORS = (1.0, 0.5, 0.25, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256)

def load_rgb_f32_from_bytes(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0

def downscale(img, factor):
    h, w = img.shape[:2]
    nw = max(1, int(round(w * factor)))
    nh = max(1, int(round(h * factor)))
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0

def make_scales(img):
    result = [img]
    for f in SCALE_FACTORS[1:]:
        result.append(downscale(result[-1], f))
        #result.append(downscale(img, f))
    return result

def pad_edge(img, pad):
    return np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode="edge")

# ── inference ─────────────────────────────────────────────────────────────────

def build_and_infer(model, device, batch_size, scales, metal_hint=0.5, progress_cb=None):
    color_full = scales[0]
    H, W = color_full.shape[:2]
    t0_ds = time.perf_counter()
    # (scales already built by caller — measure pad only)
    t_downscale_pre = 0.0  # placeholder, timed in infer_one

    t0_pad = time.perf_counter()
    p4 = pad_edge(color_full, 4)
    t_pad_pre = time.perf_counter() - t0_pad

    rows_per_strip = max(1, batch_size // W)
    preds_out = np.empty((H, W, 3), dtype=np.float32)
    t_feat = 0.0
    t_infer = 0.0

    FEAT_COLS = 420
    scale_info = []
    col = 246
    t0_bw = time.perf_counter()
    for scale_idx, img in enumerate(scales[1:]):
        Hi, Wi = img.shape[:2]
        if scale_idx == 0:   offsets = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
        elif scale_idx == 1: offsets = np.array([-1., 0., 1.],          dtype=np.float32)
        else:                offsets = np.array([-0.5, 0.5],             dtype=np.float32)
        n = len(offsets)
        width = n * n * 3

        # Row coords: (n, H) — needed per-strip
        fy_c = np.arange(H, dtype=np.float32) * Hi / H
        fy   = np.clip(fy_c[None, :] + offsets[:, None], 0.0, Hi - 1.0)
        y0   = np.floor(fy).astype(np.int32); y1 = np.minimum(y0 + 1, Hi - 1)
        wy   = (fy - y0).astype(np.float32)

        # Column interpolation precomputed once: col_interp shape (Hi, W, n, 3)
        # For each downscaled row and each full-res column, the n dx-offset samples
        fx_c = np.arange(W, dtype=np.float32) * Wi / W
        fx   = np.clip(fx_c[:, None] + offsets[None, :], 0.0, Wi - 1.0)  # (W, n)
        x0   = np.floor(fx).astype(np.int32); x1 = np.minimum(x0 + 1, Wi - 1)
        wx   = (fx - x0).astype(np.float32)  # (W, n)
        # img: (Hi, Wi, 3) → col_interp: (Hi, W, n, 3)
        col_interp = (img[:, x0, :] * (1 - wx)[None, :, :, None] +
                      img[:, x1, :] *      wx [None, :, :, None])

        scale_info.append((col_interp, Hi, n, width, y0, y1, wy, col))
        col += width
    t_bilweights_pre = time.perf_counter() - t0_bw

    # Preallocate the full output feature buffer — reused across strips
    X = np.empty((rows_per_strip * W, FEAT_COLS), dtype=np.float32)
    X[:, 0:3] = metal_hint  # hint never changes, set once

    t_downscale  = t_downscale_pre
    t_pad        = t_pad_pre
    t_bilweights = t_bilweights_pre
    t_9x9        = 0.0
    t_scale      = [0.0] * len(scale_info)

    with torch.no_grad():
        for row_start in range(0, H, rows_per_strip):
            row_end = min(row_start + rows_per_strip, H)
            n_rows  = row_end - row_start
            N       = n_rows * W
            Xv      = X[:N]

            t0 = time.perf_counter()
            strip_p4 = p4[row_start:row_end + 8]
            sp = strip_p4.strides
            Xv[:, 3:246] = np.lib.stride_tricks.as_strided(
                strip_p4,
                shape=(n_rows, W, 9, 9, 3),
                strides=(sp[0], sp[1], sp[0], sp[1], sp[2])
            ).reshape(N, 243)
            t_9x9 += time.perf_counter() - t0

            for si, (col_interp, Hi, n, width, y0, y1, wy, col_start) in enumerate(scale_info):
                t0 = time.perf_counter()
                sy0 = y0[:, row_start:row_end]  # (n, n_rows)
                sy1 = y1[:, row_start:row_end]
                swy = wy[:, row_start:row_end]
                out = Xv[:, col_start:col_start+width].reshape(n_rows, W, n, n, 3)
                for dyi in range(n):
                    r0  = sy0[dyi]               # (n_rows,)
                    r1  = sy1[dyi]
                    wyr = swy[dyi, :, None, None, None]  # (n_rows, 1, 1, 1)
                    # col_interp[r0]: (n_rows, W, n, 3) — contiguous row reads, no gather
                    out[:, :, dyi, :, :] = (col_interp[r0] * (1 - wyr) +
                                            col_interp[r1] *      wyr)
                t_scale[si] += time.perf_counter() - t0

            t0 = time.perf_counter()
            xb = torch.from_numpy(Xv).to(device)
            preds_out[row_start:row_end] = model(xb).clamp(0.0, 1.0).cpu().numpy().reshape(n_rows, W, 3)
            t_infer += time.perf_counter() - t0

            if progress_cb is not None and row_end < H:
                progress_cb(row_end, H, preds_out)

    scale_names = ["1/2(5x5)", "1/4(3x3)"] + [f"1/{8<<i}(2x2)" for i in range(6)]
    print(f"  downscale:   {t_downscale:.3f}s")
    print(f"  pad:         {t_pad:.3f}s")
    print(f"  bil.weights: {t_bilweights:.3f}s")
    print(f"  9x9:         {t_9x9:.3f}s")
    for name, t in zip(scale_names, t_scale):
        print(f"  {name}: {t:.3f}s")
    print(f"  infer:       {t_infer:.3f}s")
    t_feat = t_downscale + t_pad + t_bilweights + t_9x9 + sum(t_scale)
    return preds_out, t_feat, t_infer

def arr_to_png_b64(arr_uint8):
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
_model  = None
_device = None
_batch  = 65536

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PBRNet Inference</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: sans-serif; background: #1a1a1a; color: #ddd; padding: 24px; }
  h1 { font-size: 1.2em; margin-bottom: 16px; color: #fff; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { background: #242424; border-radius: 8px; padding: 16px; }
  .drop { border: 2px dashed #555; border-radius: 6px; padding: 32px; text-align: center;
          cursor: pointer; transition: border-color .2s; min-width: 260px; }
  .drop:hover, .drop.over { border-color: #88aaff; }
  .drop img { max-width: 100%; max-height: 260px; display: block; margin: 8px auto 0; }
  select, button { background: #333; color: #ddd; border: 1px solid #555;
                   border-radius: 4px; padding: 6px 12px; font-size: 0.95em; }
  button { cursor: pointer; background: #2a4a8a; border-color: #4466cc; color: #fff;
           padding: 8px 20px; margin-top: 8px; }
  button:disabled { opacity: 0.5; cursor: default; }
  button:hover:not(:disabled) { background: #3a5aaa; }
  .results { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 16px; }
  .result { text-align: center; }
  .result img { display: block; max-width: 256px; max-height: 256px;
                border-radius: 4px; border: 1px solid #444; }
  .result a { display: block; font-size: 0.8em; color: #88aaff; margin-top: 4px; }
  .results.stale .result img { filter: blur(3px) brightness(0.6); transition: filter 0.2s; }
  .results.stale .result a { opacity: 0.4; }
  .status { margin-top: 10px; font-size: 0.9em; color: #aaa; min-height: 1.2em; }
  label { font-size: 0.9em; color: #aaa; }
  .controls { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
</style>
</head>
<body>
<h1>PBRNet Inference</h1>
<div class="row">
  <div class="panel">
    <div class="drop" id="drop">
      <div id="drop-label">Drop image here or click to select</div>
      <img id="preview" style="display:none">
      <input type="file" id="file-input" accept="*/*" style="display:none">
    </div>
    <div class="controls">
      <div>
        <label>Output</label>
        <select id="mode">
          <option value="orm">ORM (R=AO G=Rough B=Metal)</option>
          <option value="mro">MRO (R=Metal G=Rough B=AO)</option>
          <option value="separate">Separate channels</option>
        </select>
      </div>
      <div>
        <label>Metal hint</label>
        <select id="metal-hint">
          <option value="0.0">Non-metallic (0.0)</option>
          <option value="0.5" selected>Unknown (0.5)</option>
          <option value="1.0">Metallic (1.0)</option>
        </select>
      </div>
      <div>
        <button id="run-btn" disabled>Run Inference</button>
      </div>
    </div>
    <div class="status" id="status"></div>
  </div>

  <div class="panel" id="results-panel" style="display:none">
    <div class="results" id="results"></div>
    <div style="font-size:0.85em;color:#888;margin-top:8px" id="timing"></div>
  </div>
</div>

<script>
let fileData = null;
let fileName = null;

const drop = document.getElementById('drop');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const runBtn = document.getElementById('run-btn');
const status = document.getElementById('status');
const resultsPanel = document.getElementById('results-panel');
const results = document.getElementById('results');
const timing = document.getElementById('timing');

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  fileName = file.name;
  const reader = new FileReader();
  reader.onload = async e => {
    fileData = e.target.result;
    // Send to server for decode+re-encode so any format (DDS etc.) displays correctly
    status.textContent = 'Decoding image...';
    try {
      const resp = await fetch('/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: fileData }),
      });
      const data = await resp.json();
      if (data.error) {
        status.textContent = 'Preview failed: ' + data.error + ' (inference may still work)';
      } else {
        preview.src = 'data:image/png;base64,' + data.png;
        preview.style.display = 'block';
      }
    } catch(e) {
      status.textContent = 'Preview unavailable - inference may still work.';
    }
    document.getElementById('drop-label').style.display = 'none';
    runBtn.disabled = false;
    _hasResults = false;
    results.classList.add('stale');
    status.textContent = 'Ready - press Run Inference to process this image.';
  };
  reader.readAsDataURL(file);
}

let _hasResults = false;

function applyImages(data) {
  if (!data.images) return;
  const base = fileName ? fileName.replace(/[.][^.]+$/, '') : 'output';
  const incoming = Object.keys(data.images);
  // Remove result divs whose labels are no longer in the current output set
  results.querySelectorAll('.result').forEach(d => {
    if (!incoming.includes(d.dataset.label)) d.remove();
  });
  // Update or create result divs
  const existing = {};
  results.querySelectorAll('.result').forEach(d => { existing[d.dataset.label] = d; });
  for (const [label, b64] of Object.entries(data.images)) {
    const src = 'data:image/png;base64,' + b64;
    if (existing[label]) {
      existing[label].querySelector('img').src = src;
      existing[label].querySelector('a').href = src;
      existing[label].querySelector('a').download = base + '_' + label + '.png';
    } else {
      const div = document.createElement('div');
      div.className = 'result';
      div.dataset.label = label;
      div.innerHTML = `<img src="${src}"><a href="${src}" download="${base}_${label}.png">v ${label}</a>`;
      results.appendChild(div);
    }
  }
  results.classList.remove('stale');
  resultsPanel.style.display = 'block';
}

document.getElementById('mode').addEventListener('change', async () => {
  if (!_hasResults) return;
  const mode = document.getElementById('mode').value;
  try {
    const resp = await fetch('/repack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    const data = await resp.json();
    if (data.error) { status.textContent = 'Repack error: ' + data.error; return; }
    applyImages(data);
  } catch(e) {
    status.textContent = 'Repack failed: ' + e;
  }
});

runBtn.addEventListener('click', async () => {
  if (!fileData) return;
  runBtn.disabled = true;
  status.textContent = 'Running inference...';
  results.classList.add('stale');

  const mode = document.getElementById('mode').value;
  try {
    const resp = await fetch('/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: fileData, mode, metal_hint: parseFloat(document.getElementById('metal-hint').value) }),
    });
    if (!resp.ok) { throw new Error(`HTTP ${resp.status}`); }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    function processEvents(chunk) {
      buf += chunk;
      const events = buf.split('data: ');
      buf = events.pop();
      for (const event of events) {
        const line = event.trimEnd();
        if (!line) continue;
        let data;
        try { data = JSON.parse(line); } catch { continue; }
        if (data.error) {
          status.textContent = 'Error: ' + data.error;
          results.classList.remove('stale');
          runBtn.disabled = false;
          return;
        }
        applyImages(data);
        if (data.done) {
          timing.textContent = 'Features: ' + data.t_feat.toFixed(2) + 's   Inference: ' + data.t_infer.toFixed(2) + 's   Total: ' + (data.t_feat + data.t_infer).toFixed(2) + 's';
          status.textContent = 'Done.';
          _hasResults = true;
        } else {
          status.textContent = 'Processing... ' + data.progress + '%';
        }
      }
    }
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        // Flush anything remaining in buf by forcing a split
        buf += 'data: ';
        processEvents('');
        break;
      }
      processEvents(decoder.decode(value, { stream: true }));
    }
  } catch(e) {
    status.textContent = 'Request failed: ' + e;
    results.classList.remove('stale');
  }
  runBtn.disabled = false;
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    from flask import Response
    return Response(HTML, mimetype="text/html")

@app.route("/preview", methods=["POST"])
def preview_route():
    try:
        payload   = request.get_json()
        b64       = payload["image"].split(",", 1)[-1]
        img_bytes = base64.b64decode(b64)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return jsonify({"png": arr_to_png_b64(np.array(img, dtype=np.uint8))})
    except Exception as e:
        return jsonify({"error": str(e)})


_last_preds = None

def pack_images(preds, mode):
    R = (preds[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    A = (preds[:, :, 1] * 255).clip(0, 255).astype(np.uint8)
    M = (preds[:, :, 2] * 255).clip(0, 255).astype(np.uint8)
    if mode == "orm":
        return {"ORM": arr_to_png_b64(np.stack([A, R, M], axis=2))}
    elif mode == "mro":
        return {"MRO": arr_to_png_b64(np.stack([M, R, A], axis=2))}
    else:
        return {"Roughness":        arr_to_png_b64(R),
                "AmbientOcclusion": arr_to_png_b64(A),
                "Metalness":        arr_to_png_b64(M)}


@app.route("/infer", methods=["POST"])
def infer_route():
    import json as _json
    try:
        payload    = request.get_json()
        mode       = payload.get("mode", "orm")
        metal_hint = float(payload.get("metal_hint", 0.5))
        b64        = payload["image"].split(",", 1)[-1]
        img_bytes  = base64.b64decode(b64)

        color_full = load_rgb_f32_from_bytes(img_bytes)
        scales     = make_scales(color_full)
        H          = color_full.shape[0]

        def generate():
            last_pct = [0]

            # build_and_infer is synchronous; wrap generator to drive it
            # We collect yielded progress events via a queue
            import queue, threading
            q = queue.Queue()

            def cb(row_done, H, preds_out):
                pct = int(row_done / H * 100)
                if pct - last_pct[0] >= 5:
                    last_pct[0] = pct
                    imgs = pack_images(preds_out.copy(), mode)
                    q.put(("progress", pct, imgs, None, None))

            result = [None]
            def run():
                try:
                    global _last_preds
                    preds, t_feat, t_infer = build_and_infer(
                        _model, _device, _batch, scales, metal_hint, progress_cb=cb)
                    _last_preds = preds
                    result[0] = (preds, t_feat, t_infer)
                except Exception as e:
                    result[0] = e
                finally:
                    q.put(("done", None, None, None, None))

            t = threading.Thread(target=run, daemon=True)
            t.start()

            while True:
                item = q.get()
                if item[0] == "done":
                    break
                _, pct, imgs, _, _ = item
                yield "data: " + _json.dumps({"progress": pct, "images": imgs}) + "\n\n"

            t.join()
            if isinstance(result[0], Exception):
                yield "data: " + _json.dumps({"error": str(result[0])}) + "\n\n"
            else:
                preds, t_feat, t_infer = result[0]
                imgs = pack_images(preds, mode)
                yield "data: " + _json.dumps({"progress": 100, "done": True, "images": imgs, "t_feat": t_feat, "t_infer": t_infer}) + "\n\n"

        from flask import Response, stream_with_context
        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as e:
        import traceback; traceback.print_exc()
        import json as _json
        return _json.dumps({"error": str(e)}), 500

@app.route("/repack", methods=["POST"])
def repack_route():
    import json as _json
    try:
        payload = request.get_json()
        mode = payload.get("mode", "orm")
        if _last_preds is None:
            return jsonify({"error": "no inference result cached"})
        images = pack_images(_last_preds, mode)
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)})


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="pbr_net.pt")
    ap.add_argument("--batch",  type=int, default=65536)
    ap.add_argument("--device", default=None)
    ap.add_argument("--port",   type=int, default=7860)
    ap.add_argument("--host",   default="127.0.0.1")
    args = ap.parse_args()

    _device = pick_device(args.device)
    _model  = PBRNet().to(_device)
    _model.load_state_dict(torch.load(args.model, map_location=_device))
    _model.eval()
    _batch  = args.batch
    print(f"Loaded model: {args.model}")

    import webbrowser
    url = f"http://{args.host}:{args.port}"
    print(f"Opening {url}")
    webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
