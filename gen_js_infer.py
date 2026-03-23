"""
gen_js_infer.py
───────────────
Generates a fully self-contained HTML file that does PBRNet inference
entirely in the browser, on the CPU, with no backend server.

Weights are embedded as base64-encoded float32 binary blobs.
The JS replicates the full inference pipeline:
  - Multi-scale image pyramid via Canvas
  - Bilinear patch feature extraction matching build_dataset.py exactly
  - MLP forward pass (LeakyReLU, Sigmoid, Linear layers)

Usage
─────
    uv run python gen_js_infer.py --model pbr_net.pt --out pbr_infer.html
"""

import argparse
import base64
import os
import struct

import numpy as np
import torch
import torch.nn as nn

# ── network (must match train.py) ─────────────────────────────────────────────

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


def extract_layers(model):
    layers = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32)
            layers.append((w, b))
    return layers


def f32_to_b64(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def generate(model_path, out_path):
    model = PBRNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    layers = extract_layers(model)

    # Encode each layer's weights and biases as base64 float32
    layer_data = []
    for w, b in layers:
        layer_data.append({
            "rows": int(w.shape[0]),
            "cols": int(w.shape[1]),
            "w": f32_to_b64(w),
            "b": f32_to_b64(b),
        })

    import json
    layers_json = json.dumps(layer_data)

    html = build_html(layers_json)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path}  ({size_kb:.0f} KB)")


def build_html(layers_json):
    # The JS is written as a regular Python string. No escape issues because
    # we use a different delimiter and avoid backslash sequences in string literals.
    # All JS string splitting uses indexOf/substring rather than regex or \n.
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PBRNet JS Inference</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: monospace; background: #111; color: #ccc; padding: 20px; }
h1 { font-size: 1em; color: #7af; margin-bottom: 14px; letter-spacing: 0.1em; }
.row { display: flex; gap: 16px; flex-wrap: wrap; }
.panel { background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 14px; }
.drop { border: 2px dashed #444; border-radius: 4px; padding: 24px 32px; text-align: center;
        cursor: pointer; transition: border-color .15s; min-width: 220px; }
.drop:hover, .drop.over { border-color: #7af; }
.drop img { max-width: 100%; max-height: 220px; display: block; margin: 8px auto 0; border-radius: 2px; }
select, button { background: #222; color: #ccc; border: 1px solid #444;
                 border-radius: 3px; padding: 5px 10px; font-family: monospace; font-size: 0.9em; }
button { cursor: pointer; color: #7af; border-color: #7af; margin-top: 6px; }
button:disabled { opacity: 0.4; cursor: default; }
button:hover:not(:disabled) { background: #1a2a3a; }
.controls { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
label { font-size: 0.8em; color: #888; }
.status { margin-top: 8px; font-size: 0.8em; color: #888; min-height: 1.1em; }
.progress-bar { height: 3px; background: #222; border-radius: 2px; margin-top: 6px; overflow: hidden; }
.progress-fill { height: 100%; background: #7af; width: 0%; }
.results { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.result { text-align: center; }
.result canvas { display: block; border-radius: 3px; border: 1px solid #333;
                 max-width: 256px; max-height: 256px; }
.result a { display: block; font-size: 0.75em; color: #7af; margin-top: 3px; }
.results.stale .result canvas { opacity: 0.35; transition: opacity 0.2s; }
.timing { font-size: 0.75em; color: #666; margin-top: 8px; }
</style>
</head>
<body>
<h1>PBRNet / client-side inference</h1>
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
        <label><input type="checkbox" id="use-wasm" checked> Use WASM</label>
        <button id="run-btn" disabled>Run Inference</button>
        <button id="cancel-btn" disabled>Cancel</button>
      </div>
    </div>
    <div class="status" id="status">Loading weights...</div>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  </div>
  <div class="panel" id="results-panel" style="display:none">
    <div class="results" id="results"></div>
    <div class="timing" id="timing"></div>
  </div>
</div>

<script>
// ── Weights (base64 float32) ──────────────────────────────────────────────────
const LAYERS_JSON = '""" + layers_json.replace("'", "\\'") + """';

// ── Decode weights ────────────────────────────────────────────────────────────
function b64ToF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

const LAYER_SPECS = JSON.parse(LAYERS_JSON);
const LAYERS = LAYER_SPECS.map(s => ({
  rows: s.rows, cols: s.cols,
  w: b64ToF32(s.w),
  b: b64ToF32(s.b),
}));

// ── WASM SIMD MLP ─────────────────────────────────────────────────────────────
const WASM_B64 = 'AGFzbQEAAAABEQRgAXwBfGAAAGAAAX1gAAF/AhMBA2VudgtzaWdtb2lkX2Y2NAAAAxQTAQEBAQECAgIDAwMDAwMDAwMDAwUDAQAKB+gBFAZtZW1vcnkCAAZsYXllcjAAAQZsYXllcjEAAgZsYXllcjIAAwZsYXllcjMABAZsYXllcjQABQhnZXRfaDRfMAAGCGdldF9oNF8xAAcIZ2V0X2g0XzIACAtmZWF0X29mZnNldAAJCkwwd19vZmZzZXQACgpMMGJfb2Zmc2V0AAsKTDF3X29mZnNldAAMCkwxYl9vZmZzZXQADQpMMndfb2Zmc2V0AA4KTDJiX29mZnNldAAPCkwzd19vZmZzZXQAEApMM2Jfb2Zmc2V0ABEKTDR3X29mZnNldAASCkw0Yl9vZmZzZXQAEwr4BxPBAQMDfwF7AX1BACEAAkADQCAAQYACTw0BQQAgAEGQDWxqIQJDAAAAAP0TIQNBACEBAkADQCABQekATw0BIAMgAiABQRBsav0ABABB0LQkIAFBEGxq/QAEAP3mAf3kASEDIAFBAWohAQwACwsgA/0fACAD/R8BkiAD/R8CIAP9HwOSkiEEIARBgKAaIABBBGxqKgIAkiEEQeDBJCAAQQRsaiAEIARDCtcjPJQgBEMAAAAAYBs4AgAgAEEBaiEADAALCwvDAQMDfwF7AX1BACEAAkADQCAAQYABTw0BQYCoGiAAQYAIbGohAkMAAAAA/RMhA0EAIQECQANAIAFBwABPDQEgAyACIAFBEGxq/QAEAEHgwSQgAUEQbGr9AAQA/eYB/eQBIQMgAUEBaiEBDAALCyAD/R8AIAP9HwGSIAP9HwIgA/0fA5KSIQQgBEGAqCIgAEEEbGoqAgCSIQRB4MkkIABBBGxqIAQgBEMK1yM8lCAEQwAAAABgGzgCACAAQQFqIQAMAAsLC8IBAwN/AXsBfUEAIQACQANAIABBwABPDQFBgKwiIABBgARsaiECQwAAAAD9EyEDQQAhAQJAA0AgAUEgTw0BIAMgAiABQRBsav0ABABB4MkkIAFBEGxq/QAEAP3mAf3kASEDIAFBAWohAQwACwsgA/0fACAD/R8BkiAD/R8CIAP9HwOSkiEEIARBgKwkIABBBGxqKgIAkiEEQeDNJCAAQQRsaiAEIARDCtcjPJQgBEMAAAAAYBs4AgAgAEEBaiEADAALCwu0AQMDfwF7AX1BACEAAkADQCAAQQNPDQFBgK4kIABBgAJsaiECQwAAAAD9EyEDQQAhAQJAA0AgAUEQTw0BIAMgAiABQRBsav0ABABB4M0kIAFBEGxq/QAEAP3mAf3kASEDIAFBAWohAQwACwsgA/0fACAD/R8BkiAD/R8CIAP9HwOSkiEEIARBgLQkIABBBGxqKgIAkiEEQeDPJCAAQQRsaiAEuxAAtjgCACAAQQFqIQAMAAsLC4oBAgJ/AX1BACEAAkADQCAAQQNPDQFBwLQkIABBBGxqKgIAIQJBACEBAkADQCABQQNPDQEgAkGQtCQgAEEDbCABakEEbGoqAgBB4M8kIAFBBGxqKgIAlJIhAiABQQFqIQEMAAsLQfDPJCAAQQRsakMAAIA/QwAAAAAgApeWOAIAIABBAWohAAwACwsLCQBB8M8kKgIACwkAQfTPJCoCAAsJAEH4zyQqAgALBgBB0LQkCwQAQQALBgBBgKAaCwYAQYCoGgsGAEGAqCILBgBBgKwiCwYAQYCsJAsGAEGAriQLBgBBgLQkCwYAQZC0JAsGAEHAtCQL';

let wasmMlp = null;  // set after init
let wasmMem = null;
let wasmFeat = null; // Float32Array view into wasm memory at feat offset

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function initWasm() {
  const bytes = b64ToBytes(WASM_B64);
  const result = await WebAssembly.instantiate(bytes, {
    env: {
      sigmoid_f64: (x) => 1.0 / (1.0 + Math.exp(-x)),
    }
  });
  const exp = result.instance.exports;
  wasmMem = new Float32Array(exp.memory.buffer);
  // Copy weights and biases into WASM memory
  const layerOffsets = [
    [exp.L0w_offset(), exp.L0b_offset()],
    [exp.L1w_offset(), exp.L1b_offset()],
    [exp.L2w_offset(), exp.L2b_offset()],
    [exp.L3w_offset(), exp.L3b_offset()],
    [exp.L4w_offset(), exp.L4b_offset()],
  ];
  for (let i = 0; i < 5; i++) {
    const [wo, bo] = layerOffsets[i];
    wasmMem.set(LAYERS[i].w, wo / 4);
    wasmMem.set(LAYERS[i].b, bo / 4);
  }
  wasmFeat = wasmMem.subarray(exp.feat_offset() / 4, exp.feat_offset() / 4 + 420);
  wasmMlp = {
    layer0: exp.layer0, layer1: exp.layer1, layer2: exp.layer2,
    layer3: exp.layer3, layer4: exp.layer4,
    get_h4_0: exp.get_h4_0, get_h4_1: exp.get_h4_1, get_h4_2: exp.get_h4_2,
  };
  statusEl.textContent = 'Weights loaded (WASM). Select an image to begin.';
}

initWasm().then(() => {}).catch(e => {
  console.warn('WASM init failed, falling back to JS:', e);
  statusEl.textContent = 'Weights loaded (JS fallback). Select an image to begin.';
});

// ── MLP forward (single pixel, output written into out[0..2]) ─────────────────
const _h0 = new Float32Array(256);
const _h1 = new Float32Array(128);
const _h2 = new Float32Array(64);
const _h3 = new Float32Array(3);
const _h4 = new Float32Array(3);

function mlp(feat, out) {
  // Layer 0: 420->256 LeakyReLU
  const L0 = LAYERS[0]; const w0=L0.w, b0=L0.b;
  for (let i = 0; i < 256; i++) {
    let v = b0[i]; const base = i * 420;
    for (let j = 0; j < 420; j++) v += w0[base + j] * feat[j];
    _h0[i] = v > 0 ? v : 0.01 * v;
  }
  // Layer 1: 256->128 LeakyReLU
  const L1 = LAYERS[1]; const w1=L1.w, b1=L1.b;
  for (let i = 0; i < 128; i++) {
    let v = b1[i]; const base = i * 256;
    for (let j = 0; j < 256; j++) v += w1[base + j] * _h0[j];
    _h1[i] = v > 0 ? v : 0.01 * v;
  }
  // Layer 2: 128->64 LeakyReLU
  const L2 = LAYERS[2]; const w2=L2.w, b2=L2.b;
  for (let i = 0; i < 64; i++) {
    let v = b2[i]; const base = i * 128;
    for (let j = 0; j < 128; j++) v += w2[base + j] * _h1[j];
    _h2[i] = v > 0 ? v : 0.01 * v;
  }
  // Layer 3: 64->3 Sigmoid
  const L3 = LAYERS[3]; const w3=L3.w, b3=L3.b;
  for (let i = 0; i < 3; i++) {
    let v = b3[i]; const base = i * 64;
    for (let j = 0; j < 64; j++) v += w3[base + j] * _h2[j];
    _h3[i] = 1.0 / (1.0 + Math.exp(-v));
  }
  // Layer 4: 3->3 linear
  const L4 = LAYERS[4]; const w4=L4.w, b4=L4.b;
  for (let i = 0; i < 3; i++) {
    let v = b4[i]; const base = i * 3;
    for (let j = 0; j < 3; j++) v += w4[base + j] * _h3[j];
    out[i] = Math.max(0, Math.min(1, v));
  }
}

// ── Image pyramid ─────────────────────────────────────────────────────────────
const SCALE_FACTORS = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625];

function buildScales(imgEl) {
  const scales = [];
  
  // Initial source data from the img element
  const startCanvas = document.createElement('canvas');
  startCanvas.width = imgEl.naturalWidth;
  startCanvas.height = imgEl.naturalHeight;
  const startCtx = startCanvas.getContext('2d');
  startCtx.drawImage(imgEl, 0, 0);
  
  let currentData = startCtx.getImageData(0, 0, startCanvas.width, startCanvas.height);
  scales.push({ 
    w: startCanvas.width, 
    h: startCanvas.height, 
    data: currentData.data 
  });
  
  // We skip the first factor if it's 1.0, as Python does result = [img]
  for (let s = 1; s < SCALE_FACTORS.length; s++) {
    const factor = SCALE_FACTORS[s] / SCALE_FACTORS[s-1]; // Relative factor for recursive scaling
    const prev = scales[scales.length - 1];
    
    const nw = Math.max(1, Math.round(prev.w * factor));
    const nh = Math.max(1, Math.round(prev.h * factor));
    
    const newData = manualBilinear(prev.data, prev.w, prev.h, nw, nh);
    
    // Create debug canvas
	/*
    const c = document.createElement('canvas');
    c.width = nw;
    c.height = nh;
    const ctx = c.getContext('2d');
    const imgDataObj = new ImageData(newData, nw, nh);
    ctx.putImageData(imgDataObj, 0, 0);
    document.body.appendChild(c);
	*/
    
    const scaleObj = { w: nw, h: nh, data: newData };
    scales.push(scaleObj);
  }
  
  return scales;
}

function manualBilinear(srcData, srcW, srcH, dstW, dstH) {
  const dstData = new Uint8ClampedArray(dstW * dstH * 4);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      // Pillow-style pixel centering
      const srcX = (x + 0.5) * scaleX - 0.5;
      const srcY = (y + 0.5) * scaleY - 0.5;

      let x0 = Math.floor(srcX);
      let y0 = Math.floor(srcY);
      let x1 = Math.min(x0 + 1, srcW - 1);
      let y1 = Math.min(y0 + 1, srcH - 1);
      
      x0 = Math.max(0, x0);
      y0 = Math.max(0, y0);

      const dx = srcX - x0;
      const dy = srcY - y0;

      for (let c = 0; c < 4; c++) { // RGBA channels
        const p00 = srcData[(y0 * srcW + x0) * 4 + c];
        const p10 = srcData[(y0 * srcW + x1) * 4 + c];
        const p01 = srcData[(y1 * srcW + x0) * 4 + c];
        const p11 = srcData[(y1 * srcW + x1) * 4 + c];

        // Bilinear interpolation formula
        const val = (p00 * (1 - dx) * (1 - dy)) +
                    (p10 * dx * (1 - dy)) +
                    (p01 * (1 - dx) * dy) +
                    (p11 * dx * dy);

        dstData[(y * dstW + x) * 4 + c] = val;
      }
    }
  }
  return dstData;
}

// ── Bilinear sample from ImageData (returns [r,g,b] in 0..1) ─────────────────
function bilerp(scaleData, scaleW, scaleH, fy, fx) {
  fx = Math.max(0, Math.min(scaleW - 1, fx));
  fy = Math.max(0, Math.min(scaleH - 1, fy));
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, scaleW - 1);
  const y1 = Math.min(y0 + 1, scaleH - 1);
  const wx = fx - x0, wy = fy - y0;
  const d = scaleData;
  const i00 = (y0 * scaleW + x0) * 4;
  const i01 = (y0 * scaleW + x1) * 4;
  const i10 = (y1 * scaleW + x0) * 4;
  const i11 = (y1 * scaleW + x1) * 4;
  const r = (d[i00]*(1-wy)*(1-wx) + d[i01]*(1-wy)*wx + d[i10]*wy*(1-wx) + d[i11]*wy*wx) / 255;
  const g = (d[i00+1]*(1-wy)*(1-wx) + d[i01+1]*(1-wy)*wx + d[i10+1]*wy*(1-wx) + d[i11+1]*wy*wx) / 255;
  const b = (d[i00+2]*(1-wy)*(1-wx) + d[i01+2]*(1-wy)*wx + d[i10+2]*wy*(1-wx) + d[i11+2]*wy*wx) / 255;
  return [r, g, b];
}

// ── Build 420-element feature vector for one pixel ────────────────────────────
const _feat = new Float32Array(420);

function buildFeat(scales, px, py, metalHint) {
  const s0 = scales[0];
  const W = s0.w, H = s0.h;
  let fi = 0;

  // hint (3 copies)
  _feat[fi++] = metalHint; _feat[fi++] = metalHint; _feat[fi++] = metalHint;

  // 9x9 full-res patch (integer pixel, edge-clamped)
  for (let dy = -4; dy <= 4; dy++) {
    for (let dx = -4; dx <= 4; dx++) {
      const sx = Math.max(0, Math.min(W-1, px+dx));
      const sy = Math.max(0, Math.min(H-1, py+dy));
      const idx = (sy * W + sx) * 4;
      _feat[fi++] = s0.data[idx]   / 255;
      _feat[fi++] = s0.data[idx+1] / 255;
      _feat[fi++] = s0.data[idx+2] / 255;
    }
  }

  // 5x5 @ 1/2, stride 1.0 downscaled px
  {
    const sc = scales[1];
    const fxc = px * sc.w / W, fyc = py * sc.h / H;
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const rgb = bilerp(sc.data, sc.w, sc.h, fyc+dy, fxc+dx);
        _feat[fi++] = rgb[0]; _feat[fi++] = rgb[1]; _feat[fi++] = rgb[2];
      }
    }
  }

  // 3x3 @ 1/4, stride 1.0 downscaled px
  {
    const sc = scales[2];
    const fxc = px * sc.w / W, fyc = py * sc.h / H;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const rgb = bilerp(sc.data, sc.w, sc.h, fyc+dy, fxc+dx);
        _feat[fi++] = rgb[0]; _feat[fi++] = rgb[1]; _feat[fi++] = rgb[2];
      }
    }
  }

  // 2x2 @ 1/8 through 1/256, offsets +-0.5
  for (let si = 3; si <= 8; si++) {
    const sc = scales[si];
    const fxc = px * sc.w / W, fyc = py * sc.h / H;
    for (let dy = -1; dy <= 1; dy += 2) {
      for (let dx = -1; dx <= 1; dx += 2) {
        const rgb = bilerp(sc.data, sc.w, sc.h, fyc+dy*0.5, fxc+dx*0.5);
        _feat[fi++] = rgb[0]; _feat[fi++] = rgb[1]; _feat[fi++] = rgb[2];
      }
    }
  }
}

// ── Main inference loop ───────────────────────────────────────────────────────
let currentImg = null;
let running = false;

const drop        = document.getElementById('drop');
const fileInput   = document.getElementById('file-input');
const preview     = document.getElementById('preview');
const runBtn      = document.getElementById('run-btn');
const cancelBtn   = document.getElementById('cancel-btn');
let cancelRequested = false;
const statusEl    = document.getElementById('status');
const progressFill= document.getElementById('progress-fill');
const resultsPanel= document.getElementById('results-panel');
const resultsEl   = document.getElementById('results');
const timingEl    = document.getElementById('timing');
let fileName      = null;

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function delete_color_correction_metadata(data)
{
	let gamma_chunk = [0x00, 0x00, 0x00, 0x04, 0x67, 0x41, 0x4D, 0x41];
	let srgb_chunk = [0x00, 0x00, 0x00, 0x01, 0x73, 0x52, 0x47, 0x42];
	//let srgb_chunk  = new Array([0x00, 0x00, 0x00, 0x04, 0x67, 0x41, 0x4D, 0x41]);
	function find(array)
	{
		return data.findIndex((_, i, data) =>
		{
			function equal(buf1, buf2)
			{
				if (buf1.length != buf2.length)
					return false;
				for (let i = 0 ; i != buf1.length ; i++)
				{
					if (buf1[i] != buf2[i])
						return false;
				}
				return true;
			}
			let asdf = equal(data.slice(i, i+8), array); // gAMA chunk
			return asdf;
		});
	};
	let index = find(gamma_chunk);
	if(index >= 0)
		data.splice(index, 16);
	index = find(srgb_chunk);
	if(index >= 0)
		data.splice(index, 13);
	return data;
}

async function loadFile(file) {
    fileName = file.name;
    let fileToLoad = file;

    // Check if the file is a PNG image
    if (file.type === 'image/png') {
        try {
            // Read file as an ArrayBuffer
            const arrayBuffer = await file.arrayBuffer();
            // Convert to a regular Array or Uint8Array for your logic
            let data = new Uint8Array(arrayBuffer);
            
            // Run your metadata deletion (converting back to Uint8Array if needed)
            // Note: splice works on standard arrays; if data is Uint8Array, 
            // you may need to convert it or use .slice() to reconstruct it.
            let processedData = delete_color_correction_metadata(Array.from(data));
            
            // Create a new Blob from the processed data
            fileToLoad = new Blob([new Uint8Array(processedData)], { type: 'image/png' });
        } catch (e) {
            console.error("Failed to strip PNG metadata:", e);
        }
    }

    const url = URL.createObjectURL(fileToLoad);
    
    preview.onload = () => {
        currentImg = preview;
        preview.style.display = 'block';
        document.getElementById('drop-label').style.display = 'none';
        runBtn.disabled = false;
        resultsEl.classList.add('stale');
        statusEl.textContent = 'Ready.';
        progressFill.style.width = '0%';
    };

    preview.onerror = () => {
        statusEl.textContent = 'Preview unavailable (unsupported format).';
        runBtn.disabled = true;
    };

    preview.src = url;
}

function setResult(label, canvas, final) {
  const base = fileName ? fileName.replace(/[.][^.]+$/, '') : 'output';
  const existing = resultsEl.querySelector('[data-label="' + label + '"]');
  if (existing) {
    const oldCanvas = existing.querySelector('canvas');
    const ctx = oldCanvas.getContext('2d');
    oldCanvas.replaceWith(canvas);
    if (final) {
      existing.querySelector('a').href = canvas.toDataURL('image/png');
      existing.querySelector('a').download = base + '_' + label + '.png';
    }
  } else {
    const div = document.createElement('div');
    div.className = 'result';
    div.dataset.label = label;
    const a = document.createElement('a');
    a.textContent = 'v ' + label;
    a.download = base + '_' + label + '.png';
    a.href = '#';
    div.appendChild(canvas);
    div.appendChild(a);
    resultsEl.appendChild(div);
  }
}

function removeStaleResults(expectedLabels) {
  resultsEl.querySelectorAll('.result').forEach(d => {
    if (!expectedLabels.includes(d.dataset.label)) d.remove();
  });
}

async function runInference() {
  if (!currentImg || running) return;
  running = true;
  cancelRequested = false;
  runBtn.disabled = true;
  cancelBtn.disabled = false;
  resultsEl.classList.remove('stale');
  progressFill.style.width = '0%';

  const mode       = document.getElementById('mode').value;
  const metalHint  = parseFloat(document.getElementById('metal-hint').value);
  const img        = currentImg;
  const W          = img.naturalWidth;
  const H          = img.naturalHeight;
  statusEl.textContent = 'Building scale pyramid...';
  await new Promise(r => setTimeout(r, 0));

  const t0 = performance.now();
  const scales = buildScales(img);
  const tScales = performance.now();

  // Output buffers: R=roughness G=AO B=metalness
  _outR = new Uint8ClampedArray(W * H);
  _outA = new Uint8ClampedArray(W * H);
  _outM = new Uint8ClampedArray(W * H);
  const outR = _outR, outA = _outA, outM = _outM;
  const pixOut = new Float32Array(3);

  resultsPanel.style.display = 'block';
  statusEl.textContent = 'Running inference... 0%';
  const CHUNK = 4; // rows per async chunk (keeps UI responsive)
  for (let y = 0; y < H; y += CHUNK) {
    const yEnd = Math.min(y + CHUNK, H);
    for (let cy = y; cy < yEnd; cy++) {
      for (let cx = 0; cx < W; cx++) {
        buildFeat(scales, cx, cy, metalHint);
        if (wasmMlp && document.getElementById('use-wasm').checked) {
          wasmFeat.set(_feat);
          wasmMlp.layer0(); wasmMlp.layer1(); wasmMlp.layer2();
          wasmMlp.layer3(); wasmMlp.layer4();
          pixOut[0] = wasmMlp.get_h4_0();
          pixOut[1] = wasmMlp.get_h4_1();
          pixOut[2] = wasmMlp.get_h4_2();
          if (cx === 0 && cy === 0) {
            const jsOut = new Float32Array(3);
            mlp(_feat, jsOut);
            console.log('px(0,0) WASM:', pixOut[0].toFixed(4), pixOut[1].toFixed(4), pixOut[2].toFixed(4));
            console.log('px(0,0)   JS:', jsOut[0].toFixed(4), jsOut[1].toFixed(4), jsOut[2].toFixed(4));
          }

        } else {
          mlp(_feat, pixOut);
        }
        const idx = cy * W + cx;
        outR[idx] = pixOut[0] * 255 + 0.5;
        outA[idx] = pixOut[1] * 255 + 0.5;
        outM[idx] = pixOut[2] * 255 + 0.5;
      }
    }
    // Yield to browser every chunk
    const pct = Math.round(yEnd / H * 100);
    progressFill.style.width = pct + '%';
    statusEl.textContent = 'Running inference... ' + pct + '%';
    showResults(mode, outR, outA, outM, W, H, yEnd);
    await new Promise(r => setTimeout(r, 0));
    if (cancelRequested) break;
  }

  const tInfer = performance.now();
  _lastW = W; _lastH = H;
  showResults(mode, outR, outA, outM, W, H, H, true);
  resultsEl.classList.remove('stale');
  resultsPanel.style.display = 'block';
  progressFill.style.width = '100%';
  statusEl.textContent = 'Done.';
  const scaleMs = (tScales - t0).toFixed(0);
  const inferMs = (tInfer - tScales).toFixed(0);
  timingEl.textContent = 'Pyramid: ' + scaleMs + 'ms   Inference: ' + inferMs + 'ms';
  running = false;
  cancelRequested = false;
  runBtn.disabled = false;
  cancelBtn.disabled = true;
}

function showResults(mode, outR, outA, outM, W, H, rowsDone, final) {
  function makeCanvas(r, g, b) {
    const c = document.createElement('canvas');
    c.width = W; c.height = rowsDone;
    const ctx = c.getContext('2d');
    // Fill remainder with mid-grey so unrendered rows aren't misleadingly dark
    if (rowsDone < H) {
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, rowsDone, W, H - rowsDone);
    }
    const id = ctx.createImageData(W, rowsDone);
    const d = id.data;
    for (let i = 0; i < W * rowsDone; i++) {
      d[i*4]   = r[i];
      d[i*4+1] = g[i];
      d[i*4+2] = b[i];
      d[i*4+3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    return c;
  }
  function makeGray(ch) {
    const c = document.createElement('canvas');
    c.width = W; c.height = rowsDone;
    const ctx = c.getContext('2d');
    if (rowsDone < H) {
      ctx.fillStyle = '#808080';
      ctx.fillRect(0, rowsDone, W, H - rowsDone);
    }
    const id = ctx.createImageData(W, rowsDone);
    const d = id.data;
    for (let i = 0; i < W * rowsDone; i++) {
      d[i*4] = d[i*4+1] = d[i*4+2] = ch[i]; d[i*4+3] = 255;
    }
    ctx.putImageData(id, 0, 0);
    return c;
  }

  if (mode === 'orm') {
    removeStaleResults(['ORM']);
    setResult('ORM', makeCanvas(outA, outR, outM), final);
  } else if (mode === 'mro') {
    removeStaleResults(['MRO']);
    setResult('MRO', makeCanvas(outM, outR, outA), final);
  } else {
    removeStaleResults(['Roughness', 'AmbientOcclusion', 'Metalness']);
    setResult('Roughness',        makeGray(outR), final);
    setResult('AmbientOcclusion', makeGray(outA), final);
    setResult('Metalness',        makeGray(outM), final);
  }
}

let _outR = null, _outA = null, _outM = null, _lastW = 0, _lastH = 0;

document.getElementById('mode').addEventListener('change', () => {
  if (!_outR) return;
  const mode = document.getElementById('mode').value;
  showResults(mode, _outR, _outA, _outM, _lastW, _lastH, _lastH, true);
});

runBtn.addEventListener('click', runInference);
cancelBtn.addEventListener('click', () => { cancelRequested = true; cancelBtn.disabled = true; statusEl.textContent = 'Cancelling...'; });

// Update download links when canvas is replaced
resultsEl.addEventListener('click', e => {
  if (e.target.tagName === 'A') {
    const div = e.target.closest('.result');
    if (div) {
      const canvas = div.querySelector('canvas');
      if (canvas) e.target.href = canvas.toDataURL('image/png');
    }
  }
});

statusEl.textContent = 'Weights loaded. Select an image to begin.';
</script>
</body>
</html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate self-contained JS inference HTML")
    ap.add_argument("--model", required=True, help="Trained weights (.pt)")
    ap.add_argument("--out",   default="pbr_infer.html", help="Output HTML file")
    args = ap.parse_args()
    generate(args.model, args.out)
