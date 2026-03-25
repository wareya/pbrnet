"""
infer.py
────────
Run the trained PBR model over every pixel of an input image and write
out predicted maps alongside the input.

Feature vector layout (672 floats) — must match build_dataset.py:
    [    0..  2]  metallicity hint (3 copies, one per RGB channel)
    [    3..245]  9×9  @ 1/1    (81 px × 3 ch = 243)
    [  246..392]  7×7  @ 1/2    (49 px × 3 ch = 147)
    [  393..467]  5×5  @ 1/4    (25 px × 3 ch =  75)
    [  468..515]  4×4  @ 1/8    (16 px × 3 ch =  48)
    [  516..563]  4×4  @ 1/16
    [  564..590]  3×3  @ 1/32   ( 9 px × 3 ch =  27)
    [  591..617]  3×3  @ 1/64
    [  618..644]  3×3  @ 1/128
    [  645..671]  3×3  @ 1/256

Usage
─────
    uv run python infer.py --image SomeMaterial_Color.jpg
    uv run python infer.py --dir /path/to/textures --pattern "_Color"
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

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

# ── network (must match train.py exactly) ────────────────────────────────────

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

# Must match PATCH_SPEC in build_dataset.py exactly.
# (patch_size n, edge_pad, start_offset) — one entry per scale level.
PATCH_SPEC: tuple[tuple[int, int, int], ...] = (
    (9, 4, 0),  # 1/1   : 9×9 centred, half = 4
    (7, 3, 0),  # 1/2   : 7×7 centred, half = 3
    (5, 2, 0),  # 1/4   : 5×5 centred, half = 2
    (4, 2, 1),  # 1/8   : 4×4, rows [cy_ds-1 … cy_ds+2]
    (4, 2, 1),  # 1/16
    (3, 1, 0),  # 1/32  : 3×3 centred, half = 1
    (3, 1, 0),  # 1/64
    (3, 1, 0),  # 1/128
    (3, 1, 0),  # 1/256
)

INPUT_DIM = 3 + sum(n * n * 3 for n, _, _ in PATCH_SPEC)

def load_rgb_f32(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def downscale(img, factor):
    from math import floor
    h, w = img.shape[:2]
    nw = max(1, int(floor(w * factor)))
    nh = max(1, int(floor(h * factor)))
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0

def pad_edge(img, pad):
    return np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode="edge")

def make_scales(img):
    result = [img]
    for f1, f2 in zip(SCALE_FACTORS[1:], SCALE_FACTORS[:-1]):
        result.append(downscale(result[-1], f1/f2))
    return result   # list of 9 images

# ── feature building + inference, row-strip by row-strip ─────────────────────

def build_and_infer(model, device, batch_size, scales, metal_hint=0.5):
    """
    scales : list of 9 images from make_scales(), index 0 = full-res.
    Returns (preds (H,W,3), t_feat, t_infer).
    """
    color_full = scales[0]
    H, W = color_full.shape[:2]

    n0, pad0, _ = PATCH_SPEC[0]

    # Pad full-res for n0×n0 stride-trick extraction
    p4 = pad_edge(color_full, pad0)   # (H+2*pad0, W+2*pad0, 3)

    # Pre-compute per-scale column interpolation and row coords
    FEAT_COLS = INPUT_DIM
    scale_info = []
    col = 3 + n0 * n0 * 3   # slots 0-2 = hint; 3..col-1 = full-res patch
    for scale_idx, img in enumerate(scales[1:]):
        Hi, Wi = img.shape[:2]
        # Derive float sample offsets from PATCH_SPEC: positions offset-pad … offset-pad+n-1
        n, pad, off = PATCH_SPEC[scale_idx + 1]
        offsets = (np.arange(n, dtype=np.float32) + off - pad)
        n = len(offsets)
        width = n * n * 3

        # Row coords: (n, H)
        fy_c = np.arange(H, dtype=np.float32) * Hi / H
        fy   = np.clip(fy_c[None, :] + offsets[:, None], 0.0, Hi - 1.0)
        y0   = np.floor(fy).astype(np.int32); y1 = np.minimum(y0 + 1, Hi - 1)
        wy   = (fy - y0).astype(np.float32)

        # Column interpolation precomputed once: (Hi, W, n, 3)
        fx_c = np.arange(W, dtype=np.float32) * Wi / W
        fx   = np.clip(fx_c[:, None] + offsets[None, :], 0.0, Wi - 1.0)  # (W, n)
        x0   = np.floor(fx).astype(np.int32); x1 = np.minimum(x0 + 1, Wi - 1)
        wx   = (fx - x0).astype(np.float32)
        col_interp = (img[:, x0, :] * (1 - wx)[None, :, :, None] +
                      img[:, x1, :] *      wx [None, :, :, None])  # (Hi, W, n, 3)

        scale_info.append((col_interp, Hi, n, width, y0, y1, wy, col))
        col += width

    rows_per_strip = max(1, batch_size // W)
    preds_out = np.empty((H, W, 3), dtype=np.float32)
    t_feat = 0.0
    t_infer = 0.0

    # Pre-allocate feature buffer, reused across strips
    X = np.empty((rows_per_strip * W, FEAT_COLS), dtype=np.float32)
    X[:, 0:3] = metal_hint

    with torch.no_grad():
        for row_start in range(0, H, rows_per_strip):
            row_end = min(row_start + rows_per_strip, H)
            n_rows  = row_end - row_start
            N       = n_rows * W
            Xv      = X[:N]

            tf0 = time.perf_counter()

            # Full-res n0×n0
            strip_p4 = p4[row_start:row_end + 2*pad0]
            sp = strip_p4.strides
            Xv[:, 3:3+n0*n0*3] = np.lib.stride_tricks.as_strided(
                strip_p4,
                shape=(n_rows, W, n0, n0, 3),
                strides=(sp[0], sp[1], sp[0], sp[1], sp[2])
            ).reshape(N, n0*n0*3)

            for (col_interp, Hi, n, width, y0, y1, wy, col_start) in scale_info:
                sy0 = y0[:, row_start:row_end]  # (n, n_rows)
                sy1 = y1[:, row_start:row_end]
                swy = wy[:, row_start:row_end]
                out = Xv[:, col_start:col_start+width].reshape(n_rows, W, n, n, 3)
                for dyi in range(n):
                    r0  = sy0[dyi]
                    r1  = sy1[dyi]
                    wyr = swy[dyi, :, None, None, None]
                    out[:, :, dyi, :, :] = (col_interp[r0] * (1 - wyr) +
                                            col_interp[r1] *      wyr)

            t_feat += time.perf_counter() - tf0

            ti0 = time.perf_counter()
            xb = torch.from_numpy(Xv).to(device)
            preds_out[row_start:row_end] = model(xb).clamp(0.0, 1.0).cpu().numpy().reshape(n_rows, W, 3)
            t_infer += time.perf_counter() - ti0

    return preds_out, t_feat, t_infer


# ── inference ─────────────────────────────────────────────────────────────────

def load_model(model_path, device):
    model = PBRNet().to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.load_state_dict(load_file(model_path, device=str(device)))
    model.eval()
    print(f"Loaded model: {model_path}")
    return model


def infer_one(image_path, model, batch_size, device, packed=None, metal_hint=0.5):
    print(f"Processing {image_path}")
    color_full = load_rgb_f32(image_path)
    H, W = color_full.shape[:2]
    print(f"  Size: {W}×{H}")

    scales = make_scales(color_full)

    preds, t_feat, t_infer = build_and_infer(model, device, batch_size, scales, metal_hint)
    print(f"  Features: {t_feat:.2f}s   Inference: {t_infer:.2f}s   Total: {t_feat+t_infer:.2f}s")

    base = os.path.splitext(image_path)[0]
    R = (preds[:, :, 0] * 255).clip(0, 255).astype(np.uint8)
    A = (preds[:, :, 1] * 255).clip(0, 255).astype(np.uint8)
    M = (preds[:, :, 2] * 255).clip(0, 255).astype(np.uint8)

    if packed == "orm":
        out_path = f"{base}_pred_ORM.png"
        Image.fromarray(np.stack([A, R, M], axis=2), mode="RGB").save(out_path)
        print(f"  wrote {out_path}")
    elif packed == "mro":
        out_path = f"{base}_spec.png"
        Image.fromarray(np.stack([M, R, A], axis=2), mode="RGB").save(out_path)
        print(f"  wrote {out_path}")
    else:
        for arr, name in [(R, "Roughness"), (A, "AmbientOcclusion"), (M, "Metalness")]:
            out_path = f"{base}_pred_{name}.png"
            Image.fromarray(arr, mode="L").save(out_path)
            print(f"  wrote {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import re

    ap = argparse.ArgumentParser()
    #ap.add_argument("--model",   default="pbr_net.pt")
    ap.add_argument("--model",   default="pbr_net.safetensors")
    ap.add_argument("--batch",   type=int, default=65536,
                    help="Pixels per inference batch (lower if OOM)")
    ap.add_argument("--device",  default=None,
                    help="cuda | mps | cpu  (auto-detected if omitted)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--orm", action="store_true",
                     help="Output ORM texture (R=Occlusion, G=Roughness, B=Metalness)")
    grp.add_argument("--mro", action="store_true",
                     help="Output MRO texture as _spec.png (R=Metalness, G=Roughness, B=Occlusion)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",   help="Single input image")
    mode.add_argument("--dir",     help="Directory to scan")
    ap.add_argument("--pattern",   default=".*",
                    help="Regex matched against bare filename (only used with --dir)")
    ap.add_argument("--metal-hint", type=float, default=0.5,
                    help="Metallicity hint: 0=non-metallic, 0.5=unknown (default), 1=metallic")
    args = ap.parse_args()

    packed = "orm" if args.orm else "mro" if args.mro else None
    device = pick_device(args.device)
    model  = load_model(args.model, device)

    if args.image:
        infer_one(args.image, model, args.batch, device, packed, args.metal_hint)
    else:
        pat   = re.compile(args.pattern)
        files = sorted(
            p for p in (os.path.join(args.dir, f) for f in os.listdir(args.dir))
            if os.path.isfile(p) and pat.search(os.path.basename(p))
        )
        if not files:
            print(f"No files matched pattern {args.pattern!r} in {args.dir}")
        else:
            print(f"Found {len(files)} file(s) matching {args.pattern!r}")
            for i, path in enumerate(files, 1):
                print(f"[{i}/{len(files)}]", end=" ")
                infer_one(path, model, args.batch, device, packed, args.metal_hint)