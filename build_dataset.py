"""
build_dataset.py
────────────────
Scans a directory for PBR texture sets named:

    <n>_Color.jpg
    <n>_Roughness.jpg
    <n>_AmbientOcclusion.jpg   (optional — treated as 1.0 if absent)
    <n>_Metalness.jpg          (optional — treated as 0.0 if absent)

Feature vector layout (418 floats):
    [    0.. 2]  metallicity hint (3 copies, one per RGB channel)
    [  3..245]  → was [0..242]
    [  0..242]  9×9 @ 1/1    (81 px × 3 ch)
    [243..269]  3×3 @ 1/2    ( 9 px × 3 ch)
    [270..296]  3×3 @ 1/4    ( 9 px × 3 ch)
    [297..308]  2×2 @ 1/8    ( 4 px × 3 ch)
    [309..320]  2×2 @ 1/16   ( 4 px × 3 ch)
    [321..332]  2×2 @ 1/32   ( 4 px × 3 ch)
    [333..344]  2×2 @ 1/64   ( 4 px × 3 ch)
    [345..356]  2×2 @ 1/128  ( 4 px × 3 ch)
    [357..368]  2×2 @ 1/256  ( 4 px × 3 ch)

Usage
─────
    uv run python build_dataset.py --dir /path/to/textures --out dataset.npz

Optional flags
    --samples     Samples per texture set (default: 500)
    --seed        Random seed (default: 42)
    --ao-augment  50% chance per sample of using AO-baked colour variant
    --rot-flip    Randomly rotate (90/180/270) and/or flip each sample's patches
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

# ── constants ────────────────────────────────────────────────────────────────

SCALE_FACTORS = (1.0, 0.5, 0.25, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_rgb_f32(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def load_gray_f32(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def metal_hint(metal_img: np.ndarray) -> float:
    """
    Compute a soft metallicity hint in [0, 1] from a grayscale metal map.
    mean < 2%  → 0.0  (essentially non-metallic)
    mean > 10% → 1.0  (significantly metallic)
    in between → linear gradient
    """
    mean = float(metal_img.mean())
    if mean < 0.02:
        return 0.0
    elif mean > 0.10:
        return 1.0
    else:
        return (mean - 0.02) / (0.10 - 0.02)


def downscale(img_f32: np.ndarray, factor: float) -> np.ndarray:
    h, w = img_f32.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    pil = Image.fromarray((img_f32 * 255.0).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0

def make_scales(img: np.ndarray) -> tuple:
    """Return a tuple of 9 images: full-res through 1/256."""
    result = [img]
    for f in SCALE_FACTORS[1:]:
        result.append(downscale(img, f))
    return tuple(result)

def blur_image(img: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian blur an HxWxC float32 image."""
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    from PIL import ImageFilter
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(pil, dtype=np.float32) / 255.0


def bilinear_sample(img: np.ndarray, fy: float, fx: float) -> np.ndarray:
    """Bilinearly sample an HxWxC image at float coordinates (fy, fx)."""
    H, W = img.shape[:2]
    x0 = int(fx);  x1 = min(x0 + 1, W - 1)
    y0 = int(fy);  y1 = min(y0 + 1, H - 1)
    wx = fx - x0;  wy = fy - y0
    return (img[y0, x0] * (1-wy) * (1-wx) +
            img[y0, x1] * (1-wy) * wx     +
            img[y1, x0] * wy     * (1-wx) +
            img[y1, x1] * wy     * wx).astype(np.float32)

def bilinear_patch_NxN(img: np.ndarray, fy_c: float, fx_c: float,
                       n: int, stride_ds: float) -> np.ndarray:
    """
    N×N patch centred at (fy_c, fx_c) with given stride in downscaled pixels,
    all points bilinearly sampled. Returns (N*N*3,).
    half = (n-1)/2 steps, so offsets are e.g. -2,-1,0,1,2 for n=5.
    """
    H, W = img.shape[:2]
    half = (n - 1) / 2
    out = []
    for row in range(n):
        for col in range(n):
            dy = (row - half) * stride_ds
            dx = (col - half) * stride_ds
            out.append(bilinear_sample(img,
                                       np.clip(fy_c + dy, 0.0, H - 1.0),
                                       np.clip(fx_c + dx, 0.0, W - 1.0)))
    return np.concatenate(out)

def bilinear_patch_2x2(img: np.ndarray, fy_c: float, fx_c: float) -> np.ndarray:
    """2×2 patch at ±0.5 downscaled-pixel offsets from center, bilinearly sampled. Returns (12,)."""
    H, W = img.shape[:2]
    out = []
    for dy in (-0.5, 0.5):
        for dx in (-0.5, 0.5):
            out.append(bilinear_sample(img,
                                       np.clip(fy_c + dy, 0.0, H - 1.0),
                                       np.clip(fx_c + dx, 0.0, W - 1.0)))
    return np.concatenate(out)

def safe_patch_9x9(img: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """Extract a 9×9 integer-pixel patch from the full-res image with edge-padding. Returns (243,)."""
    h, w   = img.shape[:2]
    half   = 4
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    crop = img[y0c:y1c, x0c:x1c]
    pad_top    = y0c - y0;  pad_bottom = y1  - y1c
    pad_left   = x0c - x0;  pad_right  = x1  - x1c
    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        crop = np.pad(crop,
                      [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)],
                      mode="edge")
    return crop.flatten()

def sample_pixel(cx_full: int, cy_full: int, scales: tuple,
                 patch_color: np.ndarray = None) -> np.ndarray:
    """
    Build the 417-element feature vector for one pixel.
    scales      = make_scales(color_img), a tuple of 9 images from 1/1 to 1/256.
    patch_color = optional override used only for the 9×9 full-res patch
                  (e.g. a blurred version of scales[0]).

    Layout:
      [  0..242]  9×9  @ 1/1,  stride 1   (243)
      [243..317]  5×5  @ 1/2,  stride 1.0 ds-px = 2 src-px  (75)
      [318..344]  3×3  @ 1/4,  stride 1.0 ds-px = 4 src-px  (27)
      [345..356]  2×2  @ 1/8   (12)
      ... × 6 coarse levels
    """
    color_full = scales[0]
    H_full, W_full = color_full.shape[:2]
    patch_src = patch_color if patch_color is not None else color_full

    parts = [safe_patch_9x9(patch_src, cy_full, cx_full)]

    for i, img in enumerate(scales[1:], start=1):
        Hi, Wi = img.shape[:2]
        fx_c = cx_full * Wi / W_full
        fy_c = cy_full * Hi / H_full
        if i == 1:   # 1/2 → 5×5, stride 1.0 downscaled px (= 2 source px)
            parts.append(bilinear_patch_NxN(img, fy_c, fx_c, n=5, stride_ds=1.0))
        elif i == 2: # 1/4 → 3×3, stride 1.0 downscaled px (= 4 source px)
            parts.append(bilinear_patch_NxN(img, fy_c, fx_c, n=3, stride_ds=1.0))
        else:        # 1/8 through 1/256 → 2×2
            parts.append(bilinear_patch_2x2(img, fy_c, fx_c))

    return np.concatenate(parts)


def augment_features(feat: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """
    Apply a consistent spatial augmentation across all patch inputs.
      k    : number of 90° CCW rotations (0–3)
      flip : horizontal flip after rotation
    Layout:
      [  0..242]  9×9  → (9,9,3)
      [243..317]  5×5  → (5,5,3)
      [318..344]  3×3  → (3,3,3)
      [345..356]  2×2  → (2,2,3)  × 6 levels
      ...
      [405..416]  2×2  → (2,2,3)
    """
    if k == 0 and not flip:
        return feat

    def transform(patch2d):
        a = np.rot90(patch2d, k=k, axes=(0, 1))
        if flip:
            a = np.flip(a, axis=1)
        return a

    # Skip the 3-float metallicity hint prefix, augment only the colour features
    hint = feat[:3]
    f    = feat[3:]   # 417 colour floats
    parts = [
        transform(f[  0:243].reshape(9, 9, 3)).flatten(),
        transform(f[243:318].reshape(5, 5, 3)).flatten(),
        transform(f[318:345].reshape(3, 3, 3)).flatten(),
    ]
    for i in range(6):
        s = 345 + i * 12
        parts.append(transform(f[s:s+12].reshape(2, 2, 3)).flatten())

    return np.concatenate([hint, *parts])


def tint_features(feat: np.ndarray, rng) -> np.ndarray:
    """
    Multiply all R values by one random scalar in [0.9, 1.1],
    all G values by another, all B by another.
    Skips the first 3 floats (metallicity hint) which are not colour values.
    """
    scales = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    colour = feat[3:].reshape(-1, 3) * scales
    return np.concatenate([feat[:3], colour.reshape(-1)])


# ── main ─────────────────────────────────────────────────────────────────────

def build(image_dir: str, output_path: str, samples: int, seed: int,
          ao_augment: bool = False, rot_flip: bool = False, tint: bool = False, multires: bool = False, blur: bool = False) -> None:
    rng = np.random.default_rng(seed)

    color_paths = sorted(glob.glob(os.path.join(image_dir, "*_Color.jpg")))
    if not color_paths:
        sys.exit(f"No *_Color.jpg files found in: {image_dir}")

    all_X, all_Y = [], []
    skipped = 0

    for color_path in color_paths:
        base    = color_path[: -len("_Color.jpg")]
        p_rough = base + "_Roughness.jpg"
        p_ao    = base + "_AmbientOcclusion.jpg"
        p_metal = base + "_Metalness.jpg"

        if not os.path.exists(p_rough):
            print(f"  skip {os.path.basename(base)!r} — missing: roughness")
            skipped += 1
            continue

        name  = os.path.basename(base)
        notes = []
        if not os.path.exists(p_ao):    notes.append("ao→1")
        if not os.path.exists(p_metal): notes.append("metal→0")
        suffix = f" [{', '.join(notes)}]" if notes else ""
        print(f"  processing {name!r}{suffix} …", end=" ", flush=True)

        color_full = load_rgb_f32(color_path)
        rough      = load_gray_f32(p_rough)
        H, W       = color_full.shape[:2]
        ao    = load_gray_f32(p_ao)    if os.path.exists(p_ao)    else np.ones( (H, W), dtype=np.float32)
        metal = load_gray_f32(p_metal) if os.path.exists(p_metal) else np.zeros((H, W), dtype=np.float32)
        hint  = np.float32(metal_hint(metal) if os.path.exists(p_metal) else 0.0)

        # Build a list of (color, rough, ao, metal, n_samples) per resolution
        res_variants = [(color_full, rough, ao, metal, samples)]
        if multires:
            for factor, divisor in ((0.5, 2), (0.25, 4)):
                n = max(1, samples // divisor)
                res_variants.append((
                    downscale(color_full, factor),
                    downscale(rough,      factor),
                    downscale(ao,         factor),
                    downscale(metal,      factor),
                    n,
                ))

        n_out = 0
        for res_color, res_rough, res_ao, res_metal, n_samp in res_variants:
            rH, rW = res_color.shape[:2]

            color_scales = make_scales(res_color)
            if ao_augment:
                ao_rgb       = res_ao[:, :, np.newaxis]
                color_baked  = (res_color * ao_rgb).clip(0.0, 1.0).astype(np.float32)
                baked_scales = make_scales(color_baked)

            # Pre-compute blurred version for this resolution if blur is enabled
            if blur:
                pil_blur = Image.fromarray(
                    (res_color * 255).clip(0, 255).astype(np.uint8))
                color_blurred = np.array(
                    #pil_blur.filter(ImageFilter.GaussianBlur(radius=1.0)),
                    pil_blur.filter(ImageFilter.GaussianBlur(radius=1.3)),
                    dtype=np.float32) / 255.0
            else:
                color_blurred = None

            xs = rng.integers(0, rW, size=n_samp)
            ys = rng.integers(0, rH, size=n_samp)

            for cx, cy in zip(xs, ys):
                r_cy = np.clip(cy, 0, res_rough.shape[0] - 1)
                r_cx = np.clip(cx, 0, res_rough.shape[1] - 1)
                target = np.array([res_rough[r_cy, r_cx],
                                    res_ao   [r_cy, r_cx],
                                    res_metal[r_cy, r_cx]], dtype=np.float32)

                scales = baked_scales if (ao_augment and rng.random() < 0.5) else color_scales
                patch_color = color_blurred if (blur and rng.random() < 0.5) else None
                feat   = np.concatenate([[hint, hint, hint], sample_pixel(cx, cy, scales, patch_color)])
                if rot_flip:
                    feat = augment_features(feat,
                                            int(rng.integers(0, 4)),
                                            bool(rng.integers(0, 2)))
                if tint:
                    feat = tint_features(feat, rng)
                all_X.append(feat)
                all_Y.append(target)
                n_out += 1

        print(f"{n_out} samples")

    if not all_X:
        sys.exit("No samples collected — check your directory and filenames.")

    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.float32)
    np.savez_compressed(output_path, X=X, Y=Y)
    print(f"\n✓ Saved {len(X):,} samples → {output_path}")
    print(f"  X shape: {X.shape}   Y shape: {Y.shape}")
    if skipped:
        print(f"  ({skipped} texture sets skipped due to missing roughness map)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build PBR pixel-regression dataset")
    ap.add_argument("--dir",        required=True)
    ap.add_argument("--out",        default="dataset.npz")
    ap.add_argument("--samples",    type=int, default=500)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ao-augment", action="store_true",
                    help="50%% chance per sample of using AO-baked colour variant")
    ap.add_argument("--rot-flip",   action="store_true",
                    help="Randomly rotate (90/180/270) and/or flip each sample's patches")
    ap.add_argument("--tint",         action="store_true",
                    help="Multiply R/G/B channels by independent random scalars in [0.9, 1.1] per sample")
    ap.add_argument("--multires",     action="store_true",
                    help="Also sample at 50%% (1/4 samples) and 25%% (1/16 samples) resolution")
    ap.add_argument("--blur",         action="store_true",
                    help="50%% chance per sample of using a Gaussian-blurred version for the 9x9 patch")
    args = ap.parse_args()
    build(args.dir, args.out, args.samples, args.seed, args.ao_augment, args.rot_flip, args.tint, args.multires, args.blur)
