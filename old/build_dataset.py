"""
build_dataset.py
────────────────
Scans a directory for PBR texture sets named:

    <n>_Color.jpg
    <n>_Roughness.jpg
    <n>_AmbientOcclusion.jpg   (optional — treated as 1.0 if absent)
    <n>_Metalness.jpg          (optional — treated as 0.0 if absent)

Feature vector layout (672 floats):
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
    uv run python build_dataset.py --dir /path/to/textures --out dataset.npz

Optional flags
    --samples     Samples per texture set (default: 1000)
    --seed        Random seed (default: 42)
    --ao-augment  50% chance per sample of using AO-baked colour variant
    --rot-flip    Randomly rotate (90/180/270) and/or flip each sample's patches
    --tint        Multiply R/G/B channels by independent random scalars in [0.9, 1.1]
    --multires    Also sample at 50% (1/2 samples) and 25% (1/4 samples) resolution
    --blur        50% chance per sample of using a Gaussian-blurred version for the 9×9 patch
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

# ── constants ────────────────────────────────────────────────────────────────

SCALE_FACTORS: tuple[float, ...] = (1.0, 0.5, 0.25, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256)

# Per scale level: (patch_size n, edge_pad, start_offset).
#
# After edge-padding a downscaled image by `pad` pixels on every side,
# the patch for an in-bounds ds-coordinate (cy_ds, cx_ds) is always:
#
#   padded[cy_ds + offset : cy_ds + offset + n,
#          cx_ds + offset : cx_ds + offset + n]
#
# — no clamping, no per-pixel bilinear sampling, no Python loop.
#
# For centered odd patches (9×9, 5×5, 3×3):
#   pad = half = (n-1)//2,  offset = 0
#   → padded[cy_ds : cy_ds+n]  gives rows  cy_ds-half … cy_ds+half  ✓
#
# For the 4×4 (bracketing the ds-coordinate with one row of context on each side):
#   pad = 2,  offset = 1
#   → padded[cy_ds+1 : cy_ds+5]  gives rows  cy_ds-1 … cy_ds+2  ✓
#   (one row above and two rows below the integer ds-coordinate)
PATCH_SPEC: tuple[tuple[int, int, int], ...] = (
    (9, 4, 0),  # 1/1   : 9×9 centred, half = 4
    (7, 3, 0),  # 1/2   : 7×7 centred, half = 3
    (5, 2, 0),  # 1/4   : 5×5 centred, half = 2
    (4, 2, 1),  # 1/8   : 4×4, rows [cy_ds, cy_ds+1]
    (4, 2, 1),  # 1/16
    (3, 1, 0),  # 1/32  : 3×3 centred, half = 1
    (3, 1, 0),  # 1/64
    (3, 1, 0),  # 1/128
    (3, 1, 0),  # 1/256
)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_rgb_f32(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def load_gray_f32(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def metal_hint(metal_img: np.ndarray) -> float:
    """
    Soft metallicity hint in [0, 1].
    mean < 2%  → 0.0 (essentially non-metallic)
    mean > 10% → 1.0 (significantly metallic)
    """
    mean = float(metal_img.mean())
    if mean < 0.02:  return 0.0
    if mean > 0.10:  return 1.0
    return (mean - 0.02) / (0.10 - 0.02)


def downscale(img_f32: np.ndarray, factor: float) -> np.ndarray:
    from math import floor
    h, w = img_f32.shape[:2]
    new_w = max(1, int(floor(w * factor)))
    new_h = max(1, int(floor(h * factor)))
    pil = Image.fromarray((img_f32 * 255.0).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0


def make_padded_scales(img: np.ndarray) -> list[np.ndarray]:
    """
    Build one edge-padded image per scale level.

    For each factor in SCALE_FACTORS the image is bilinearly downsampled to
    that level's native resolution (the bilinear resampling is done once here,
    ahead of all sample extraction).  Each result is then edge-padded so that
    any valid ds-coordinate (cy_ds, cx_ds) can be extracted as the plain slice

        padded[cy_ds + offset : cy_ds + offset + n,
               cx_ds + offset : cx_ds + offset + n]

    with no clamping and no per-pixel bilinear sampling at extract time.

    One PIL conversion covers all 8 partial-res levels; a single np.pad per
    level finishes the job.
    """
    H, W     = img.shape[:2]
    img_u8   = (img * 255.0).clip(0, 255).astype(np.uint8)
    pil_full = Image.fromarray(img_u8)

    result: list[np.ndarray] = []
    prev_scaled = pil_full
    for factor, (_, pad, _) in zip(SCALE_FACTORS, PATCH_SPEC):
        if factor == 1.0:
            prev_scaled = pil_full
            scaled = img
        else:
            from math import floor
            sw = max(1, int(floor(W * factor)))
            sh = max(1, int(floor(H * factor)))
            prev_scaled = prev_scaled.resize((sw, sh), Image.BILINEAR)
            scaled = np.array(prev_scaled, dtype=np.float32) / 255.0
        result.append(np.pad(scaled, [(pad, pad), (pad, pad), (0, 0)], mode='edge'))
    return result


def make_padded_blur(img: np.ndarray) -> np.ndarray:
    """
    Gaussian-blurred full-res colour image, edge-padded ready for 9×9 extraction.
    Only the level-0 (full-res) slot needs a blur variant.
    """
    pil     = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    blurred = np.array(pil.filter(ImageFilter.GaussianBlur(radius=1.3)),
                       dtype=np.float32) / 255.0
    pad = PATCH_SPEC[0][1]   # 4
    return np.pad(blurred, [(pad, pad), (pad, pad), (0, 0)], mode='edge')


# ── vectorised feature extraction ────────────────────────────────────────────

def build_feature_matrix(
    ys: np.ndarray,                           # (N,) full-res row indices
    xs: np.ndarray,                           # (N,) full-res col indices
    H_full: int, W_full: int,
    padded_scales: list[np.ndarray],
    padded_baked:  list[np.ndarray] | None = None,
    padded_blur0:  np.ndarray       | None = None,
    use_baked: np.ndarray | None = None,      # (N,) bool
    use_blur:  np.ndarray | None = None,      # (N,) bool
) -> np.ndarray:
    """
    Vectorised batch patch extraction.  Returns (N, 669) float32.

    For each of the 9 scale levels one fancy-indexed numpy read pulls all N
    patches simultaneously — no Python loop over samples, no coordinate
    clamping, no bilinear interpolation kernel at extract time.

    Coordinate mapping (floor division):
        cy_ds = (ys * H_ds) // H_full
    This is always in [0, H_ds - 1] without explicit clipping, because
        ys ≤ H_full - 1  →  (H_full-1) * H_ds // H_full ≤ H_ds - 1.

    Variant mixing uses np.where rather than separate loops, so the baked
    and blur arrays are only read where their mask is True.
    """
    N = len(ys)
    parts: list[np.ndarray] = []

    for i, ((n, pad, offset), padded) in enumerate(zip(PATCH_SPEC, padded_scales)):
        H_ds = padded.shape[0] - 2 * pad
        W_ds = padded.shape[1] - 2 * pad

        # Map full-res coords → this level's integer ds coords (floor, always in-bounds).
        if H_ds == H_full:          # level 0: no remapping needed
            cy_ds: np.ndarray = ys
            cx_ds: np.ndarray = xs
        else:
            cy_ds = (ys * H_ds // H_full).astype(np.int32)
            cx_ds = (xs * W_ds // W_full).astype(np.int32)

        # Build index grids for the n×n patch — shape (N, n, 1) and (N, 1, n).
        arange_n = np.arange(n)
        row_idx = cy_ds[:, None, None] + offset + arange_n[None, :, None]
        col_idx = cx_ds[:, None, None] + offset + arange_n[None, None, :]

        # Single strided read: (N, n, n, 3) — no Python loop, no clamping.
        patches = padded[row_idx, col_idx]

        # ── variant mixing ────────────────────────────────────────────────────
        # Level 0 only: blur overrides colour (but NOT the baked path — that is
        # handled below and uses ~use_blur to yield baked only when not blurred).
        if i == 0:
            if use_baked is not None and padded_baked is not None:
                # Baked applies where use_baked is True AND blur isn't overriding.
                baked_mask = use_baked if use_blur is None else (use_baked & ~use_blur)
                patches = np.where(baked_mask[:, None, None, None],
                                   padded_baked[0][row_idx, col_idx],
                                   patches)
            if use_blur is not None and padded_blur0 is not None:
                patches = np.where(use_blur[:, None, None, None],
                                   padded_blur0[row_idx, col_idx],
                                   patches)
        elif use_baked is not None and padded_baked is not None:
            patches = np.where(use_baked[:, None, None, None],
                               padded_baked[i][row_idx, col_idx],
                               patches)

        parts.append(patches.reshape(N, n * n * 3))

    return np.concatenate(parts, axis=1)    # (N, 669)


# ── per-sample spatial augmentation (kept per-sample: different k each time) ─

def augment_features(feat: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """
    Apply a consistent spatial augmentation across all patch inputs.
      k    : number of 90° CCW rotations (0–3)
      flip : horizontal flip after rotation
    Operates on the 669-float colour block (feat[3:]), leaves the 3-float hint.
    """
    if k == 0 and not flip:
        return feat

    def transform(patch2d: np.ndarray) -> np.ndarray:
        a = np.rot90(patch2d, k=k, axes=(0, 1))
        return np.flip(a, axis=1) if flip else a

    hint = feat[:3]
    f    = feat[3:]
    parts = [
        transform(f[  0:243].reshape(9, 9, 3)).flatten(),   # 9×9  @ 1/1
        transform(f[243:390].reshape(7, 7, 3)).flatten(),   # 7×7  @ 1/2
        transform(f[390:465].reshape(5, 5, 3)).flatten(),   # 5×5  @ 1/4
        transform(f[465:513].reshape(4, 4, 3)).flatten(),   # 4×4  @ 1/8
        transform(f[513:561].reshape(4, 4, 3)).flatten(),   # 4×4  @ 1/16
        transform(f[561:588].reshape(3, 3, 3)).flatten(),   # 3×3  @ 1/32
        transform(f[588:615].reshape(3, 3, 3)).flatten(),   # 3×3  @ 1/64
        transform(f[615:642].reshape(3, 3, 3)).flatten(),   # 3×3  @ 1/128
        transform(f[642:669].reshape(3, 3, 3)).flatten(),   # 3×3  @ 1/256
    ]
    return np.concatenate([hint, *parts])


# ── main ─────────────────────────────────────────────────────────────────────

def build(image_dir: str, output_path: str, samples: int, seed: int,
          ao_augment: bool = False, rot_flip: bool = False,
          tint: bool = False, multires: bool = False, blur: bool = False) -> None:
    rng = np.random.default_rng(seed)

    color_paths = sorted(glob.glob(os.path.join(image_dir, "*_Color.jpg")))
    if not color_paths:
        sys.exit(f"No *_Color.jpg files found in: {image_dir}")

    all_X: list[np.ndarray] = []
    all_Y: list[np.ndarray] = []
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
        notes: list[str] = []
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

        res_variants: list[tuple] = [(color_full, rough, ao, metal, samples)]
        if multires:
            for factor, divisor in ((0.5, 2), (0.25, 4)):
                n_mv = max(1, samples // divisor)
                res_variants.append((
                    downscale(color_full, factor),
                    downscale(rough,      factor),
                    downscale(ao,         factor),
                    downscale(metal,      factor),
                    n_mv,
                ))

        n_out = 0
        for res_color, res_rough, res_ao, res_metal, n_samp in res_variants:
            rH, rW = res_color.shape[:2]

            # ── pre-build padded scale pyramids once per resolution ───────────
            # Bilinear downsampling for every partial-res level is done here,
            # ahead of all sample extraction.  Each level is immediately
            # edge-padded so that patch extraction is a plain strided copy.
            padded_scales = make_padded_scales(res_color)

            padded_baked: list[np.ndarray] | None = None
            if ao_augment:
                color_baked = (res_color * res_ao[:, :, np.newaxis]).clip(0.0, 1.0).astype(np.float32)
                padded_baked = make_padded_scales(color_baked)

            padded_blur0: np.ndarray | None = None
            if blur:
                padded_blur0 = make_padded_blur(res_color)

            # ── sample coordinates ───────────────────────────────────────────
            xs_arr = rng.integers(0, rW, size=n_samp)
            ys_arr = rng.integers(0, rH, size=n_samp)

            # Decide all variant flags up-front (avoids per-sample rng calls).
            use_baked_arr: np.ndarray | None = (rng.random(n_samp) < 0.5) if ao_augment else None
            use_blur_arr:  np.ndarray | None = (rng.random(n_samp) < 0.5) if blur        else None

            # ── vectorised extraction → (N, 669) ─────────────────────────────
            feats = build_feature_matrix(
                ys_arr, xs_arr, rH, rW,
                padded_scales, padded_baked, padded_blur0,
                use_baked_arr, use_blur_arr,
            )

            # Prepend metallicity hint → (N, 672)
            hint_block = np.full((n_samp, 3), hint, dtype=np.float32)
            feats = np.concatenate([hint_block, feats], axis=1)

            # ── targets: (N, 3) — direct integer indexing, no clipping needed ─
            # (ys_arr / xs_arr already lie in [0, rH-1] / [0, rW-1])
            targets = np.stack([
                res_rough[ys_arr, xs_arr],
                res_ao   [ys_arr, xs_arr],
                res_metal[ys_arr, xs_arr],
            ], axis=1).astype(np.float32)

            # ── augmentation ─────────────────────────────────────────────────
            # Tint is a pure per-channel scale → fully vectorised.
            if tint:
                scale_rgb = rng.uniform(0.8, 1.2, size=(n_samp, 1, 3)).astype(np.float32)
                feats[:, 3:] = (feats[:, 3:].reshape(n_samp, -1, 3) * scale_rgb).reshape(n_samp, -1)

            # Rot/flip requires a different rot90 per sample; keep a tight loop
            # but only touch samples that actually need transforming.
            if rot_flip:
                ks    = rng.integers(0, 4, n_samp)
                flips = rng.integers(0, 2, n_samp, dtype=bool)
                for j in range(n_samp):
                    if ks[j] != 0 or flips[j]:
                        feats[j] = augment_features(feats[j], int(ks[j]), bool(flips[j]))

            all_X.append(feats)
            all_Y.append(targets)
            n_out += n_samp

        print(f"{n_out} samples")

    if not all_X:
        sys.exit("No samples collected — check your directory and filenames.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    Y = np.concatenate(all_Y, axis=0).astype(np.float32)
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
    ap.add_argument("--samples",    type=int, default=1000)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ao-augment", action="store_true",
                    help="50%% chance per sample of using AO-baked colour variant")
    ap.add_argument("--rot-flip",   action="store_true",
                    help="Randomly rotate (90/180/270) and/or flip each sample's patches")
    ap.add_argument("--tint",       action="store_true",
                    help="Multiply R/G/B channels by independent scalars in [0.9, 1.1] per sample")
    ap.add_argument("--multires",   action="store_true",
                    help="Also sample at 50%% (1/2 samples) and 25%% (1/4 samples) resolution")
    ap.add_argument("--blur",       action="store_true",
                    help="50%% chance per sample of using a Gaussian-blurred 9×9 patch")
    args = ap.parse_args()
    build(args.dir, args.out, args.samples, args.seed,
          args.ao_augment, args.rot_flip, args.tint, args.multires, args.blur)