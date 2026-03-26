"""
train.py
────────
Trains a small MLP on the dataset produced by build_dataset.py.

Maps 303-element multi-scale colour features → (Roughness, AO, Metalness).

GPU support:
  - NVIDIA:        works out of the box if you install the CUDA-enabled torch wheel
  - Apple Silicon: works out of the box (MPS)
  - AMD Linux:     ROCm (probably requires manual installation)
  - AMD Windows:   oh no
  - Everything else / fallback: CPU, parallelised across all cores

Usage
─────
    uv run python train.py --data dataset.npz
    uv run python train.py --data dataset.npz --device cuda
    uv run python train.py --data dataset.npz --device cpu
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ── device ────────────────────────────────────────────────────────────────────

def pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return dev
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple Silicon)")
        return torch.device("mps")
    # Use all CPU cores
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    print(f"Using device: CPU ({n} threads)")
    return torch.device("cpu")

# ── network ───────────────────────────────────────────────────────────────────

class PBRNet(nn.Module):
    """303 → 256 → 128 → 64 → 3  (ReLU hidden, Sigmoid output)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            #nn.Linear(420, 256), nn.LeakyReLU(0.01),
            nn.Linear(672, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64),  nn.LeakyReLU(0.01),
            nn.Linear(64, 3),    nn.Sigmoid(),
            nn.Linear(3, 3),
        )
        nn.init.eye_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# ── training ──────────────────────────────────────────────────────────────────
import math
from torch.optim.lr_scheduler import LambdaLR

def train(data_path, epochs, batch, lr, val_split, seed, save_path, device, weight_decay):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    print(f"Loading {data_path} …")
    data = np.load(data_path)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    print(f"  {len(X):,} samples  |  X {X.shape}  Y {Y.shape}")

    idx   = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_split))
    X_trn, Y_trn = X[idx[n_val:]], Y[idx[n_val:]]
    X_val, Y_val = X[idx[:n_val]], Y[idx[:n_val]]
    print(f"  train: {len(X_trn):,}   val: {len(X_val):,}")

    model   = PBRNet().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    #sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.033)
    # Exponential cosine annealing:
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 
        math.exp(math.log((lr*0.033) / lr) * (1 - 0.5 * (1 + math.cos(math.pi * epoch / epochs))))
    )
    
    #loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss(delta=0.25)
    loss_mse = nn.MSELoss()

    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    print(f"Epochs: {epochs}  Batch: {batch}  LR: {lr}\n")

    steps    = max(1, int(np.ceil(len(X_trn) / batch)))
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm   = rng.permutation(len(X_trn))
        Xs, Ys = X_trn[perm], Y_trn[perm]
        t0       = time.perf_counter()
        trn_loss = 0.0

        for start in range(0, len(Xs), batch):
            xb = torch.from_numpy(Xs[start:start+batch]).to(device)
            yb = torch.from_numpy(Ys[start:start+batch]).to(device)
            # 50% chance: replace metallicity hint (indices 0-2) with 0.5 ("unknown")
            mask = (torch.rand(xb.shape[0], device=device) < 0.5).unsqueeze(1)
            xb[:, :3] = torch.where(mask, torch.full_like(xb[:, :3], 0.5), xb[:, :3])

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            trn_loss += loss.item()

        trn_loss /= steps
        elapsed   = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t)
            v_mse = loss_mse(preds, Y_val_t).item()
            v_mae = (preds - Y_val_t).abs().mean().item()

        sched.step()
        marker = " ✓" if v_mse < best_val else ""
        if v_mse < best_val:
            best_val = v_mse
            #torch.save(model.state_dict(), save_path)
            from safetensors.torch import save_file
            save_file(model.state_dict(), save_path)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"trn_loss={trn_loss:.5f}  "
            f"val_mse={v_mse:.5f}  "
            f"val_mae={v_mae:.4f}  "
            f"lr={sched.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s){marker}"
        )

    print(f"\nDone. Best val MSE: {best_val:.5f}  →  {save_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      required=True)
    ap.add_argument("--epochs",    type=int,   default=40)
    ap.add_argument("--batch",     type=int,   default=1024)
    ap.add_argument("--lr",        type=float, default=2e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed",      type=int,   default=0)
    #ap.add_argument("--save",      default="pbr_net.pt")
    ap.add_argument("--save",      default="pbr_net.safetensors")
    ap.add_argument("--device",    default=None,
                    help="cuda | mps | cpu  (auto-detected if omitted)")
    ap.add_argument("--weight-decay", type=float, default=1e-6,
                    help="L2 weight decay for Adam (default: 0, e.g. 1e-4)")
    args = ap.parse_args()

    device = pick_device(args.device)
    train(args.data, args.epochs, args.batch, args.lr,
          args.val_split, args.seed, args.save, device, args.weight_decay)
