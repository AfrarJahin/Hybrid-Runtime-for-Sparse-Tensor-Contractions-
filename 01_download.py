#!/usr/bin/env python3
# 01_download.py
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np

try:
    # medmnist >=3 uses dataset-specific classes
    from medmnist import INFO, OrganMNIST3D
except ImportError as e:
    print("ERROR: medmnist not installed or incompatible.\n"
          "Try: pip install medmnist torchvision", file=sys.stderr)
    raise

# ----------------------------
# Utilities
# ----------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def as_numpy_4d(x) -> np.ndarray:
    """Ensure array is [1, D, H, W] float32 without copying more than needed."""
    if hasattr(x, "numpy"):  # torch.Tensor
        arr = x.numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 3:
        arr = arr[None, ...]
    elif arr.ndim != 4:
        raise ValueError(f"Unexpected volume shape {arr.shape}; expected 3D or 4D.")

    # MedMNIST volumes are uint8; convert once here
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr

def to_int_label(lab) -> int:
    # Works for torch.Tensor, numpy scalar/array, or plain int
    if hasattr(lab, "item"):
        return int(lab.item())
    arr = np.asarray(lab)
    if arr.shape == ():     # 0-D numpy scalar
        return int(arr)
    return int(arr.reshape(-1)[0])  # fallback
# ----------------------------
# Core export
# ----------------------------
def export_split_to_npz(
    split: str,
    root: str,
    out_path: str,
    normalize: bool = True,
) -> Tuple[int, Tuple[int, int, int, int]]:
    """
    Download and export one dataset split (train/val/test)
    into a single compressed NPZ file.

    Output file contains:
      - imgs: [N, 1, D, H, W] float32  (optionally normalized to [0,1])
      - labels: [N] int64
    """
    # Create the dataset object (downloads automatically if needed)
    ds = OrganMNIST3D(root=root, split=split, download=True)
    n = len(ds)
    if n == 0:
        raise RuntimeError(f"No samples found for split '{split}'.")

    # Peek at first image to know shape and type
    first_img, first_lab = ds[0]
    first = as_numpy_4d(first_img)   # [1, D, H, W]
    c, d, h, w = first.shape         # channel, depth, height, width

    # Pre-allocate large NumPy arrays for efficiency
    imgs = np.empty((n, c, d, h, w), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    # Store first item
    imgs[0] = first / 255.0 if normalize else first
    labels[0] = int(first_lab)

    log(f"Exporting split='{split}'  N={n}, shape=(1,{d},{h},{w}), normalize={normalize}")

    # Iterate over all samples
    for i in range(1, n):
        img, lab = ds[i]
        arr = as_numpy_4d(img)
        # Verify consistent shape
        if arr.shape != (c, d, h, w):
            raise ValueError(f"Inconsistent shape at {i}: {arr.shape} vs {(c,d,h,w)}")

        if normalize:
            arr = arr / 255.0  # scale pixel values to [0,1]

        imgs[i] = arr
        labels[i] = int(lab)

        # Print progress every 500 samples
        if (i % 500) == 0 or i == n - 1:
            log(f"  processed {i+1}/{n}")

    # Save compressed .npz file (both arrays together)
    ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
    np.savez_compressed(out_path, imgs=imgs, labels=labels)
    log(f"Saved: {out_path}  imgs={imgs.shape}  labels={labels.shape}")

    return n, (c, d, h, w)

# =============================================================
# Command-line interface
# Allows you to run from terminal like:
#    python 01_download.py --split train --normalize
# =============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download OrganMNIST3D and export a split to a single NPZ."
    )
    p.add_argument("--split", choices=("train", "val", "test"), default="train",
                   help="Which split to export.")
    p.add_argument("--root", default="./data",
                   help="Directory to store MedMNIST cache/downloads.")
    p.add_argument("--out", default=None,
                   help="Output .npz path (default: ./organmnist3d_<split>.npz).")
    p.add_argument("--normalize", action="store_true",
                   help="Scale volumes to [0,1] (from uint8 0..255).")
    return p.parse_args()

# =============================================================
# Program entrypoint
# =============================================================
def main() -> None:
    args = parse_args()
    ensure_dir(args.root)

    # Determine output filename
    out_path = args.out or f"./organmnist3d_{args.split}.npz"

    # INFO dictionary provides meta-info (not essential but useful)
    info = INFO.get("organmnist3d")
    if info is None:
        log("WARNING: INFO for 'organmnist3d' not found; continuing anyway.")

    try:
        export_split_to_npz(
            split=args.split,
            root=args.root,
            out_path=out_path,
            normalize=args.normalize,
        )
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)



if __name__ == "__main__":
    main()
