#!/usr/bin/env python3
# compare_probe_debug.py

import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# 保证能 import pci_torch
sys.path.insert(0, str(Path(__file__).parent))
from pci_torch.path_config import PathConfig


def main():
    # 读取 paths.json，拿到 output_dir
    path_config = PathConfig.from_config_file(str(Path(__file__).parent / "config" / "paths.json"))
    out_dir = path_config.output_dir

    py_mat = out_dir / "probe_debug_py.mat"
    ml_mat = out_dir / "probe_debug_ml.mat"

    print(f"使用输出目录: {out_dir}")

    if not py_mat.exists():
        print(f"[ERROR] 找不到 Python 调试文件: {py_mat}")
        return
    if not ml_mat.exists():
        print(f"[ERROR] 找不到 Octave 调试文件: {ml_mat}")
        return

    d_py = loadmat(str(py_mat))
    d_ml = loadmat(str(ml_mat))

    z_py = float(np.squeeze(d_py["z_py"]))
    z_ml = float(np.squeeze(d_ml["z_ml"]))

    R0   = float(np.squeeze(d_py["R_debug"]))
    Z0   = float(np.squeeze(d_py["Z_debug"]))
    PHI0 = float(np.squeeze(d_py["PHI_debug"]))

    print("\n=== 探针点坐标 (来自 Python) ===")
    print(f"  R0   = {R0:.10f}")
    print(f"  Z0   = {Z0:.10f}")
    print(f"  PHI0 = {PHI0:.10f}")

    print("\n=== 插值结果对比 ===")
    print(f"  z_py = {z_py:.16e}")
    print(f"  z_ml = {z_ml:.16e}")

    diff = abs(z_py - z_ml)
    denom = max(abs(z_py), abs(z_ml), 1e-30)
    rel = diff / denom

    print(f"\n  |Δ|         = {diff:.16e}")
    print(f"  |Δ| / max|z|= {rel:.16e}")

    tol_abs = 1e-10
    tol_rel = 1e-10

    if (diff < tol_abs) or (rel < tol_rel):
        print("\n[OK] 单点插值结果在浮点精度内一致 ✅")
    else:
        print("\n[WARN] 单点插值结果存在可见差异，需要继续排查 ❌")


if __name__ == "__main__":
    main()
