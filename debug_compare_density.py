#!/usr/bin/env python3
"""
比较 GENE 密度 3D 读取结果：
  - Python: debug_gene_density_py_301_t9807.npz (density3d)
  - MATLAB: debug_gene_density_ml_301_t9807.mat (p2)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat


def main(py_npz_path: str, ml_mat_path: str):
    py_npz_path = Path(py_npz_path)
    ml_mat_path = Path(ml_mat_path)

    print("========== 加载 Python 侧 npz ==========")
    npz = np.load(py_npz_path)
    a_py = npz["density3d"]
    print(f"py density3d shape = {a_py.shape}")
    print(f"py range: [{a_py.min():.3e}, {a_py.max():.3e}], mean={a_py.mean():.3e}")

    print("\n========== 加载 MATLAB 侧 mat ==========")
    # 尝试加载 MATLAB 文件，支持 v7.2 和 v7.3 (HDF5) 格式
    print("\n========== 加载 MATLAB 侧 mat ==========")
    try:
        mat = loadmat(ml_mat_path)
    except Exception as e:
        print(f"loadmat 读取 MATLAB 文件失败: {e}")
        print("请确认 MATLAB/Octave 脚本使用了 `save('-mat', ...)` 保存为 Matlab binary 格式。")
        return

    if "p2" not in mat:
        print(f"MAT 文件中找不到变量 'p2'，实际变量有: {list(mat.keys())}")
        return

    a_ml = np.array(mat["p2"])
    print(f"ml p2 shape      = {a_ml.shape}")
    print(f"ml range: [{a_ml.min():.3e}, {a_ml.max():.3e}], mean={a_ml.mean():.3e}")

    print("\n========== 形状对比 ==========")
    if a_py.shape != a_ml.shape:
        print("⛔ shape 不一致")
        print(f"   py: {a_py.shape}")
        print(f"   ml: {a_ml.shape}")
        print("👉 说明两边在 reshape/维度顺序 上有差异，先检查 (ntheta, nx, nz) 维度排列。")
        return
    else:
        print(f"✅ shape 一致: {a_py.shape}")

    print("\n========== 数值对比 ==========")
    diff = a_py - a_ml
    max_abs = float(np.max(np.abs(diff)))
    max_ml = float(np.max(np.abs(a_ml))) + 1e-12
    rel = max_abs / max_ml

    print(f"max abs diff = {max_abs:.6e}")
    print(f"rel diff     = {rel:.6e}")

    # 顺便看几层简单统计
    ntheta, nx, nz = a_py.shape
    center_theta = ntheta // 2
    center_z = nz // 2

    slice_py = a_py[center_theta, :, center_z]
    slice_ml = a_ml[center_theta, :, center_z]
    slice_diff = slice_py - slice_ml

    print("\n--- 中心截面 (theta_mid, z_mid) 上的一维径向差异 ---")
    print(f"  slice max abs diff = {np.max(np.abs(slice_diff)):.6e}")
    print(f"  slice py range = [{slice_py.min():.3e}, {slice_py.max():.3e}]")
    print(f"  slice ml range = [{slice_ml.min():.3e}, {slice_ml.max():.3e}]")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法:")
        print("  python3 debug_compare_density.py "
              "debug_gene_density_py_301_t9807.npz "
              "../TDS_class/plot/debug_gene_density_ml_301_t9807.mat")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
