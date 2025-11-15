#!/usr/bin/env python3
"""
比较 MATLAB / Octave 导出的 processed_density_field_ml.mat
和 Python 导出的 processed_density_field_py.mat

使用方法（在含有这两个 mat 文件的目录下）:
    python compare_processed_fields.py
"""

import numpy as np
from scipy.io import loadmat

# 1. 读取两个 mat 文件
ml_mat = loadmat("/work/DTMP/lhqing/PCI/code/TDS_class/plot/processed_density_field_ml.mat")
py_mat = loadmat("/work/DTMP/lhqing/PCI/Data/python_output/test_single/processed_density_field_py.mat")

# 2. 取出字段（变量名按你前面保存的来）
ml = ml_mat["processed_density_field_ml"]
py = py_mat["processed_density_field_py"]

print("MATLAB field shape :", ml.shape)
print("Python  field shape:", py.shape)

# 3. 先检查形状是否一致
if ml.shape != py.shape:
    print("\n[ERROR] 形状不一致！")
    print("  MATLAB:", ml.shape)
    print("  Python:", py.shape)
    # 如果真不一致，你可以先只看这个信息，然后再贴给我
else:
    print("\n[OK] 形状一致，开始做数值对比...")

    diff = ml - py
    abs_diff = np.abs(diff)

    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()

    ml_max = np.abs(ml).max()
    if ml_max == 0:
        rel_max = np.nan
    else:
        rel_max = max_abs / ml_max

    print(f"  max |Δ|   = {max_abs:.6e}")
    print(f"  mean |Δ|  = {mean_abs:.6e}")
    print(f"  max |Δ| / max|ML| = {rel_max:.6e}")

    # 可以额外看看几个切片（比如 φ = 0 平面）误差分布的大致量级
    phi_index = 0
    slice_diff = abs_diff[:, :, phi_index]
    print(f"\n  φ = {phi_index} 平面上：")
    print(f"    max |Δ|   = {slice_diff.max():.6e}")
    print(f"    mean |Δ|  = {slice_diff.mean():.6e}")
