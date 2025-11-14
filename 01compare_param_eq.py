#!/usr/bin/env python3
"""
对比 Python (npz) 和 MATLAB (mat) 导出的参数 & EQ 数据
用法:
    python compare_param_eq.py debug_gene_param_eq_py.npz debug_gene_param_eq_ml_301_t9807.mat
"""

import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat


SCALAR_KEYS = [
    "n_spec","nx0","nky0","nz0","nv0","nw0",
    "kymin","lv","lw","lx","nexc",
    "beta","debye2",
    "q0","shat","trpeps","major_R",
    "B_ref","T_ref","n_ref","L_ref","m_ref",
    "q_ref","c_ref","omega_ref","rho_ref",
    "inside","outside","IRMAX","FVER",
    "Rmax","Rmin","Zmax","Zmin",
    "NSGMAX","NTGMAX","NRGM","NZGM","NPHIGM",
    "DR1","DR2","DR3","B0",
]

ARRAY_KEYS = [
    "GRC","GZC","GFC","GAC","GTC_f","GTC_c",
    "GBPR_2d","GBPZ_2d","GBTP_2d","GBPP_2d",
    "RG1","RG2","RG3",
]


def load_npz(path):
    return np.load(path)


def load_mat(path):
    # loadmat 会把标量读成 shape=(1,1)，数组按原 shape
    return loadmat(path)


def get_scalar(container, key):
    x = container[key]
    x = np.array(x)
    return float(np.squeeze(x))


def main(py_npz_path, ml_mat_path):
    py_npz_path = Path(py_npz_path)
    ml_mat_path = Path(ml_mat_path)

    npz = load_npz(py_npz_path)
    mat = load_mat(ml_mat_path)

    print("========== 标量对比 ==========")
    for k in SCALAR_KEYS:
        if k not in npz or k not in mat:
            print(f"{k:10s}: ❌ 缺失 (npz 中有? {k in npz}, mat 中有? {k in mat})")
            continue
        py_val = get_scalar(npz, k)
        ml_val = get_scalar(mat, k)
        diff = py_val - ml_val
        print(f"{k:10s}: py={py_val: .8e}, ml={ml_val: .8e}, diff={diff: .3e}")

    print("\n========== 数组对比 ==========")
    for k in ARRAY_KEYS:
        if k not in npz or k not in mat:
            print(f"{k:10s}: ❌ 缺失 (npz 中有? {k in npz}, mat 中有? {k in mat})")
            continue
        a_py = np.squeeze(npz[k])
        a_ml = np.squeeze(mat[k])

        print(f"{k:10s}: py shape={a_py.shape}, ml shape={a_ml.shape}")
        if a_py.shape != a_ml.shape:
            print("           ⛔ shape 不一致")
            continue

        diff = np.max(np.abs(a_py - a_ml))
        rel = diff / (np.max(np.abs(a_ml)) + 1e-12)
        print(f"           max abs diff={diff:.6e}, rel={rel:.6e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python compare_param_eq.py debug_gene_param_eq_py.npz debug_gene_param_eq_ml_XXX_tYYYY.mat")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
