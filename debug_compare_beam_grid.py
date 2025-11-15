#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from scipy.io import loadmat


def main():
    base_dir = Path(__file__).resolve().parent
    ml_file = base_dir.parent / "TDS_class" / "plot" / "debug_stage_3_matlab.mat"
    py_file = base_dir / "debug_stage_3_python.npz"

    print("========== 加载 MATLAB 侧 beam grid ==========")
    print(f"MATLAB 文件: {ml_file}")
    ml = loadmat(str(ml_file))

    ml_grid = ml["beam_grid_3d"]  # (div1_2, div2_2, divls_2, 3)
    ml_B2   = ml["B2"]            # (2,3)
    ml_p1   = ml["p1"].reshape(-1)  # (3,)

    print(f"ml beam_grid_3d 形状: {ml_grid.shape}")
    print(f"ml B2(1,:) (start): {ml_B2[0,:]}")
    print(f"ml B2(2,:) (end)  : {ml_B2[1,:]}")
    print(f"ml p1             : {ml_p1}, |p1|={np.linalg.norm(ml_p1):.6f}")

    print("\n========== 加载 Python 侧 beam grid ==========")
    print(f"Python 文件: {py_file}")
    py = np.load(str(py_file))

    py_grid = py["beam_grid_3d"]
    py_B2_start = py["B2_start"]
    py_B2_end   = py["B2_end"]
    py_p1       = py["p1"]

    print(f"py beam_grid_3d 形状: {py_grid.shape}")
    print(f"py B2_start: {py_B2_start}")
    print(f"py B2_end  : {py_B2_end}")
    print(f"py p1      : {py_p1}, |p1|={np.linalg.norm(py_p1):.6f}")

    print("\n========== 形状对比 ==========")
    if ml_grid.shape == py_grid.shape:
        print(f"✅ grid 形状一致: {ml_grid.shape}")
    else:
        print(f"❌ grid 形状不一致: ml {ml_grid.shape} vs py {py_grid.shape}")
        return

    print("\n========== B2 / p1 数值对比 ==========")
    print("B2_start 差值 (py - ml):", py_B2_start - ml_B2[0,:])
    print("B2_end   差值 (py - ml):", py_B2_end   - ml_B2[1,:])
    print("p1 差值 (py - ml):", py_p1 - ml_p1)

    print("\n========== 3D 网格数值对比 ==========")
    diff = py_grid - ml_grid
    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))
    print(f"max abs diff = {max_abs_diff:.3e}")
    print(f"mean abs diff= {mean_abs_diff:.3e}")

    # 分量分开看一下
    for name, idx in zip(["X", "Y", "Z"], [0, 1, 2]):
        ml_comp = ml_grid[..., idx]
        py_comp = py_grid[..., idx]
        d = py_comp - ml_comp
        print(f"\n--- {name} 分量 ---")
        print(f"  ml {name} range = [{ml_comp.min():.6f}, {ml_comp.max():.6f}]")
        print(f"  py {name} range = [{py_comp.min():.6f}, {py_comp.max():.6f}]")
        print(f"  max abs diff   = {np.max(np.abs(d)):.3e}")
        print(f"  mean abs diff  = {np.mean(np.abs(d)):.3e}")

    # 中心线检查 (div1+1, div2+1, :)
    div1_2, div2_2, divls_2, _ = ml_grid.shape
    i0 = div1_2 // 2
    j0 = div2_2 // 2
    ml_center = ml_grid[i0, j0, :, :]
    py_center = py_grid[i0, j0, :, :]
    center_diff = py_center - ml_center
    print("\n========== 中心光束路径对比 ==========")
    print(f"中心线点数: {divls_2}")
    print(f"中心线 max abs diff = {np.max(np.abs(center_diff)):.3e}")
    print("前 3 个点 (ML vs PY):")
    for k in range(min(3, divls_2)):
        print(f"  k={k}: ml={ml_center[k,:]}, py={py_center[k,:]}")

    print("\n✅ Stage 3 几何对比完成")


if __name__ == "__main__":
    main()
