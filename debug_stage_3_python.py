#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from pathlib import Path

from pci_torch.data_loader import load_beam_config
from pci_torch.beam_geometry import compute_beam_grid


def main():
    base_dir = Path(__file__).resolve().parent
    # 实际路径：从项目根目录到 Data/python_input
    ls_condition = base_dir.parent.parent / "Data" / "python_input" / "LS_condition_JT60SA.txt"

    print("========== 加载 LS_condition_JT60SA.txt ==========")
    print(f"路径: {ls_condition}")
    beam_config = load_beam_config(str(ls_condition))

    print("========== 计算 Python 光束网格 ==========")
    beam_grid = compute_beam_grid(beam_config, config=None, device="cpu", debug=True)

    grid_xyz = beam_grid["grid_xyz"].cpu().numpy()       # (div1_2, div2_2, divls_2, 3)
    beam_start = beam_grid["beam_start"].cpu().numpy()   # 注入点 B2(1,:)
    beam_end   = beam_grid["beam_end"].cpu().numpy()     # 探测点 B2(2,:)
    p1_unit    = beam_grid["beam_vector"].cpu().numpy()  # 单位向量
    b2ls       = beam_grid["beam_length"].item()         # 总长度
    p1         = p1_unit * b2ls                          # 还原出和 MATLAB 一样的 p1
    xl_unit    = beam_grid["perpendicular_vectors"].cpu().numpy()

    div1_2, div2_2, divls_2, _ = grid_xyz.shape
    print(f"grid_xyz 形状: {grid_xyz.shape}")
    print(f"B2_start (Python): {beam_start}")
    print(f"B2_end   (Python): {beam_end}")
    print(f"p1 (Python): {p1}, |p1|={np.linalg.norm(p1):.6f}")

    out_path = base_dir / "debug_stage_3_python.npz"
    np.savez(
        out_path,
        beam_grid_3d=grid_xyz,
        B2_start=beam_start,
        B2_end=beam_end,
        p1=p1,
        xl_unit=xl_unit,
    )
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
