#!/usr/bin/env python3
"""
从 00009807.dat 读取 3D 密度场（Python fread_data_s）
并保存为 debug_gene_density_py_301_t9807.npz
"""

import numpy as np
from pathlib import Path

from pci_torch.path_config import PathConfig
from pci_torch.data_loader import (
    load_gene_config_from_parameters,
    fread_data_s,
    generate_timedata,
)


def main():
    base_dir = Path(__file__).parent

    # 读取路径配置（和入口脚本保持一致）
    path_config = PathConfig.from_config_file(
        str(base_dir / "config" / "paths.json")
    )

    data_n = 301
    time_point = 98.07
    time_int = int(round(time_point * 100))  # 9807

    print("=" * 80)
    print(f"[PY] 调试 GENE 密度 3D：data_n={data_n}, t={time_point} (int={time_int})")
    print("=" * 80)

    # 加载 GENEConfig（会连 EQ 一起读）
    gene_config = load_gene_config_from_parameters(
        str(path_config.parameters_file),
        str(path_config.input_dir),
        device="cpu",
    )

    # 获取二进制文件路径 00009807.dat
    binary_file = path_config.get_binary_data_file(time_int)
    print(f"[PY] 目标二进制文件: {binary_file}")

    # 建议：优先使用 MATLAB 生成好的 00009807.dat，
    # 如果不存在，再用 Python generate_timedata 生成
    if not binary_file.exists():
        print("[PY] 警告: 未找到二进制文件，将用 Python generate_timedata 生成")
        text_file = path_config.get_time_data_file(time_int)
        print(f"[PY] 对应文本文件: {text_file}")
        if not text_file.exists():
            raise FileNotFoundError(f"找不到文本数据文件: {text_file}")
        generate_timedata(gene_config, str(text_file), time_point, str(path_config.input_dir))

    # 重新确认
    if not binary_file.exists():
        raise FileNotFoundError(f"[PY] 二进制文件仍不存在: {binary_file}")

    print(f"[PY] 使用二进制文件: {binary_file}")

    # 关键：调用你现在的 Python fread_data_s
    density_3d = fread_data_s(gene_config, str(binary_file), device="cpu")
    arr = density_3d.cpu().numpy()

    print(f"[PY] 读入完成，density_3d shape = {arr.shape}")
    print(
        f"[PY] 数据范围: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}"
    )

    # 保存为 npz，方便后面直接载入对比
    out_file = base_dir / "debug_gene_density_py_301_t9807.npz"
    np.savez(
        out_file,
        density3d=arr,
        shape=np.array(arr.shape),
        nx0=np.array(gene_config.nx0),
        KYMt=np.array(gene_config.KYMt),
        KZMt=np.array(gene_config.KZMt),
        LYM2=np.array(gene_config.LYM2),
    )

    print(f"[PY] ✅ 调试文件已保存: {out_file}")


if __name__ == "__main__":
    main()
