#!/usr/bin/env python3
"""
导出 Python 侧的 GENE 参数 + EQ 数据，用于和 MATLAB 对比
"""

import numpy as np
import torch
from pathlib import Path

from pci_torch.path_config import PathConfig
from pci_torch.data_loader import load_gene_config_from_parameters


def _to_np(x):
    """统一转成 numpy 数组"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def main():
    base_dir = Path(__file__).parent
    paths_json = base_dir / "config" / "paths.json"

    # 用和主入口一样的路径配置
    path_config = PathConfig.from_config_file(str(paths_json))

    # 关键：用 load_gene_config_from_parameters，这个会调用 load_equilibrium_data
    gene_config = load_gene_config_from_parameters(
        str(path_config.parameters_file),
        str(path_config.input_dir),
        device="cpu",
    )

    # -------- 标量部分（和 MATLAB 那边保持同名） --------
    scalars = {}

    def put_scalar(name, value, default=np.nan):
        if value is None:
            scalars[name] = float(default)
        else:
            scalars[name] = float(value)

    put_scalar("n_spec", getattr(gene_config, "n_spec", None))
    put_scalar("nx0", getattr(gene_config, "nx0", None))
    put_scalar("nky0", getattr(gene_config, "nky0", None))
    put_scalar("nz0", getattr(gene_config, "nz0", None))
    put_scalar("nv0", getattr(gene_config, "nv0", None))
    put_scalar("nw0", getattr(gene_config, "nw0", None))
    put_scalar("kymin", getattr(gene_config, "kymin", None))
    put_scalar("lv", getattr(gene_config, "lv", None))
    put_scalar("lw", getattr(gene_config, "lw", None))
    put_scalar("lx", getattr(gene_config, "lx", None))
    put_scalar("nexc", getattr(gene_config, "nexc", None))

    put_scalar("beta", getattr(gene_config, "beta", None))
    put_scalar("debye2", getattr(gene_config, "debye2", None))

    put_scalar("q0", getattr(gene_config, "q0", None))
    put_scalar("shat", getattr(gene_config, "shat", None))
    put_scalar("trpeps", getattr(gene_config, "trpeps", None))
    put_scalar("major_R", getattr(gene_config, "major_R", None))

    put_scalar("B_ref", getattr(gene_config, "B_ref", None))
    put_scalar("T_ref", getattr(gene_config, "T_ref", None))
    put_scalar("n_ref", getattr(gene_config, "n_ref", None))
    put_scalar("L_ref", getattr(gene_config, "L_ref", None))
    put_scalar("m_ref", getattr(gene_config, "m_ref", None))

    put_scalar("q_ref", getattr(gene_config, "q_ref", None))
    put_scalar("c_ref", getattr(gene_config, "c_ref", None))
    put_scalar("omega_ref", getattr(gene_config, "omega_ref", None))
    put_scalar("rho_ref", getattr(gene_config, "rho_ref", None))

    put_scalar("inside", getattr(gene_config, "inside", None))
    put_scalar("outside", getattr(gene_config, "outside", None))
    put_scalar("IRMAX", getattr(gene_config, "IRMAX", getattr(gene_config, "nx0", None)))
    put_scalar("FVER", getattr(gene_config, "FVER", 5.0))

    # GRC/GZC 等插值出来后才有的
    for name in ["Rmax", "Rmin", "Zmax", "Zmin", "NSGMAX", "NTGMAX",
                 "NRGM", "NZGM", "NPHIGM", "DR1", "DR2", "DR3", "B0"]:
        if hasattr(gene_config, name):
            put_scalar(name, getattr(gene_config, name))

    # -------- 数组部分 --------
    arrays = {}

    # 坐标网格
    for name in ["GRC", "GZC", "GFC", "GAC", "GTC_f", "GTC_c"]:
        if hasattr(gene_config, name):
            arrays[name] = _to_np(getattr(gene_config, name))

    # B 场 2D 截面
    for name in ["GBPR_2d", "GBPZ_2d", "GBTP_2d", "GBPP_2d"]:
        if hasattr(gene_config, name):
            arrays[name] = _to_np(getattr(gene_config, name))

    # 网格范围 RG1/2/3
    for name in ["RG1", "RG2", "RG3"]:
        if hasattr(gene_config, name):
            arrays[name] = np.array(getattr(gene_config, name), dtype=np.float64)

    out_file = base_dir / "debug_gene_param_eq_py.npz"
    np.savez(out_file, **scalars, **arrays)

    print("✅ Python 侧参数 / EQ 数据已导出到:")
    print(f"   {out_file}\n")
    print("部分标量：")
    for k in ["nx0", "nky0", "nz0", "q0", "trpeps", "B_ref", "rho_ref", "inside", "outside"]:
        if k in scalars:
            print(f"   {k:8s} = {scalars[k]:.8e}")


if __name__ == "__main__":
    main()
