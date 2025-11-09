#!/usr/bin/env python3
"""
测试GAC数据的Python vs MATLAB一致性
"""

import torch
import numpy as np
from pathlib import Path
from pci_torch.data_loader import load_gene_config

def test_gac_consistency():
    print("=== GAC数据一致性测试 ===")
    
    # 加载Python配置（使用与MATLAB相同的数据）
    data_dir = Path("/work/DTMP/lhqing/PCI/Data/sample/input/parameters.dat")
    if not data_dir.exists():
        # 尝试使用父目录
        data_dir = Path("/work/DTMP/lhqing/PCI/Data/sample/input")
    config = load_gene_config(str(data_dir), device='cpu')
    
    print(f"Python Config加载成功")
    print(f"GAC 形状: {config.GAC.shape}")
    print(f"GAC 最小值: {torch.min(config.GAC).item():.6f}")
    print(f"GAC 最大值: {torch.max(config.GAC).item():.6f}")
    print(f"GAC 平均值: {torch.mean(config.GAC).item():.6f}")
    
    # 分析GAC的边界层
    GAC_numpy = config.GAC.cpu().numpy()
    
    # MATLAB调试中显示的边界值在1.14-1.15范围
    # 检查所有径向层
    print(f"\n各径向层的边界值:")
    for i in range(0, GAC_numpy.shape[0], 10):  # 每10层检查一次
        layer = GAC_numpy[i, :]
        print(f"第{i}层: 范围[{np.min(layer):.6f}, {np.max(layer):.6f}], 前5个值: {layer[:5]}")
    
    # 特别检查最后一层（最外层）
    last_layer = GAC_numpy[-1, :]
    print(f"\n=== 最外层分析 (MATLAB调试显示1.14-1.15) ===")
    print(f"最外层范围: [{np.min(last_layer):.6f}, {np.max(last_layer):.6f}]")
    print(f"最外层前20个值: {last_layer[:20]}")
    
    # 与MATLAB期望值对比
    matlab_expected_range = (1.14, 1.15)
    print(f"\n=== 与MATLAB期望值对比 ===")
    print(f"MATLAB期望的最外层边界: {matlab_expected_range}")
    print(f"Python实际的最外层范围: [{np.min(last_layer):.6f}, {np.max(last_layer):.6f}]")
    
    if np.min(last_layer) > 2.0:  # 如果最小值都大于2.0，说明单位不匹配
        print("❌ 警告: Python的GAC值明显大于MATLAB期望值 (单位不匹配?)")
        print(f"   Python最小值: {np.min(last_layer):.6f}, MATLAB期望: ~1.15")
        print(f"   比例因子: {np.min(last_layer) / 1.15:.3f}")
    else:
        print("✅ GAC值范围与MATLAB期望基本一致")
    
    # 检查坐标数据
    print(f"\n=== 坐标数据检查 ===")
    print(f"GRC 范围: [{torch.min(config.GRC).item():.6f}, {torch.max(config.GRC).item():.6f}]")
    print(f"GZC 范围: [{torch.min(config.GZC).item():.6f}, {torch.max(config.GZC).item():.6f}]")
    print(f"Plasma Axis: [{config.PA[0].item():.6f}, {config.PA[1].item():.6f}]")
    
    return config

if __name__ == "__main__":
    test_gac_consistency()
