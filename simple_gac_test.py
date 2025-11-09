#!/usr/bin/env python3
"""
简化的GAC数据测试 - 使用python_debug_test.py的加载方式
"""

import torch
import numpy as np
from pathlib import Path
from pci_torch.config import GENEConfig
from pci_torch.data_loader import load_gene_config_from_parameters, load_gene_config

def simple_gac_test():
    print("=== 简化的GAC测试 ===")
    
    # 使用与MATLAB相同的数据目录
    input_dir = Path("/work/DTMP/lhqing/PCI/Data/matlab_input/301")
    parameters_file = input_dir / "parameters.dat"
    
    print(f"加载参数文件: {parameters_file}")
    
    # 正确的调用方式
    config = load_gene_config_from_parameters(
        str(parameters_file),  # 参数文件
        str(input_dir),        # 数据目录  
        device='cpu'
    )
    
    print(f"参数文件加载完成")
    
    # load_gene_config_from_parameters已经加载了equilibrium数据，不需要重复加载
    # config = load_gene_config(str(equilibrium_dir), device='cpu')
    
    print(f"✓ 配置加载成功")
    print(f"GAC 形状: {config.GAC.shape}")
    print(f"GAC 最小值: {torch.min(config.GAC).item():.6f}")
    print(f"GAC 最大值: {torch.max(config.GAC).item():.6f}")
    print(f"GAC 平均值: {torch.mean(config.GAC).item():.6f}")
    
    # 分析最后一层（最外层）
    GAC_numpy = config.GAC.cpu().numpy()
    last_layer = GAC_numpy[-1, :]
    
    print(f"\n=== 最外层边界分析 ===")
    print(f"最外层范围: [{np.min(last_layer):.6f}, {np.max(last_layer):.6f}]")
    print(f"最外层前20个值: {last_layer[:20]}")
    
    # 与MATLAB期望对比
    matlab_expected = 1.15  # MATLAB调试显示的边界值
    print(f"\n=== 与MATLAB期望值对比 ===")
    print(f"MATLAB期望边界: ~{matlab_expected}")
    print(f"Python实际最小值: {np.min(last_layer):.6f}")
    print(f"差异倍数: {np.min(last_layer) / matlab_expected:.3f}")
    
    if np.min(last_layer) > 2.0:
        print("❌ 发现问题: Python的GAC值远大于MATLAB期望")
        print("   这可能解释了为什么Python认为光束在内部而MATLAB认为在外部")
    else:
        print("✅ GAC值范围正常")
    
    # 显示Plasma Axis和坐标范围
    print(f"\n=== 坐标信息 ===")
    print(f"Plasma Axis: R={config.PA[0].item():.6f}, Z={config.PA[1].item():.6f}")
    print(f"GRC范围: [{torch.min(config.GRC).item():.6f}, {torch.max(config.GRC).item():.6f}]")
    print(f"GZC范围: [{torch.min(config.GZC).item():.6f}, {torch.max(config.GZC).item():.6f}]")

if __name__ == "__main__":
    simple_gac_test()
