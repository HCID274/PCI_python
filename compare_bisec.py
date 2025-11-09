#!/usr/bin/env python3
"""
比较Python和MATLAB版本的bisec函数行为
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, '/work/DTMP/lhqing/PCI/code/pyPCI')

from pci_torch.config import GENEConfig, BeamConfig
from pci_torch.data_loader import load_gene_config_from_parameters, load_beam_config
from pci_torch.utils import bisec

def compare_bisec_implementations():
    print("=== 比较bisec函数实现 ===")
    print()
    
    # 设置设备
    device = 'cpu'
    
    # 路径配置
    input_dir = '/work/DTMP/lhqing/PCI/Data/sample/input'
    
    # 加载配置
    print("1. 加载配置...")
    config = load_gene_config_from_parameters(
        str(Path(input_dir) / 'parameters.dat'),
        str(input_dir),
        device=device
    )
    
    beam_config = load_beam_config(str(Path(input_dir) / 'LS_condition_JT60SA.txt'))
    
    # 测试theta值
    GTC_c_last = config.GTC_c[-1, :]
    print(f"GTC_c_last形状: {GTC_c_last.shape}")
    print(f"GTC_c_last范围: [{GTC_c_last.min():.6f}, {GTC_c_last.max():.6f}]")
    
    # 测试几个theta值
    test_thetas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.28]
    
    print("2. 测试bisec函数...")
    for theta in test_thetas:
        try:
            idx1, idx2 = bisec(theta, GTC_c_last)
            print(f"   theta={theta:.3f}: idx1={idx1}, idx2={idx2}")
            if idx1 < len(GTC_c_last) and idx2 < len(GTC_c_last):
                print(f"             GTC_c[{idx1}]={GTC_c_last[idx1]:.6f}, GTC_c[{idx2}]={GTC_c_last[idx2]:.6f}")
            
            # 检查边界值
            r_boundary_l = config.GAC[-1, idx1]
            r_boundary_u = config.GAC[-1, idx2]
            print(f"             边界: [{r_boundary_l.item():.6f}, {r_boundary_u.item():.6f}]")
        except Exception as e:
            print(f"   theta={theta:.3f}: 错误 - {e}")
    
    # 模拟MATLAB的GAC查找逻辑
    print("3. 测试GAC查找...")
    test_r = 1.5  # 一个测试的r值
    
    # 对于每个theta索引，测试GAC查找
    for theta_idx in [100, 200, 300]:
        try:
            if theta_idx < GTC_c_last.shape[0]:
                theta_val = GTC_c_last[theta_idx].item()
                print(f"   theta_idx={theta_idx}, theta={theta_val:.6f}")
                
                # 获取该theta对应的GAC列
                GAC_at_theta = config.GAC[:, theta_idx]
                print(f"   GAC_at_theta范围: [{GAC_at_theta.min():.6f}, {GAC_at_theta.max():.6f}]")
                
                # 查找r的索引
                r_diffs = torch.abs(GAC_at_theta - test_r)
                r_p_lower = torch.argmin(r_diffs)
                r_boundary = config.GAC[r_p_lower, theta_idx]
                
                print(f"   r={test_r}: 最近索引={r_p_lower}, 边界值={r_boundary.item():.6f}")
                print(f"   r < 边界: {test_r < r_boundary.item()}")
        except Exception as e:
            print(f"   theta_idx={theta_idx}: 错误 - {e}")
    
    # 详细分析一个具体的点
    print("4. 详细分析光束起点...")
    start_r = beam_config.injection_point[0]
    start_z = beam_config.injection_point[1]
    PA = config.PA
    
    r = torch.sqrt((start_r - PA[0])**2 + (start_z - PA[1])**2).item()
    theta = torch.remainder(torch.atan2(start_z - PA[1], start_r - PA[0]), 2*np.pi).item()
    
    print(f"   光束起点: R={start_r:.6f}, Z={start_z:.6f}")
    print(f"   相对坐标: r={r:.6f}, theta={theta:.6f}")
    
    # bisec查找theta索引
    theta_idx1, theta_idx2 = bisec(theta, GTC_c_last)
    print(f"   bisec结果: idx1={theta_idx1}, idx2={theta_idx2}")
    
    # 边界检查
    r_boundary_l = config.GAC[-1, theta_idx1].item()
    r_boundary_u = config.GAC[-1, theta_idx2].item()
    
    print(f"   边界检查:")
    print(f"     r={r:.6f} < {r_boundary_l:.6f} = {r < r_boundary_l}")
    print(f"     r={r:.6f} < {r_boundary_u:.6f} = {r < r_boundary_u}")
    print(f"     总体条件: {(r < r_boundary_l) and (r < r_boundary_u)}")
    
    # 如果条件不满足，尝试找到能通过检查的theta值
    print("5. 寻找能通过边界检查的theta值...")
    found_valid = False
    for test_theta in np.linspace(0, 2*np.pi, 100):
        try:
            idx1, idx2 = bisec(test_theta, GTC_c_last)
            boundary1 = config.GAC[-1, idx1].item()
            boundary2 = config.GAC[-1, idx2].item()
            
            # 找到合适的r值
            for test_r_candidate in np.linspace(0.5, 1.5, 20):
                if (test_r_candidate < boundary1) and (test_r_candidate < boundary2):
                    print(f"   ✓ 找到有效点: theta={test_theta:.3f}, r={test_r_candidate:.3f}")
                    print(f"     边界: [{boundary1:.3f}, {boundary2:.3f}]")
                    found_valid = True
                    break
            if found_valid:
                break
        except:
            continue
    
    if not found_valid:
        print("   ✗ 没有找到能通过边界检查的点")
        
    # 检查密度场的实际数据
    print("6. 检查密度场数据...")
    time_point = 98.07
    time_int = int(time_point * 100)
    binary_file = Path(input_dir) / f"{time_int:08d}.dat"
    
    from pci_torch.data_loader import fread_data_s
    density_3d = fread_data_s(config, str(binary_file), device=device)
    
    print(f"   密度场形状: {density_3d.shape}")
    print(f"   密度场非零元素: {(density_3d != 0).sum().item()}")
    print(f"   密度场最大值: {density_3d.max().item():.6f}")
    print(f"   密度场最小值: {density_3d.min().item():.6f}")
    
    # 检查一些实际的密度值
    non_zero_indices = torch.where(density_3d != 0)
    if len(non_zero_indices[0]) > 0:
        sample_idx = 0
        n_idx = non_zero_indices[0][sample_idx].item()
        m_idx = non_zero_indices[1][sample_idx].item()  
        p_idx = non_zero_indices[2][sample_idx].item()
        value = density_3d[n_idx, m_idx, p_idx].item()
        print(f"   第一个非零值: data({n_idx}, {m_idx}, {p_idx}) = {value:.6f}")

if __name__ == "__main__":
    compare_bisec_implementations()
