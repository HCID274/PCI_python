#!/usr/bin/env python3
"""
测试光束在等离子体中的位置
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, '/work/DTMP/lhqing/PCI/code/pyPCI')

from pci_torch import (
    GENEConfig, 
    BeamConfig,
    load_gene_config_from_parameters,
    load_beam_config
)

def main():
    print("光束位置测试")
    print("=" * 60)
    
    # 设置设备
    device = 'cpu'
    
    # 路径配置
    input_dir = '/work/DTMP/lhqing/PCI/Data/sample/input'
    
    # 加载配置
    print("加载GENE配置...")
    config = load_gene_config_from_parameters(
        str(Path(input_dir) / 'parameters.dat'),
        str(input_dir),
        device=device
    )
    
    print("加载光束配置...")
    beam_config = load_beam_config(str(Path(input_dir) / 'LS_condition_JT60SA.txt'))
    
    print(f"Plasma Axis (PA): R={config.PA[0]:.4f}m, Z={config.PA[1]:.4f}m")
    print(f"GAC range: [{config.GAC.min():.4f}, {config.GAC.max():.4f}]m")
    print()
    
    # 光束起点和终点
    start_r = beam_config.injection_point[0]
    start_z = beam_config.injection_point[1]
    end_r = beam_config.detection_point[0] 
    end_z = beam_config.detection_point[1]
    
    print(f"光束起点: R={start_r:.4f}m, Z={start_z:.4f}m")
    print(f"光束终点: R={end_r:.4f}m, Z={end_z:.4f}m")
    print()
    
    # 计算到plasma axis的距离
    start_dist = np.sqrt((start_r - config.PA[0])**2 + (start_z - config.PA[1])**2)
    end_dist = np.sqrt((end_r - config.PA[0])**2 + (end_z - config.PA[1])**2)
    
    print(f"起点到PA的距离: {start_dist:.4f}m")
    print(f"终点到PA的距离: {end_dist:.4f}m")
    print(f"PA半径: ~{config.PA[0]:.4f}m")
    print()
    
    # 检查光束是否在等离子体边界内
    print("边界检查:")
    
    # 简化检查：假设等离子体半径约为3.0m
    plasma_radius = 3.0
    print(f"起点在边界内: {start_dist < plasma_radius}")
    print(f"终点在边界内: {end_dist < plasma_radius}")
    
    if start_dist < plasma_radius and end_dist < plasma_radius:
        print("-> 光束在等离子体内部，应该有非零值")
    elif start_dist >= plasma_radius and end_dist >= plasma_radius:
        print("-> 光束在等离子体外部，应该返回0")
    else:
        print("-> 光束穿过等离子体边界，部分点应该有非零值")

if __name__ == "__main__":
    main()
