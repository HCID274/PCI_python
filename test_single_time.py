#!/usr/bin/env python3
"""
单时间点测试脚本 - 用于验证与MATLAB的一致性
使用配置文件管理路径
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json

# 添加pci_torch到路径
sys.path.insert(0, str(Path(__file__).parent))

from pci_torch.data_loader import (
    load_gene_config_from_parameters,
    load_beam_config,
    fread_data_s
)
from pci_torch.forward_model import forward_projection
from pci_torch.beam_geometry import compute_beam_grid
from pci_torch.visualization import PCIVisualizer, plot_wavenumber_space_2d
from pci_torch.path_config import PathConfig

def test_single_timepoint(
    config_file: str = None,
    time_t: float = None,
    device: str = None
):
    """
    测试单个时间点
    
    Args:
        config_file: 配置文件路径（可选）
        time_t: 时间点（可选，默认从配置文件读取）
        device: 设备（可选，默认从配置文件读取）
    """
    # 从配置文件加载路径配置
    try:
        path_config = PathConfig.from_config_file(config_file)
        print(f"✓ 配置文件加载成功")
        print(f"  输入目录: {path_config.input_dir}")
        print(f"  输出目录: {path_config.output_dir}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return
    
    # 加载完整的配置文件获取处理参数
    if config_file is None:
        config_file_path = Path(__file__).parent / 'config' / 'paths.json'
    else:
        config_file_path = Path(config_file)
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
    except Exception as e:
        print(f"✗ 配置文件读取失败: {e}")
        return
    
    # 获取处理参数
    processing_config = full_config.get('processing', {})
    if time_t is None:
        time_t = 0.83  # 使用固定默认值，与MATLAB版本一致
    if device is None:
        device = processing_config.get('device', 'cpu')
    
    input_path = path_config.input_dir
    output_path = path_config.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"单时间点测试: t={time_t} (设备: {device})")
    print("=" * 80)
    
    # 步骤1: 加载配置
    print("\n步骤1: 加载配置...")
    try:
        config = load_gene_config_from_parameters(
            str(path_config.parameters_file),
            equilibrium_dir=str(path_config.input_dir),
            device=device
        )
        
        print(f"  nx0={config.nx0}, inside={config.inside}, outside={config.outside}")
        print(f"  KYMt={config.KYMt}, KZMt={config.KZMt}")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return
    
    # 步骤2: 加载光束配置
    print("\n步骤2: 加载光束配置...")
    try:
        beam_config = load_beam_config(str(path_config.beam_config_file))
        print(f"  检测器: {beam_config.n_detectors_v} x {beam_config.n_detectors_t}")
        print(f"  光束采样点: {beam_config.n_beam_points}")
    except Exception as e:
        print(f"  ✗ 光束配置加载失败: {e}")
        return
    
    # 步骤3: 读取二进制数据文件
    print(f"\n步骤3: 读取时间点 t={time_t}...")
    binary_file = path_config.get_binary_data_file(int(time_t * 100))
    
    if not binary_file.exists():
        print(f"  错误: 文件不存在 {binary_file}")
        return
    
    # 检查原始数据
    try:
        raw_data = np.fromfile(binary_file, dtype=np.float64)
        print(f"  原始数据: {len(raw_data)} 个元素")
        print(f"  数据范围: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
        print(f"  数据均值: {raw_data.mean():.6f}")
        print(f"  数据标准差: {raw_data.std():.6f}")
    except Exception as e:
        print(f"  ✗ 原始数据读取失败: {e}")
        return
    
    # 加载并处理密度场
    try:
        density_3d = fread_data_s(config, str(binary_file), device=device)
        print(f"  处理后密度场shape: {density_3d.shape}")
        print(f"  密度场范围: [{density_3d.min():.6f}, {density_3d.max():.6f}]")
        print(f"  密度场均值: {density_3d.mean():.6f}")
    except Exception as e:
        print(f"  ✗ 密度场处理失败: {e}")
        return
    
    # 步骤4: 执行PCI正向投影
    print("\n步骤4: 执行PCI正向投影...")
    try:
        pci_result = forward_projection(
            density_3d,
            config,
            beam_config,
            device=device,
            return_line_integral=False
        )
        
        print(f"  PCI结果shape: {pci_result.shape}")
        print(f"  信号范围: [{pci_result.min():.6f}, {pci_result.max():.6f}]")
        print(f"  信号均值: {pci_result.mean():.6f}")
        print(f"  信号标准差: {pci_result.std():.6f}")
    except Exception as e:
        print(f"  ✗ PCI投影失败: {e}")
        return
    
    # 步骤5: 生成可视化图片
    print("\n步骤5: 生成可视化图片...")
    try:
        visualizer = PCIVisualizer(config)
        
        # 生成检测器信号等高线图（对应MATLAB Figure 3）
        detector_contour_path = output_path / f"detector_contour_t{int(time_t*100):03d}.png"
        visualizer.plot_detector_contour(
            pci_result,
            beam_config,
            time_t,
            save_path=str(detector_contour_path)
        )
        print(f"  检测器等高线图已保存: {detector_contour_path}")
        
        # 生成密度场切片图
        density_slice_path = output_path / f"density_slice_t{int(time_t*100):03d}.png"
        visualizer.plot_density_slice(
            density_3d,
            config,
            beam_config,
            save_path=str(density_slice_path)
        )
        print(f"  密度场切片图已保存: {density_slice_path}")
        
        # 生成3D正交切片图
        density_3d_path = output_path / f"density_3d_orthogonal_t{int(time_t*100):03d}.png"
        visualizer.plot_density_3d_orthogonal(
            density_3d,
            config,
            save_path=str(density_3d_path)
        )
        print(f"  3D正交切片图已保存: {density_3d_path}")
        
        # 生成波数空间图（对应MATLAB Figure 4）
        # 创建网格坐标（按照MATLAB的方式）
        wid1 = beam_config.width_vertical
        wid2 = beam_config.width_toroidal
        div1 = beam_config.div_vertical    # 垂直半范围
        div2 = beam_config.div_toroidal    # 环向半范围
        
        xx = np.meshgrid(
            wid1/2 * np.arange(-div1, div1+1) / div1,
            -wid2/2 * np.arange(-div2, div2+1) / div2
        )[0]
        yy = np.meshgrid(
            wid1/2 * np.arange(-div1, div1+1) / div1,
            -wid2/2 * np.arange(-div2, div2+1) / div2
        )[1]
        xx = np.fliplr(xx).copy()  # 添加.copy()避免负stride
        
        wavenumber_path = output_path / f"wavenumber_space_t{int(time_t*100):03d}.png"
        plot_wavenumber_space_2d(
            torch.from_numpy(xx),
            torch.from_numpy(yy),
            pci_result,
            config,
            save_path=str(wavenumber_path)
        )
        print(f"  波数空间图已保存: {wavenumber_path}")
        
    except Exception as e:
        print(f"  ✗ 可视化生成失败: {e}")
        return
    
    # 步骤6: 保存结果
    print("\n步骤6: 保存结果...")
    try:
        np.save(output_path / f"pci_result_t{int(time_t*100):03d}.npy", 
                pci_result.cpu().numpy())
        
        np.save(output_path / f"density_3d_t{int(time_t*100):03d}.npy",
                density_3d.cpu().numpy())
        
        # 保存详细报告
        with open(output_path / f"report_t{int(time_t*100):03d}.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"单时间点测试报告: t={time_t}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("配置参数:\n")
            f.write(f"  nx0={config.nx0}, inside={config.inside}, outside={config.outside}\n")
            f.write(f"  KYMt={config.KYMt}, KZMt={config.KZMt}\n")
            f.write(f"  LYM2={config.LYM2}, LZM2={config.LZM2}\n\n")
            
            f.write("原始数据:\n")
            f.write(f"  元素数: {len(raw_data)}\n")
            f.write(f"  范围: [{raw_data.min():.6f}, {raw_data.max():.6f}]\n")
            f.write(f"  均值: {raw_data.mean():.6f}\n")
            f.write(f"  标准差: {raw_data.std():.6f}\n\n")
            
            f.write("处理后密度场:\n")
            f.write(f"  Shape: {density_3d.shape}\n")
            f.write(f"  范围: [{density_3d.min():.6f}, {density_3d.max():.6f}]\n")
            f.write(f"  均值: {density_3d.mean():.6f}\n")
            f.write(f"  标准差: {density_3d.std():.6f}\n\n")
            
            f.write("PCI信号:\n")
            f.write(f"  Shape: {pci_result.shape}\n")
            f.write(f"  范围: [{pci_result.min():.6f}, {pci_result.max():.6f}]\n")
            f.write(f"  均值: {pci_result.mean():.6f}\n")
            f.write(f"  标准差: {pci_result.std():.6f}\n")
        
        print(f"  结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"  ✗ 结果保存失败: {e}")
        return
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='单时间点测试（使用配置文件）')
    parser.add_argument('--config_file', type=str, 
                        help='配置文件路径（可选）')
    parser.add_argument('--time_t', type=float, 
                        help='时间点（可选，默认从配置文件读取）')
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'cuda'],
                        help='计算设备（可选，默认从配置文件读取）')
    
    args = parser.parse_args()
    
    test_single_timepoint(
        config_file=args.config_file,
        time_t=args.time_t,
        device=args.device
    )

