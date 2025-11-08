#!/usr/bin/env python3
"""
PCI分析统一入口脚本
支持单时间点验证和完整时间序列分析
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 添加pci_torch到路径
sys.path.insert(0, str(Path(__file__).parent))

from pci_torch.path_config import PathConfig
from pci_torch.batch_processor import process_time_series
from pci_torch.data_loader import load_gene_config_from_parameters, load_beam_config
from pci_torch.forward_model import forward_projection
from pci_torch.visualization import PCIVisualizer

def load_config(config_file: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_file is None:
        config_file = "config/paths.json"
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_single_time(config: Dict[str, Any], device: str = None):
    """运行单时间点分析"""
    print("=" * 80)
    print("单时间点分析")
    print("=" * 80)
    
    # 加载配置
    path_config = PathConfig.from_config_file("config/paths.json")
    path_config.create_output_dirs()
    
    # 获取任务参数
    task_config = config['task']
    exec_config = config['execution']
    
    # 设备选择
    if device is None:
        device = exec_config['device']
    
    # 加载数据配置
    print("加载GENE配置...")
    gene_config = load_gene_config_from_parameters(
        str(path_config.parameters_file),
        str(path_config.input_dir),
        device=device
    )
    
    print("加载光束配置...")
    beam_config = load_beam_config(str(path_config.beam_config_file))
    
    # 加载数据文件
    time_point = task_config['time_point']
    time_int = int(time_point * 100)
    binary_file = path_config.get_binary_data_file(time_int)
    
    print(f"时间点: {time_point} (文件: {binary_file.name})")
    
    # 检查并生成二进制文件
    if not binary_file.exists():
        text_file = path_config.get_time_data_file(time_int)
        if text_file.exists():
            print("生成二进制文件...")
            from pci_torch.data_loader import generate_timedata
            binary_file = generate_timedata(gene_config, str(text_file), time_point, str(path_config.input_dir))
        else:
            raise FileNotFoundError(f"数据文件不存在: {text_file}")
    
    # 更新配置
    gene_config.compute_derived_params()
    
    # 读取密度场
    print("读取密度场数据...")
    from pci_torch.data_loader import fread_data_s
    density_3d = fread_data_s(gene_config, str(binary_file), device=device)
    print(f"  密度场shape: {density_3d.shape}")
    
    # 执行PCI正向投影
    print("执行PCI正向投影...")
    pci_result = forward_projection(
        density_3d,
        gene_config,
        beam_config,
        device=device,
        return_line_integral=True
    )
    
    # 保存结果
    print("保存结果...")
    output_path = path_config.output_dir / f"single_time_t{time_point:.2f}_var{task_config['var_type']}.mat"
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为MATLAB兼容格式
    try:
        import scipy.io as sio
        sio.savemat(str(output_path), {
            'pci_signal': pci_result.cpu().numpy(),
            'time_point': time_point,
            'var_type': task_config['var_type'],
            'data_n': task_config['data_n'],
            'device': device
        })
        print(f"  结果已保存: {output_path}")
    except ImportError:
        # 如果没有scipy，保存为numpy格式
        numpy_path = output_path.with_suffix('.npy')
        import torch
        torch.save(pci_result.cpu(), numpy_path)
        print(f"  结果已保存: {numpy_path}")
    
    # 生成可视化
    if exec_config.get('save_detailed_results', True):
        print("生成可视化图表...")
        visualizer = PCIVisualizer(gene_config)
        
        # 确保输出目录存在
        path_config.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 3D光束几何图 (Figure 1 - 对应MATLAB)
        print("  Generating 3D beam geometry plot...")
        from pci_torch.beam_geometry import compute_beam_grid
        beam_grid = compute_beam_grid(beam_config, device)
        beam_fig_path = path_config.figures_dir / f"fig1_beam_geometry_t{time_point:.2f}.png"
        visualizer.plot_beam_geometry_3d(beam_grid, str(beam_fig_path))
        
        # 2. 检测器信号等高线图 (Figure 3 - 对应MATLAB)
        print("  Generating detector signal contour plot...")
        detector_fig_path = path_config.figures_dir / f"fig3_detector_signal_t{time_point:.2f}.png"
        visualizer.plot_detector_contour(
            pci_result, beam_config, time_point, str(detector_fig_path)
        )
        
        # 3. 密度场poloidal截面图 (按照MATLAB cont_data2_s.m逻辑修正)
        print("  Generating density poloidal cross-section plot...")
        density_fig_path = path_config.figures_dir / f"fig2_density_poloidal_t{time_point:.2f}.png"
        visualizer.plot_density_slice(
            density_3d, gene_config, beam_config, 
            save_path=str(density_fig_path)
        )
        
        # 4. 2D波数空间图 (Figure 4 - 对应MATLAB)
        print("  Generating 2D wavenumber space plot...")
        from pci_torch.visualization import plot_wavenumber_space_2d
        wavenumber_fig_path = path_config.figures_dir / f"fig4_wavenumber_space_t{time_point:.2f}.png"
        plot_wavenumber_space_2d(
            None, None, pci_result, gene_config, str(wavenumber_fig_path)
        )
        
        # 5. 光束位置图
        print("  生成光束位置图...")
        # 这里可以添加光束位置的可视化，如果需要的话
        
        print("  所有图表已保存")
    
    # 统计信息
    print(f"信号统计:")
    print(f"  范围: [{pci_result.min():.6f}, {pci_result.max():.6f}]")
    print(f"  均值: {pci_result.mean():.6f}")
    print(f"  标准差: {pci_result.std():.6f}")
    
    return pci_result

def run_time_series(config: Dict[str, Any], device: str = None):
    """运行时间序列分析"""
    print("=" * 80)
    print("时间序列分析")
    print("=" * 80)
    
    # 加载配置
    path_config = PathConfig.from_config_file("config/paths.json")
    path_config.create_output_dirs()
    
    # 获取任务参数
    task_config = config['task']
    exec_config = config['execution']
    
    # 设备选择
    if device is None:
        device = exec_config['device']
    
    # 加载数据配置
    print("加载GENE配置...")
    gene_config = load_gene_config_from_parameters(
        str(path_config.parameters_file),
        str(path_config.input_dir),
        device=device
    )
    
    print("加载光束配置...")
    beam_config = load_beam_config(str(path_config.beam_config_file))
    
    # 执行时间序列处理
    print("执行时间序列处理...")
    pout1, pout2 = process_time_series(
        str(path_config.input_dir),
        task_config['data_n'],
        gene_config,
        beam_config,
        var=task_config['var_type'],
        device=device,
        save_results=True,
        output_dir=str(path_config.mat_dir)
    )
    
    print(f"时间序列处理完成!")
    print(f"  LocalCross-Section shape: {pout1.shape}")
    print(f"  IntegratedSignal shape: {pout2.shape}")
    
    return pout1, pout2

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PCI分析统一入口')
    parser.add_argument('--config', type=str, default='config/paths.json',
                       help='配置文件路径')
    parser.add_argument('--task', type=str, choices=['single_time', 'time_series'],
                       help='任务类型 (覆盖配置文件)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='计算设备 (覆盖配置文件)')
    parser.add_argument('--time', type=float,
                       help='时间点 (仅单时间点任务)')
    parser.add_argument('--var', type=int, choices=[1, 2, 3, 4, 5],
                       help='变量类型 (覆盖配置文件)')
    parser.add_argument('--data_n', type=int,
                       help='数据编号 (覆盖配置文件)')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 覆盖配置参数
        if args.task:
            config['task']['type'] = args.task
        if args.device:
            config['execution']['device'] = args.device
        if args.time:
            config['task']['time_point'] = args.time
        if args.var:
            config['task']['var_type'] = args.var
        if args.data_n:
            config['task']['data_n'] = args.data_n
        
        # 显示配置
        print("配置信息:")
        print(f"  任务类型: {config['task']['type']}")
        print(f"  数据编号: {config['task']['data_n']}")
        print(f"  变量类型: {config['task']['var_type']} (1:potential, 2:A, 3:v, 4:n, 5:Te)")
        print(f"  计算设备: {config['execution']['device']}")
        
        if config['task']['type'] == 'single_time':
            print(f"  时间点: {config['task']['time_point']}")
        
        print()
        
        # 执行任务
        if config['task']['type'] == 'single_time':
            result = run_single_time(config, args.device)
        elif config['task']['type'] == 'time_series':
            result = run_time_series(config, args.device)
        else:
            raise ValueError(f"不支持的任务类型: {config['task']['type']}")
        
        print("\n" + "=" * 80)
        print("任务完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
