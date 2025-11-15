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
import matplotlib.pyplot as plt
import torch
import numpy as np


def load_config(config_file: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_file is None:
        config_file = "config/paths.json"
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_probe_debug_data(
    gene_config,
    beam_config,
    density_3d,
    debug_info,
    device: str,
    path_config: PathConfig,
):
    """
    使用 Python 插值内核在单个采样点上做调试，并导出给 Octave 使用。

    生成文件:
      - {output_dir}/probe_debug_py.mat
    """
    try:
        from pci_torch.beam_geometry import compute_beam_grid
        from pci_torch.interpolation import probe_local_trilinear
        import scipy.io as sio
        import numpy as np
        import torch
    except Exception as e:
        print(f"[WARN] export_probe_debug_data 依赖导入失败，跳过探针调试导出: {e}")
        return

    if "processed_density_field_py" not in debug_info:
        print("[WARN] debug_info 中没有 'processed_density_field_py'，跳过探针调试导出")
        return

    data3 = debug_info["processed_density_field_py"]
    if isinstance(data3, torch.Tensor):
        # shape: (1, ntheta+1, nx_ext, nz+1) 或 (ntheta+1, nx_ext, nz+1)
        data3 = data3.squeeze(0).detach().to("cpu")

    # 重新计算 beam 网格，取中间的一个点作为探针位置
    beam_grid = compute_beam_grid(beam_config, config=gene_config, device=device, debug=False)
    grid_xyz = beam_grid["grid_xyz"]  # (n_det_v, n_det_t, n_beam, 3)
    n_det_v, n_det_t, n_beam, _ = grid_xyz.shape

    iv = n_det_v // 2
    it = n_det_t // 2
    ib = n_beam // 2

    point = grid_xyz[iv, it, ib, :]  # (3,)
    x = float(point[0].item())
    y = float(point[1].item())
    z = float(point[2].item())

    # 直角坐标 -> 柱坐标 (R, Z, PHI)
    # 与 MATLAB 一样: x = R cos(phi), y = R sin(phi), z = Z
    R_debug = (x ** 2 + y ** 2) ** 0.5
    PHI_debug = np.arctan2(y, x)
    Z_debug = z

    print("\n=== Python 单点插值调试 ===")
    print(f"  选取探针点索引: iv={iv}, it={it}, ib={ib}")
    print(f"  Cartesian: x={x:.6f}, y={y:.6f}, z={z:.6f}")
    print(f"  Cylindrical: R={R_debug:.6f}, Z={Z_debug:.6f}, PHI={PHI_debug:.6f}")

    # 调用你已经重写好的 probe_local_trilinear
    R_t = torch.tensor(R_debug, dtype=data3.dtype, device=device)
    Z_t = torch.tensor(Z_debug, dtype=data3.dtype, device=device)
    PHI_t = torch.tensor(PHI_debug, dtype=data3.dtype, device=device)

    data3_dev = data3.to(device)

    z_py_t = probe_local_trilinear(data3_dev, R_t, Z_t, PHI_t, gene_config)
    z_py = float(z_py_t.detach().cpu().item())
    print(f"  Python 插值结果 z_py = {z_py:.16e}")

    # equilibrium 信息转 numpy
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        import numpy as np
        return np.asarray(x)

    import numpy as np
    mat_dict = {
        "processed_density_field_py": data3.cpu().numpy(),   # (ntheta+1, nx_ext, nz+1)
        "PA": _to_np(gene_config.PA),                        # (2,)
        "GAC": _to_np(gene_config.GAC),                      # (nx, ntheta)
        "GTC_c": _to_np(gene_config.GTC_c),                  # (something, ntheta)
        "KZMt": np.array(gene_config.KZMt, dtype=np.int32),  # 标量
        "R_debug": np.array(R_debug, dtype=np.float64),
        "Z_debug": np.array(Z_debug, dtype=np.float64),
        "PHI_debug": np.array(PHI_debug, dtype=np.float64),
        "z_py": np.array(z_py, dtype=np.float64),
    }

    out_path = path_config.output_dir / "probe_debug_py.mat"
    sio.savemat(str(out_path), mat_dict)
    print(f"  调试数据已保存到: {out_path}\n")


def run_single_time(config: Dict[str, Any], device: str = None):
    """运行单时间点分析"""
    print("=" * 80)
    print("单时间点分析")
    print("=" * 80)
    
    # 加载配置
    path_config = PathConfig.from_config_file(str(Path(__file__).parent / "config" / "paths.json"))
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
    
        # 加载数据文件（文本 + 生成二进制）
    time_point = task_config['time_point']
    time_int = int(time_point * 100)

    print(f"时间点: {time_point} (t*100 = {time_int:04d})")

    # 对应 MATLAB: GENEdata = sprintf('%sTORUSIons_act_%.0f.dat', dataC.indir, t*100)
    text_file = path_config.get_time_data_file(time_int)
    if not text_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {text_file}")

    print("从文本生成/刷新二进制密度数据 (generate_timedata)...")
    from pci_torch.data_loader import generate_timedata, fread_data_s

    # ⚠️ 这里不再判断二进制是否存在，始终按 MATLAB 流程重建，
    # 同时在 config 上写入 KYMt / KZMt
    binary_file = generate_timedata(
        gene_config,
        str(text_file),
        time_point,
        str(path_config.input_dir),
    )

    # generate_timedata 设置了 KYMt / KZMt，这里重新计算衍生参数
    if hasattr(gene_config, "compute_derived_params"):
        gene_config.compute_derived_params()

    # 读取密度场 3D (ntheta, nx, nz)，对应 MATLAB 的 p2
    print("读取密度场数据 (fread_data_s)...")
    density_3d = fread_data_s(gene_config, str(binary_file), device=device)
    print(f"  密度场 shape: {tuple(density_3d.shape)}")
    
    # 执行PCI正向投影
    print("执行PCI正向投影...")
    pci_result, debug_info = forward_projection(
        density_3d, gene_config, beam_config, 
        device=device, return_line_integral=True, return_debug_info=True  # DEBUG: 设置为True获取中间数据
    )
    
    # === 导出 Stage 4: Python 侧 processed_density_field_py 用于和 MATLAB 对比 ===
    from scipy.io import savemat
    import numpy as np
    import torch
    
    if 'processed_density_field_py' in debug_info:
        proc = debug_info['processed_density_field_py']
        if isinstance(proc, torch.Tensor):
            # 去掉 batch 维（如果有），放到 CPU
            proc_np = proc.squeeze(0).detach().cpu().numpy()
        else:
            proc_np = np.array(proc)
        proc_mat_path = path_config.output_dir / "processed_density_field_py.mat"
        savemat(str(proc_mat_path), {'processed_density_field_py': proc_np})
        print(f"Saved processed_density_field_py.mat to: {proc_mat_path}")
    else:
        print("[WARN] debug_info 里没有 'processed_density_field_py'，请检查 forward_projection 是否按 Stage 4 修改")
    
    # === 单点插值调试导出 (给 Octave 用) ===
    export_probe_debug_data(
        gene_config,
        beam_config,
        density_3d,
        debug_info,
        device=device,
        path_config=path_config,
    )
    
    # 保存结果
    print("保存结果...")
    output_path = path_config.output_dir / f"single_time_t{time_point:.2f}_var{task_config['var_type']}.mat"
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为MATLAB兼容格式
    try:
        import scipy.io as sio
        
        # 保存调试信息
        debug_data = {
            'pci_signal': pci_result.cpu().numpy(),
            'time_point': time_point,
            'var_type': task_config['var_type'],
            'data_n': task_config['data_n'],
            'device': device
        }
        
        # 添加MATLAB兼容的变量名
        pci_np = pci_result.cpu().numpy()
        
        # 确保numpy在作用域内可用
        import numpy as np
        # pout: 沿光束路径的1D信号（取中心检测器位置的平均）
        center_v = pci_np.shape[0] // 2  # 垂直中心
        center_t = pci_np.shape[1] // 2  # 环向中心
        pout = pci_np[center_v, center_t, :].flatten()  # 1D信号对应MATLAB的plot(abs(pout))
        debug_data['pout'] = pout
        
        # pout2: 2D检测器信号（取光束中点）
        # 修复：不要只取中间点，可能该点数据为0
        center_beam = pci_np.shape[2] // 2
        
        # 尝试多个时间点，看哪个有有效数据
        candidate_points = [center_beam, center_beam//2, center_beam*3//2, 0, pci_np.shape[2]-1]
        pout2 = None
        
        for pt in candidate_points:
            if pt < pci_np.shape[2]:
                temp_pout2 = pci_np[:, :, pt]
                if np.any(np.abs(temp_pout2) > 1e-6):  # 检查是否有非零数据
                    pout2 = temp_pout2
                    break
        
        # 如果所有候选点都是0，取所有时间点的平均
        if pout2 is None or np.all(np.abs(pout2) < 1e-6):
            pout2 = np.mean(pci_np, axis=2)  # 沿光束路径取平均
            # 如果平均还是0，取最大值时间点
            if np.all(np.abs(pout2) < 1e-6):
                max_indices = np.unravel_index(np.argmax(np.abs(pci_np), axis=None), pci_np.shape)
                pout2 = pci_np[:, :, max_indices[2]]
        else:
            pout2 = pout2
        debug_data['pout2'] = pout2
        
        # 如果有中间结果，也保存
        if 'debug_info' in locals():
            import torch
            for key, value in debug_info.items():
                if isinstance(value, torch.Tensor):
                    debug_data[key] = value.cpu().numpy()
                else:
                    debug_data[key] = value
        
        sio.savemat(str(output_path), debug_data)
        print(f"  结果已保存: {output_path}")
        
        # 同时保存NPZ格式用于详细分析
        npz_path = output_path.with_suffix('.npz')
        import torch
        import numpy as np
        npz_data = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                   for k, v in debug_data.items()}
        np.savez(str(npz_path), **npz_data)
        print(f"  调试数据已保存: {npz_path}")
        
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
        
        # 3. 密度场poloidal截面图 -> 修正为沿光束路径的PCI信号分布图
        print("  Generating beam path PCI signal distribution plot...")
        # 获取沿光束路径的PCI信号（对应MATLAB的pout）
        pci_result, debug_info = forward_projection(
            density_3d, gene_config, beam_config, 
            device=device, return_line_integral=True, return_debug_info=True
        )
        # pci_result形状: (n_det_v, n_det_t, n_beam_points)
        # 需要flatten到1D信号用于绘图
        pci_signal_1d = pci_result.flatten()  # 对应MATLAB的abs(pout)
        
        # 生成信号分布图
        beam_signal_fig_path = path_config.figures_dir / f"fig2_density_poloidal_t{time_point:.2f}.png"
        create_beam_path_signal_plot(pci_signal_1d, str(beam_signal_fig_path))
        
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

def create_beam_path_signal_plot(pci_signal_1d, save_path):
    """生成沿光束路径的PCI信号分布图（对应MATLAB的plot(abs(pout))）
    
    Args:
        pci_signal_1d: 1D PCI信号数组
        save_path: 图片保存路径
    """
    fig = plt.figure(figsize=(12, 8))
    
    # 计算信号绝对值（对应MATLAB的abs(pout)）
    signal_abs = torch.abs(pci_signal_1d)
    signal_np = signal_abs.cpu().numpy()
    
    # 绘制信号分布
    plt.plot(signal_np, 'b-', linewidth=1.5, label='PCI Signal')
    plt.xlabel('Beam Path Point')
    plt.ylabel('Signal Magnitude')
    plt.title('PCI Signal Distribution Along Beam Path (Figure 2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    max_val = signal_np.max()
    min_val = signal_np.min()
    mean_val = signal_np.mean()
    plt.text(0.02, 0.98, f'Max: {max_val:.2f}\nMin: {min_val:.2f}\nMean: {mean_val:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Beam path signal plot saved: {save_path}")
    plt.close()

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
        
        # 智能设备检测
        if args.device in ['cuda']:
            import torch
            if torch.cuda.is_available():
                if hasattr(torch.version, 'hip'):
                    print(f"  检测到可用AMD GPU (ROCM): {torch.cuda.get_device_name(0)}")
                    print(f"  ROCm版本: {torch.version.hip}")
                elif args.device == 'cuda':
                    print(f"  检测到可用NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                    print(f"  CUDA版本: {torch.version.cuda}")
            else:
                print(f"  警告: 指定了{args.device}但系统无可用GPU，回退到CPU")
                args.device = 'cpu'
        
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
