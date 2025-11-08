"""
完整的PCI分析Pipeline

从数据加载到结果分析的完整流程
"""

import sys
from pathlib import Path
import argparse
import torch

# 添加pci_torch到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pci_torch.path_config import PathConfig
from pci_torch.data_loader import (
    load_gene_config_from_parameters,
    load_beam_config,
    separate_torusdata
)
from pci_torch.batch_processor import process_time_series
from pci_torch.fft_analysis import FFT3DAnalyzer
from pci_torch.visualization import PCIVisualizer
from pci_torch.beam_geometry import compute_beam_grid


def main():
    """完整的PCI分析pipeline"""
    parser = argparse.ArgumentParser(description='PyPCI完整分析Pipeline')
    
    # 输入输出路径
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出结果目录')
    parser.add_argument('--data_n', type=int, default=246,
                        help='数据编号')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='PyTorch设备')
    
    # 可选操作
    parser.add_argument('--separate_data', action='store_true',
                        help='是否执行数据分割（如果有TORUSIons_act.dat）')
    parser.add_argument('--time_n', type=int, default=153,
                        help='时间快照数量（用于separate_data）')
    parser.add_argument('--tol_n', type=int, default=101,
                        help='toroidal切片数+1（用于separate_data）')
    
    parser.add_argument('--save_all_figures', action='store_true',
                        help='保存所有图表')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式FFT分析')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PyPCI 完整分析Pipeline")
    print("=" * 80)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建路径配置（统一管理所有路径）
    path_config = PathConfig(
        input_dir=input_dir,
        output_dir=output_dir
    )
    path_config.create_output_dirs()
    
    print("\n路径配置:")
    print(path_config)
    
    # 检查GPU
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU")
            args.device = 'cpu'
        else:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("使用CPU")
    
    # 步骤0: 可选的数据预处理
    if args.separate_data:
        print("\n" + "=" * 80)
        print("步骤0: 数据预处理 - 分割TORUSIons_act.dat")
        print("=" * 80)
        
        torus_file = input_dir / 'TORUSIons_act.dat'
        if torus_file.exists():
            print(f"分割文件: {torus_file}")
            separate_torusdata(
                str(torus_file),
                str(input_dir),
                args.time_n,
                args.tol_n
            )
        else:
            print(f"未找到文件: {torus_file}，跳过")
    
    # 步骤1: 加载配置
    print("\n" + "=" * 80)
    print("步骤1: 加载配置")
    print("=" * 80)
    
    parameters_file = input_dir / 'parameters.dat'
    ls_condition_file = input_dir / 'LS_condition_JT60SA.txt'
    
    print(f"\n加载GENE配置: {parameters_file}")
    config = load_gene_config_from_parameters(
        str(parameters_file),
        str(input_dir),
        device=args.device
    )
    print(f"  网格: nx0={config.nx0}, nky0={config.nky0}, nz0={config.nz0}")
    print(f"  参数: q0={config.q0}, shat={config.shat}")
    
    print(f"\n加载光束配置: {ls_condition_file}")
    beam_config = load_beam_config(str(ls_condition_file))
    print(f"  检测器: {beam_config.n_detectors_v} x {beam_config.n_detectors_t}")
    print(f"  光束采样点: {beam_config.n_beam_points}")
    
    # 步骤2: 批量处理时间序列
    print("\n" + "=" * 80)
    print("步骤2: 批量处理时间序列")
    print("=" * 80)
    
    pout1, pout2 = process_time_series(
        str(input_dir),
        args.data_n,
        config,
        beam_config,
        var=4,
        device=args.device,
        save_results=True,
        output_dir=str(output_dir / 'mat')
    )
    
    # 步骤3: 可视化几个代表性的时间点
    if args.save_all_figures:
        print("\n" + "=" * 80)
        print("步骤3: 可视化代表性时间点")
        print("=" * 80)
        
        visualizer = PCIVisualizer(config)
        
        # 可视化几个时间点（使用PathConfig）
        fig_dir = path_config.figures_dir
        
        # 生成空间网格
        wid1 = beam_config.width_vertical
        wid2 = beam_config.width_toroidal
        div1 = beam_config.div_vertical
        div2 = beam_config.div_toroidal
        
        y_coords = torch.linspace(-wid2/2, wid2/2, 2*div2+1)
        x_coords = torch.linspace(-wid1/2, wid1/2, 2*div1+1)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        xx = torch.fliplr(xx)
        
        # 可视化几个时间快照
        time_indices = [i for i in range(0, pout2.shape[2], max(1, pout2.shape[2]//10))]
        for t_idx in time_indices:
            time_point = t_idx * 0.01
            visualizer.plot_detector_contour(
                pout2[:, :, t_idx],
                xx, yy,
                time_point,
                save_path=str(fig_dir / f'fig3_detector_signal_t{time_point:.2f}.png')
            )
        
        # 绘制光束几何
        try:
            beam_grid = compute_beam_grid(beam_config, device=args.device)
            visualizer.plot_beam_geometry_3d(
                beam_grid,
                save_path=str(fig_dir / 'fig1_beam_geometry_t0.00.png')
            )
        except Exception as e:
            print(f"绘制光束几何失败: {e}")
    
    # 步骤4: FFT分析
    print("\n" + "=" * 80)
    print("步骤4: FFT频谱分析")
    print("=" * 80)
    
    # FFT分析器使用PathConfig（所有路径由配置管理）
    analyzer = FFT3DAnalyzer(config, args.data_n, path_config)
    analyzer.load_time_series(device='cpu')  # FFT在CPU上可能更快
    analyzer.compute_3d_fft()
    
    if args.interactive:
        print("\n进入交互式分析模式...")
        while True:
            print("\n" + "-" * 60)
            print("1: kx-ky spectrum")
            print("2: ω-ky spectrum")
            print("3: ω-kx spectrum")
            print("4: 时间快照")
            print("5: 2D FFT (时间平均)")
            print("0: 退出")
            print("-" * 60)
            
            try:
                mode = int(input("选择模式: "))
                
                if mode == 0:
                    break
                elif mode == 1:
                    print(f"频率索引范围: 0 - {len(analyzer.f)-1}")
                    f_idx = int(input("选择频率索引: "))
                    analyzer.analyze_mode1_kxky(f_idx)
                elif mode == 2:
                    print(f"kx索引范围: 0 - {len(analyzer.kx)-1}")
                    kx_idx = int(input("选择kx索引: "))
                    analyzer.analyze_mode2_wky(kx_idx)
                elif mode == 3:
                    print(f"ky索引范围: 0 - {len(analyzer.ky)-1}")
                    ky_idx = int(input("选择ky索引: "))
                    analyzer.analyze_mode3_wkx(ky_idx)
                elif mode == 4:
                    Nt = analyzer.realSpaceData.shape[2]
                    print(f"时间索引范围: 0 - {Nt-1}")
                    t_idx = int(input("选择时间索引: "))
                    analyzer.analyze_mode4_snapshot(t_idx)
                elif mode == 5:
                    analyzer.analyze_mode5_2dfft()
                else:
                    print("无效的选项")
            except (ValueError, KeyboardInterrupt):
                break
    elif args.save_all_figures:
        print("\n批量生成FFT分析图表...")
        fft_dir = output_dir / 'fft_analysis'
        analyzer.analyze_all_modes(str(fft_dir))
    
    # 步骤5: 生成总结报告
    print("\n" + "=" * 80)
    print("步骤5: 生成分析报告")
    print("=" * 80)
    
    report_path = output_dir / 'ANALYSIS_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PyPCI 分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"数据编号: {args.data_n}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("配置信息\n")
        f.write("=" * 80 + "\n")
        f.write(f"网格: nx0={config.nx0}, nky0={config.nky0}, nz0={config.nz0}\n")
        f.write(f"几何参数: q0={config.q0}, shat={config.shat}, trpeps={config.trpeps}\n")
        f.write(f"归一化: B_ref={config.B_ref} T, rho_ref={config.rho_ref:.6f} m\n\n")
        
        f.write(f"检测器: {beam_config.n_detectors_v} x {beam_config.n_detectors_t}\n")
        f.write(f"光束采样点: {beam_config.n_beam_points}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("处理结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"处理的时间快照数: {pout2.shape[2]}\n")
        f.write(f"pout1 shape: {pout1.shape}\n")
        f.write(f"pout2 shape: {pout2.shape}\n")
        f.write(f"信号范围: [{pout2.min():.4e}, {pout2.max():.4e}]\n\n")
        
        if analyzer.Amp is not None:
            f.write("=" * 80 + "\n")
            f.write("FFT分析\n")
            f.write("=" * 80 + "\n")
            f.write(f"kx范围: [{analyzer.kx.min():.4f}, {analyzer.kx.max():.4f}]\n")
            f.write(f"ky范围: [{analyzer.ky.min():.4f}, {analyzer.ky.max():.4f}]\n")
            f.write(f"f范围: [{analyzer.f.min():.4f}, {analyzer.f.max():.4f}]\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("输出文件\n")
        f.write("=" * 80 + "\n")
        f.write(f"MAT文件目录: {output_dir / 'mat'}\n")
        if args.save_all_figures:
            f.write(f"图表目录: {output_dir / 'figures'}\n")
            f.write(f"FFT分析目录: {output_dir / 'fft_analysis'}\n")
    
    print(f"报告已保存到: {report_path}")
    
    print("\n" + "=" * 80)
    print("Pipeline执行完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

