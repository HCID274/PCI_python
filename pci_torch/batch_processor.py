"""
批量时间序列处理

对应MATLAB的LSview_com_loop.m
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import scipy.io
import glob
import re

from .config import GENEConfig, BeamConfig
from .forward_model import forward_projection
from .data_loader import fread_data_s, generate_timedata


def find_time_data_files(
    input_dir: str,
    pattern: str = 'TORUSIons_act_*.dat'
) -> List[Tuple[str, float]]:
    """
    查找所有时间数据文件并按时间排序
    
    Args:
        input_dir: 输入目录
        pattern: 文件名模式
    
    Returns:
        [(file_path, time_value), ...] 按时间排序的列表
    """
    file_pattern = str(Path(input_dir) / pattern)
    files = glob.glob(file_pattern)
    
    # 提取时间值
    file_time_pairs = []
    for file_path in files:
        filename = Path(file_path).name
        # 从文件名提取时间：TORUSIons_act_XXXX.dat
        match = re.search(r'TORUSIons_act_(\d+)\.dat', filename)
        if match:
            time_int = int(match.group(1))
            time_value = time_int / 100.0
            file_time_pairs.append((file_path, time_value))
    
    # 按时间排序
    file_time_pairs.sort(key=lambda x: x[1])
    
    return file_time_pairs


def process_time_series(
    input_dir: str,
    data_n: int,
    config: GENEConfig,
    beam_config: BeamConfig,
    var: int = 4,
    device: str = 'cuda',
    save_results: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量处理所有时间快照
    
    对应LSview_com_loop.m的功能
    
    Args:
        input_dir: 输入数据目录
        data_n: 数据编号
        config: GENE配置对象
        beam_config: 光束配置
        var: 变量类型（4=密度）
        device: PyTorch设备
        save_results: 是否保存MAT文件
        output_dir: 输出目录（None则使用input_dir/result_PCI/mat/）
    
    Returns:
        pout1: LocalCross-Section数据 (n_det_v, n_det_t, n_beam, time_n)
        pout2: IntegratedSignal数据 (n_det_v, n_det_t, time_n)
    """
    print(f"=" * 60)
    print(f"批量处理时间序列数据")
    print(f"输入目录: {input_dir}")
    print(f"数据编号: {data_n}")
    print(f"=" * 60)
    
    # 查找所有时间数据文件
    file_time_pairs = find_time_data_files(input_dir)
    time_n = len(file_time_pairs)
    
    if time_n == 0:
        raise ValueError(f"在 {input_dir} 中未找到时间数据文件")
    
    print(f"\n找到 {time_n} 个时间快照")
    print(f"时间范围: {file_time_pairs[0][1]:.2f} - {file_time_pairs[-1][1]:.2f}")
    
    # 获取检测器和光束维度
    n_det_v = beam_config.n_detectors_v
    n_det_t = beam_config.n_detectors_t
    n_beam = beam_config.n_beam_points
    
    # 初始化输出数组
    pout1 = torch.zeros(n_det_v, n_det_t, n_beam, time_n, device=device)
    pout2 = torch.zeros(n_det_v, n_det_t, time_n, device=device)
    
    # 处理每个时间点
    for i, (file_path, time_value) in enumerate(file_time_pairs):
        print(f"\n[{i+1}/{time_n}] 处理 t={time_value:.2f} ({Path(file_path).name})")
        
        try:
            # 读取密度数据
            if time_value == 0:
                print(f"  跳过 t=0")
                continue
            
            # 检查是否需要生成二进制数据
            binary_file = str(Path(input_dir) / f'{int(time_value * 100):08d}.dat')
            if not Path(binary_file).exists():
                print(f"  生成二进制文件...")
                binary_file = generate_timedata(config, file_path, time_value, input_dir)
            
            # 更新config的派生参数
            config.compute_derived_params()
            
            # 读取密度场
            density_3d = fread_data_s(config, binary_file, device=device)
            print(f"  密度场shape: {density_3d.shape}")
            
            # 执行PCI正向投影
            pci_result = forward_projection(
                density_3d,
                config,
                beam_config,
                device=device,
                return_line_integral=True  # 返回完整的line integral数据
            )
            # pci_result shape: (n_det_v, n_det_t, n_beam)
            
            # 存储结果
            pout1[:, :, :, i] = pci_result.cpu()
            
            # 计算积分信号（沿光束方向求和）
            pout2[:, :, i] = torch.sum(pci_result, dim=-1).cpu()
            
            print(f"  完成！信号范围: [{pout2[:, :, i].min():.4f}, {pout2[:, :, i].max():.4f}]")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果 - 与MATLAB LSview_com_loop.m (第87-90行)严格一致
    if save_results:
        if output_dir is None:
            output_dir = str(Path(input_dir) / 'result_PCI' / 'mat')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # MATLAB: dat1path = sprintf('%s%s%s%d%s',dataC.indir,'result_PCI/mat/','LocalCross-Section_',data_n,'_overall.mat')
        # MATLAB: save(dat1path, 'pout1','-v7.3');
        pout1_path = Path(output_dir) / f'LocalCross-Section_{data_n}_overall.mat'
        print(f"\n保存LocalCross-Section到: {pout1_path}")
        try:
            # 使用v7.3格式（HDF5）- 与MATLAB '-v7.3'参数一致
            import h5py
            with h5py.File(str(pout1_path), 'w') as f:
                f.create_dataset('pout1', data=pout1.numpy(), compression='gzip')
            print(f"  使用MATLAB v7.3兼容格式保存（HDF5压缩）")
        except ImportError:
            # 如果没有h5py，使用标准格式并标记兼容性
            scipy.io.savemat(
                str(pout1_path),
                {'pout1': pout1.numpy()},
                do_compression=True
            )
            print(f"  警告: 使用MATLAB v7格式保存（建议安装h5py以支持v7.3）")
        
        # MATLAB: dat2path = sprintf('%s%s%s%d%s',dataC.indir,'result_PCI/mat/','IntegratedSignal_',data_n,'_overall.mat')
        # MATLAB: save(dat2path, 'pout2');
        pout2_path = Path(output_dir) / f'IntegratedSignal_{data_n}_overall.mat'
        print(f"保存IntegratedSignal到: {pout2_path}")
        scipy.io.savemat(
            str(pout2_path),
            {'pout2': pout2.numpy()},
            do_compression=False  # 与MATLAB一致，不压缩pout2
        )
    
    print(f"\n" + "=" * 60)
    print(f"批量处理完成！")
    print(f"  处理了 {time_n} 个时间点")
    print(f"  pout1 shape: {pout1.shape}")
    print(f"  pout2 shape: {pout2.shape}")
    print(f"=" * 60)
    
    return pout1, pout2


def process_single_timepoint(
    input_dir: str,
    data_n: int,
    time_t: float,
    config: GENEConfig,
    beam_config: BeamConfig,
    var: int = 4,
    fver: float = 5.0,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    处理单个时间点
    
    Args:
        input_dir: 输入数据目录
        data_n: 数据编号
        time_t: 时间值
        config: GENE配置
        beam_config: 光束配置
        var: 变量类型
        fver: 格式版本 (3.5=R5F格式, 5.0=GENE格式)
        device: PyTorch设备
    
    Returns:
        pout1: LocalCross-Section数据
        pout2: IntegratedSignal数据
    """
    print(f"处理单个时间点: t={time_t:.2f} (FVER {fver})")
    
    # 根据FVER设置文件名模式
    if fver == 3.5:
        # FVER 3.5: 纯数字文件名
        time_int = int(time_t * 100)
        text_file = str(Path(input_dir) / f'{time_int:08d}.dat')
        binary_file = str(Path(input_dir) / f'{time_int:08d}.dat')
    else:
        # FVER 5.0: GENE格式文件名
        time_int = int(time_t * 100)
        text_file = str(Path(input_dir) / f'TORUSIons_act_{time_int}.dat')
        binary_file = str(Path(input_dir) / f'{time_int:08d}.dat')
    
    if not Path(binary_file).exists():
        if Path(text_file).exists():
            print(f"生成二进制文件...")
            binary_file = generate_timedata(config, text_file, time_t, input_dir)
        else:
            raise FileNotFoundError(f"找不到数据文件: {text_file}")
    
    # 更新config
    config.compute_derived_params()
    
    # 读取密度场
    density_3d = fread_data_s(config, binary_file, device=device)
    
    # 执行PCI正向投影
    pout1 = forward_projection(
        density_3d,
        config,
        beam_config,
        device=device,
        return_line_integral=True
    )
    
    # 计算积分信号
    pout2 = torch.sum(pout1, dim=-1)
    
    return pout1.cpu(), pout2.cpu()

