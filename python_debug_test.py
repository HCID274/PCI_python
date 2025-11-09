#!/usr/bin/env python3
"""
Python数据排列验证测试脚本
验证核心reshape操作后的关键位置数据值
"""

import sys
import torch
import numpy as np
import scipy.io
from pathlib import Path
import os

# 添加路径
sys.path.insert(0, '/work/DTMP/lhqing/PCI/code/pyPCI')

from pci_torch import (
    GENEConfig, 
    BeamConfig,
    load_gene_config_from_parameters,
    load_beam_config,
    forward_projection
)
from pci_torch.data_loader import fread_data_s, generate_timedata

def main():
    print("Python数据排列验证测试")
    print("=" * 60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 关键修正：使用MATLAB的相同数据目录以确保数据一致性
    input_dir = '/work/DTMP/lhqing/PCI/Data/matlab_input/301'  # 使用MATLAB的数据目录
    parameters_file = Path(input_dir) / 'parameters.dat'
    beam_config_file = Path('/work/DTMP/lhqing/PCI/Data/sample/input') / 'LS_condition_JT60SA.txt'  # beam config可以使用原来的
    
    # 检查文件是否存在
    if not parameters_file.exists():
        print(f"错误: 参数文件不存在 {parameters_file}")
        return
    if not beam_config_file.exists():
        print(f"错误: 光束配置文件不存在 {beam_config_file}")
        return
        
    print(f"参数文件: {parameters_file}")
    print(f"光束配置文件: {beam_config_file}")
    
    try:
        # 加载配置
        print("\n加载GENE配置...")
        config = load_gene_config_from_parameters(
            str(parameters_file),
            str(input_dir),  # 使用数据目录
            device=device
        )
        
        print("加载光束配置...")
        beam_config = load_beam_config(str(beam_config_file))
        
        # 更新config的派生参数
        config.compute_derived_params()
        
        print(f"GENE配置加载完成")
        print(f"  nx0: {config.nx0}")
        print(f"  KYMt: {config.KYMt}")
        print(f"  KZMt: {config.KZMt}")
        print(f"  PA: {config.PA}")
        print(f"  GAC: {config.GAC is not None}")
        if config.GAC is not None:
            print(f"    GAC shape: {config.GAC.shape}")
            print(f"    GAC range: [{config.GAC.min():.6f}, {config.GAC.max():.6f}]")
        else:
            print("    警告: GAC数据未加载！")
        
        print(f"光束配置加载完成")
        print(f"  div_vertical: {beam_config.div_vertical}")
        print(f"  div_toroidal: {beam_config.div_toroidal}")
        print(f"  div_beam: {beam_config.div_beam}")
        
        # 计算目标维度（与MATLAB对应）
        div1_2 = 2 * beam_config.div_vertical + 1
        div2_2 = 2 * beam_config.div_toroidal + 1
        divls_2 = beam_config.div_beam + 1
        
        print(f"\n目标reshape维度: ({div1_2}, {div2_2}, {divls_2})")
        print(f"总元素数: {div1_2 * div2_2 * divls_2}")
        
        # 加载时间数据
        time_t = 98.07
        time_int = int(time_t * 100)
        binary_file = Path(input_dir) / f'{time_int:08d}.dat'
        
        if not binary_file.exists():
            text_file = Path(input_dir) / f'TORUSIons_act_{time_int}.dat'
            if text_file.exists():
                print(f"生成二进制文件...")
                binary_file = generate_timedata(config, str(text_file), time_t, str(input_dir))
            else:
                print(f"错误: 数据文件不存在")
                return
        
        # 读取密度场
        print(f"\n读取密度场数据...")
        density_3d = fread_data_s(config, str(binary_file), device=device)
        print(f"密度场shape: {density_3d.shape}")
        
        # 执行PCI正向投影
        print("\n执行PCI正向投影...")
        pci_result = forward_projection(
            density_3d,
            config,
            beam_config,
            device=device,
            return_line_integral=True  # 返回完整的line integral数据
        )
        
        print(f"PCI结果shape: {pci_result.shape}")
        
        # 转换为numpy并验证维度
        pci_numpy = pci_result.cpu().numpy()
        print(f"PCI结果numpy shape: {pci_numpy.shape}")
        
        # 验证是否与期望维度匹配
        expected_shape = (div1_2, div2_2, divls_2)
        if pci_numpy.shape == expected_shape:
            print(f"✓ 维度匹配: {pci_numpy.shape}")
        else:
            print(f"✗ 维度不匹配: 实际 {pci_numpy.shape}, 期望 {expected_shape}")
            # 尝试调整维度
            if pci_numpy.size == div1_2 * div2_2 * divls_2:
                print("尝试重新reshape...")
                pci_numpy = pci_numpy.reshape(expected_shape, order='C')  # 默认行主序
                print(f"重新reshape后: {pci_numpy.shape}")
            else:
                print("无法reshape，元素数量不匹配")
                return
        
        # 显示Python版本的关键位置值（Python是0基索引）
        print(f"\n=== Python pci_image 关键位置值 ===")
        print(f"pci_image[0,0,0] = {pci_numpy[0,0,0]:.10f}")
        print(f"pci_image[1,0,0] = {pci_numpy[1,0,0]:.10f}  (第一个维度变化)")
        print(f"pci_image[0,1,0] = {pci_numpy[0,1,0]:.10f}  (第二个维度变化)")
        print(f"pci_image[0,0,1] = {pci_numpy[0,0,1]:.10f}  (第三个维度变化)")
        
        # 显示更多位置值用于对比
        print(f"\n=== Python pci_image 更多位置值 ===")
        print(f"pci_image[0,0,2] = {pci_numpy[0,0,2]:.10f}")
        print(f"pci_image[2,0,0] = {pci_numpy[2,0,0]:.10f}")
        print(f"pci_image[0,2,0] = {pci_numpy[0,2,0]:.10f}")
        
        # 验证积分结果
        pci_integrated = np.sum(pci_numpy, axis=2)
        print(f"\n=== 积分结果信息 ===")
        print(f"积分后shape: {pci_integrated.shape}")
        print(f"积分范围: [{pci_integrated.min():.6f}, {pci_integrated.max():.6f}]")
        
        # 保存Python数据
        output_file = '/work/DTMP/lhqing/PCI/code/python_debug_data.mat'
        scipy.io.savemat(output_file, {
            'pci_image': pci_numpy,
            'pci_integrated': pci_integrated,
            'dimensions': [div1_2, div2_2, divls_2],
            'device': device
        })
        print(f"\nPython数据已保存到: {output_file}")
        
        # 尝试读取MATLAB数据进行对比
        matlab_file = '/work/DTMP/lhqing/PCI/code/matlab_debug_data.mat'
        if Path(matlab_file).exists():
            print(f"\n=== MATLAB数据对比 ===")
            matlab_data = scipy.io.loadmat(matlab_file)
            pout1 = matlab_data['pout1']
            
            print(f"MATLAB pout1 维度: {pout1.shape}")
            print(f"Python pci_image 维度: {pci_numpy.shape}")
            
            # 关键位置对比
            print(f"\n关键位置值对比:")
            print(f"MATLAB pout1(1,1,1) vs Python pci_image[0,0,0]")
            print(f"  MATLAB: {pout1[0,0,0]:.10f}")
            print(f"  Python: {pci_numpy[0,0,0]:.10f}")
            print(f"  差异: {abs(pout1[0,0,0] - pci_numpy[0,0,0]):.2e}")
            
            print(f"\nMATLAB pout1(2,1,1) vs Python pci_image[1,0,0]")
            print(f"  MATLAB: {pout1[1,0,0]:.10f}")
            print(f"  Python: {pci_numpy[1,0,0]:.10f}")
            print(f"  差异: {abs(pout1[1,0,0] - pci_numpy[1,0,0]):.2e}")
            
            print(f"\nMATLAB pout1(1,2,1) vs Python pci_image[0,1,0]")
            print(f"  MATLAB: {pout1[0,1,0]:.10f}")
            print(f"  Python: {pci_numpy[0,1,0]:.10f}")
            print(f"  差异: {abs(pout1[0,1,0] - pci_numpy[0,1,0]):.2e}")
            
            print(f"\nMATLAB pout1(1,1,2) vs Python pci_image[0,0,1]")
            print(f"  MATLAB: {pout1[0,0,1]:.10f}")
            print(f"  Python: {pci_numpy[0,0,1]:.10f}")
            print(f"  差异: {abs(pout1[0,0,1] - pci_numpy[0,0,1]):.2e}")
            
            # 检查是否匹配
            max_diff = np.max(np.abs(pout1 - pci_numpy))
            print(f"\n最大差异: {max_diff:.2e}")
            
            if max_diff < 1e-10:
                print("✓ 数据匹配！")
            else:
                print("✗ 数据不匹配！存在数据排列问题。")
                
        else:
            print(f"\n警告: MATLAB数据文件不存在 {matlab_file}")
            print("请先运行MATLAB测试脚本")
        
        print(f"\n=== Python测试完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

# 在main函数末尾添加数据对比
print("\n=== 数据源对比 ===")
print("MATLAB使用目录:", "/work/DTMP/lhqing/PCI/Data/matlab_input/301/")
print("Python使用目录:", "/work/DTMP/lhqing/PCI/Data/sample/input/")

# 检查两个目录的equdata_BZ是否相同
matlab_bz = "/work/DTMP/lhqing/PCI/Data/matlab_input/301/equdata_BZ"
python_bz = "/work/DTMP/lhqing/PCI/Data/sample/input/equdata_BZ"

print(f"\nequdata_BZ文件大小对比:")
print(f"MATLAB: {os.path.getsize(matlab_bz)} bytes")
print(f"Python: {os.path.getsize(python_bz)} bytes")

# 读取少量数据进行对比
with open(matlab_bz, 'rb') as f:
    matlab_data = np.fromfile(f, dtype=np.float64, count=10)
    
with open(python_bz, 'rb') as f:
    python_data = np.fromfile(f, dtype=np.float64, count=10)

print(f"\nequdata_BZ前10个float64数据对比:")
print(f"MATLAB: {matlab_data}")
print(f"Python: {python_data}")
print(f"数据是否相同: {np.allclose(matlab_data, python_data)}")

# 检查config中的实际GAC值
try:
    print(f"\nPython Config GAC 统计:")
    print(f"GAC 形状: {config.GAC.shape}")
    print(f"GAC 最小值: {torch.min(config.GAC).item():.6f}")
    print(f"GAC 最大值: {torch.max(config.GAC).item():.6f}")
    print(f"GAC 平均值: {torch.mean(config.GAC).item():.6f}")

    # 显示GAC的边界值（用于与MATLAB对比）
    GAC_numpy = config.GAC.cpu().numpy()
    print(f"\nGAC边界值分析:")
    print(f"最后一层径向边界 (MATLAB调试中的1.14-1.15范围):")
    last_layer = GAC_numpy[-1, :]
    print(f"最后一层范围: [{np.min(last_layer):.6f}, {np.max(last_layer):.6f}]")
    print(f"最后一层前10个值: {last_layer[:10]}")
except Exception as e:
    print(f"无法访问config.GAC: {e}")
    # 尝试从函数中获取
    try:
        from pci_torch.data_loader import load_gene_config
        test_config = load_gene_config(
            str(Path("/work/DTMP/lhqing/PCI/Data/sample/input")),
            device='cpu'
        )
        print(f"重新加载的config GAC统计:")
        print(f"GAC 形状: {test_config.GAC.shape}")
        print(f"GAC 最小值: {torch.min(test_config.GAC).item():.6f}")
        print(f"GAC 最大值: {torch.max(test_config.GAC).item():.6f}")
        print(f"GAC 平均值: {torch.mean(test_config.GAC).item():.6f}")
        
        GAC_numpy = test_config.GAC.cpu().numpy()
        last_layer = GAC_numpy[-1, :]
        print(f"最后一层范围: [{np.min(last_layer):.6f}, {np.max(last_layer):.6f}]")
        print(f"最后一层前10个值: {last_layer[:10]}")
    except Exception as e2:
        print(f"重新加载也失败: {e2}")

if __name__ == "__main__":
    main()
