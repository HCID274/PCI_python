#!/usr/bin/env python
"""
数据预处理脚本
分割TORUSIons_act.dat文件并生成二进制数据
使用配置文件管理路径
"""
import sys
import os
import json
sys.path.insert(0, '/work/DTMP/lhqing/PCI/code/pyPCI')

from pci_torch.data_loader import separate_torusdata, load_gene_config_from_parameters
from pci_torch.path_config import PathConfig
from pathlib import Path

def main():
    print("="*70)
    print("PyPCI 数据预处理 (使用配置文件)")
    print("="*70)
    
    # 从配置文件加载路径配置
    try:
        path_config = PathConfig.from_config_file()
        print(f"✓ 配置文件加载成功")
        print(f"  输入目录: {path_config.input_dir}")
        print(f"  输出目录: {path_config.output_dir}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return 1
    
    # 加载配置
    print("\n步骤1: 加载配置...")
    try:
        config = load_gene_config_from_parameters(
            str(path_config.parameters_file),
            str(path_config.input_dir),
            device='cpu'  # 预处理使用CPU
        )
        print(f"  ✓ 配置加载成功: nx0={config.nx0}, nky0={config.nky0}, nz0={config.nz0}")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return 1
    
    # 分割TORUSIons_act.dat
    print("\n步骤2: 分割TORUSIons_act.dat...")
    torus_file = path_config.torus_data_file
    
    if not torus_file.exists():
        print(f"  ✗ 未找到文件: {torus_file}")
        return 1
    
    file_size_gb = torus_file.stat().st_size / (1024**3)
    print(f"  文件大小: {file_size_gb:.2f} GB")
    
    # 使用配置文件中的参数
    # 读取配置文件中的处理参数
    config_file_path = Path(__file__).parent / 'config' / 'paths.json'
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
    except Exception as e:
        print(f"  ✗ 配置文件读取失败: {e}")
        return 1
    
    processing_config = full_config.get('processing', {})
    preprocess_config = full_config.get('preprocess', {})
    
    time_n = processing_config.get('time_n', 20000)
    tol_n = processing_config.get('tol_n', 10)
    
    print(f"  time_n={time_n}, tol_n={tol_n}")
    print("  开始分割...")
    
    try:
        time_files = separate_torusdata(
            str(torus_file), 
            str(path_config.input_dir), 
            time_n, 
            tol_n
        )
        print(f"  ✓ 成功分割为 {len(time_files)} 个文件")
        if time_files:
            print(f"  文件示例: {time_files[0]}")
    except Exception as e:
        print(f"  ✗ 分割失败: {e}")
        return 1
    
    print("\n"+"="*70)
    print("预处理完成！")
    print("="*70)
    
    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

