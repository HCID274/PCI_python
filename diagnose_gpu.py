#!/usr/bin/env python3
"""
GPU兼容性诊断脚本
用于确定AMD MI300A是否需要升级或降级PyTorch/ROCm
"""

import torch
import subprocess
import sys

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_rocm_version():
    """检查系统ROCm版本"""
    try:
        result = subprocess.run(['rocm-smi', '--showdriverversion'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # 尝试从环境变量获取
            result = subprocess.run(['cat', '/opt/rocm/.info/version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else "未知"
    except Exception as e:
        return f"无法检测: {e}"

def check_gpu_arch():
    """检查GPU架构"""
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'gfx' in line.lower() and 'name' in line.lower():
                    # 查找架构信息
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if 'gfx' in lines[j]:
                            return lines[j].strip()
        return "未知"
    except Exception as e:
        return f"无法检测: {e}"

def main():
    print_section("AMD GPU兼容性诊断报告")
    
    # 1. PyTorch信息
    print_section("1. PyTorch配置")
    print(f"PyTorch版本:     {torch.__version__}")
    print(f"ROCm支持:        {'是' if torch.cuda.is_available() else '否'}")
    print(f"构建时ROCm版本:  {torch.version.hip if hasattr(torch.version, 'hip') else '未知'}")
    
    # 2. GPU信息
    print_section("2. GPU硬件信息")
    if torch.cuda.is_available():
        print(f"检测到GPU数量:   {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称:          {torch.cuda.get_device_name(i)}")
            print(f"  显存:          {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # 尝试获取架构信息
            props = torch.cuda.get_device_properties(i)
            if hasattr(props, 'gcnArchName'):
                print(f"  GCN架构:       {props.gcnArchName}")
    else:
        print("未检测到GPU或GPU不可用")
    
    # 3. 系统ROCm版本
    print_section("3. 系统ROCm版本")
    rocm_version = check_rocm_version()
    print(f"ROCm版本: {rocm_version}")
    
    gpu_arch = check_gpu_arch()
    print(f"GPU架构信息: {gpu_arch}")
    
    # 4. 简单的GPU功能测试
    print_section("4. GPU功能测试")
    
    if torch.cuda.is_available():
        try:
            print("测试1: 创建tensor...")
            x = torch.randn(10, 10, device='cuda')
            print("  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
        
        try:
            print("测试2: 基础运算...")
            y = torch.randn(10, 10, device='cuda')
            z = x + y
            print("  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
        
        try:
            print("测试3: grid_sample (复杂操作)...")
            input_tensor = torch.randn(1, 1, 4, 4, device='cuda')
            grid = torch.randn(1, 2, 2, 2, device='cuda')
            output = torch.nn.functional.grid_sample(input_tensor, grid, align_corners=False)
            print("  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    else:
        print("GPU不可用，跳过测试")
    
    # 5. 给出建议
    print_section("5. 诊断结论和建议")
    
    pytorch_version = torch.__version__
    major_minor = '.'.join(pytorch_version.split('.')[:2])
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        if 'MI300' in gpu_name or 'gfx942' in gpu_arch:
            print("\n【GPU识别】: AMD Instinct MI300A (gfx942架构)")
            print("\n【关键问题】: MI300A是2023年底发布的新架构GPU")
            
            if 'rocm5.7' in pytorch_version.lower() or '2.3' in pytorch_version:
                print("\n【当前配置】:")
                print(f"  - PyTorch: {pytorch_version}")
                print(f"  - ROCm: 5.7 (从PyTorch版本推断)")
                
                print("\n【问题分析】:")
                print("  ✗ ROCm 5.7 发布于2023年中期，早于MI300A发布")
                print("  ✗ 对gfx942架构支持不完整/不稳定")
                print("  ✗ 部分GPU kernel可能未针对gfx942优化")
                
                print("\n【解决方案】: ⬆️ 需要【升级】而非降级!")
                print("\n  推荐配置:")
                print("  ┌─────────────────────────────────────────┐")
                print("  │  选项1 (推荐): ROCm 6.1+ & PyTorch 2.4+ │")
                print("  │  - ROCm 6.1.0 或更高版本                │")
                print("  │  - PyTorch 2.4.0 或更高版本             │")
                print("  │  - 完整支持MI300A gfx942架构            │")
                print("  │                                         │")
                print("  │  选项2 (稳定): ROCm 6.0 & PyTorch 2.3+ │")
                print("  │  - ROCm 6.0.0                           │")
                print("  │  - PyTorch 2.3.1 或 2.4.0               │")
                print("  │  - 基本支持MI300A                       │")
                print("  └─────────────────────────────────────────┘")
                
                print("\n  具体步骤:")
                print("  1. 联系系统管理员升级ROCm到6.0+")
                print("  2. 重新安装PyTorch:")
                print("     pip uninstall torch torchvision")
                print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1")
                print("     # 或使用系统对应的ROCm版本")
                
                print("\n【为什么不能降级?】")
                print("  ✗ ROCm 5.6及更早版本: 完全不支持gfx942")
                print("  ✗ PyTorch 2.2及更早版本: 没有MI300A的kernel")
                print("  ✗ 降级只会使问题更严重")
                
            else:
                print(f"\n当前PyTorch版本: {pytorch_version}")
                print("需要更多信息来判断，请检查是否存在GPU错误")
    else:
        print("\nGPU不可用，请检查:")
        print("  1. ROCm驱动是否正确安装")
        print("  2. PyTorch是否为ROCm版本")
        print("  3. 环境变量是否正确设置")
    
    print_section("诊断完成")
    print()

if __name__ == '__main__':
    main()



