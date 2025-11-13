#!/usr/bin/env python3
"""
快速GZC测试脚本 - 验证修正效果
"""

import sys
from pathlib import Path

# 添加pci_torch到路径
sys.path.insert(0, str(Path(__file__).parent))

from pci_torch.data_loader import load_gene_config_from_parameters

def quick_gzc_test():
    """快速测试GZC数据加载"""
    print("=" * 60)
    print("快速GZC测试 - 验证修正效果")
    print("=" * 60)
    
    # 设置路径
    input_dir = Path(__file__).parent.parent / "Data" / "python_input"
    parameters_file = input_dir / "parameters.dat"
    
    if not parameters_file.exists():
        print(f"错误: 找不到parameters.dat文件: {parameters_file}")
        return
    
    print(f"加载配置文件: {parameters_file}")
    
    try:
        # 加载配置
        gene_config = load_gene_config_from_parameters(
            str(parameters_file),
            str(input_dir),
            device='cuda'
        )
        
        print("✅ 配置加载成功")
        
        # 检查GZC数据
        if gene_config.GZC is None:
            print("❌ GZC数据未加载")
            return
            
        gzc_tensor = gene_config.GZC
        grc_tensor = gene_config.GRC
        
        print(f"\n=== 修正后的Python GZC数据 ===")
        print(f"GZC Tensor 维度: {gzc_tensor.shape}")
        print(f"GZC 整体最小值: {gzc_tensor.min():.6f}")
        print(f"GZC 整体最大值: {gzc_tensor.max():.6f}")
        
        # 提取最后一行（用于绘图）
        gzc_last_row = gzc_tensor[-1, :].cpu().numpy()
        grc_last_row = grc_tensor[-1, :].cpu().numpy()
        
        print(f"GZC 最后一行最小值: {gzc_last_row.min():.6f}")
        print(f"GZC 最后一行最大值: {gzc_last_row.max():.6f}")
        print(f"GZC 最后一行范围: {gzc_last_row.max() - gzc_last_row.min():.6f}")
        
        print(f"\nGRC 维度: {grc_tensor.shape}")
        print(f"GRC 最后一行范围: {grc_last_row.max() - grc_last_row.min():.6f}")
        
        # 检查是否修复成功
        z_range = gzc_last_row.max() - gzc_last_row.min()
        if z_range > 1.0:
            print(f"\n✅ 修复成功！Z轴范围: {z_range:.3f}m (>1.0m)")
        else:
            print(f"\n❌ 仍需进一步修复。Z轴范围: {z_range:.3f}m (<1.0m)")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_gzc_test()

