"""
批处理示例

展示如何使用批处理模式处理多个时间步
"""

import torch
import time
from pathlib import Path
from pci_torch import (
    load_gene_config,
    load_beam_config,
)
from pci_torch.forward_model import batch_forward_projection


def main():
    """批处理示例"""
    print("=" * 60)
    print("PyPCI 批处理示例")
    print("=" * 60)
    
    # 设置路径
    base_path = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input"
    
    parameters_file = base_path / "parameters.dat"
    ls_condition_file = base_path / "LS_condition_JT60SA.txt"
    equilibrium_dir = base_path
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载配置
    print("\n加载配置...")
    config = load_gene_config(str(parameters_file), str(equilibrium_dir), device=device)
    beam_config = load_beam_config(str(ls_condition_file))
    
    # 设置简化模式
    if config.PA is None:
        config.PA = torch.tensor([config.major_R * config.L_ref, 0.0], device=device)
        config.GAC = torch.ones(config.nx0+1, 400+1, device=device) * config.trpeps * config.major_R * config.L_ref
    
    # 创建多个时间步的数据
    n_timesteps = 16
    ntheta = 400
    nx = config.nx0
    nz = config.nz0
    
    print(f"\n创建{n_timesteps}个时间步的测试数据...")
    density_list = []
    for i in range(n_timesteps):
        # 每个时间步有不同的随机场
        density = torch.randn(ntheta, nx, nz, device=device) * 0.1
        density_list.append(density)
    
    print(f"  每个场shape: {density_list[0].shape}")
    
    # 批处理模拟
    print(f"\n执行批处理正向模拟 (批大小={n_timesteps})...")
    
    start_time = time.time()
    pci_images = batch_forward_projection(
        density_list,
        config,
        beam_config,
        device=device,
        batch_size=None  # 一次处理全部
    )
    elapsed = time.time() - start_time
    
    print(f"  成功！")
    print(f"  总耗时: {elapsed:.3f} s")
    print(f"  平均每帧: {elapsed/n_timesteps*1000:.2f} ms")
    print(f"  吞吐量: {n_timesteps/elapsed:.1f} 帧/秒")
    print(f"  输出shape: {pci_images.shape}")
    
    # 测试不同批大小的性能
    print(f"\n性能对比 (不同批大小):")
    print("-" * 50)
    
    batch_sizes = [1, 4, 8, 16]
    for bs in batch_sizes:
        if bs > n_timesteps:
            continue
        
        start_time = time.time()
        _ = batch_forward_projection(
            density_list[:bs],
            config,
            beam_config,
            device=device,
            batch_size=bs
        )
        elapsed = time.time() - start_time
        
        print(f"  批大小={bs:2d}: {elapsed/bs*1000:6.2f} ms/帧, "
              f"吞吐量={bs/elapsed:5.1f} 帧/秒")
    
    # 分析结果
    print(f"\n结果分析:")
    print(f"  最小信号: {pci_images.min().item():.6f}")
    print(f"  最大信号: {pci_images.max().item():.6f}")
    print(f"  平均信号: {pci_images.mean().item():.6f}")
    print(f"  标准差: {pci_images.std().item():.6f}")
    
    # 可视化时间序列
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("\n生成可视化...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # 显示前8个时间步
        for i in range(min(8, n_timesteps)):
            im = axes[i].imshow(pci_images[i].cpu().numpy(), 
                               origin='lower', cmap='RdBu_r')
            axes[i].set_title(f'时间步 {i+1}')
            axes[i].set_xlabel('Toroidal')
            axes[i].set_ylabel('Vertical')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        output_path = Path(__file__).parent / "example_batch_output.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  结果已保存到: {output_path}")
        
    except ImportError:
        print("\n  (跳过可视化 - 需要matplotlib)")
    
    print("\n" + "=" * 60)
    print("批处理示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()



