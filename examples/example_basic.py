"""
基础使用示例

展示如何加载配置和执行正向模拟
"""

import torch
from pathlib import Path
from pci_torch import (
    load_gene_config,
    load_beam_config,
    forward_projection
)


def main():
    """基础示例"""
    print("=" * 60)
    print("PyPCI 基础使用示例")
    print("=" * 60)
    
    # 设置路径
    base_path = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input"
    
    parameters_file = base_path / "parameters.dat"
    ls_condition_file = base_path / "LS_condition_JT60SA.txt"
    equilibrium_dir = base_path
    
    # 检查文件是否存在
    if not parameters_file.exists():
        print(f"错误：找不到文件 {parameters_file}")
        return
    
    # 设置设备（如果有GPU则使用GPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 步骤1: 加载配置
    print("\n步骤1: 加载GENE配置...")
    config = load_gene_config(
        str(parameters_file),
        str(equilibrium_dir),
        device=device
    )
    print(f"  网格: nx0={config.nx0}, nky0={config.nky0}, nz0={config.nz0}")
    print(f"  参数: q0={config.q0}, shat={config.shat}")
    print(f"  归一化: B_ref={config.B_ref} T, rho_ref={config.rho_ref:.6f} m")
    
    print("\n步骤2: 加载光束配置...")
    beam_config = load_beam_config(str(ls_condition_file))
    print(f"  检测器: {beam_config.n_detectors_v} x {beam_config.n_detectors_t}")
    print(f"  光束采样点: {beam_config.n_beam_points}")
    print(f"  检测区域: {beam_config.width_vertical}m x {beam_config.width_toroidal}m")
    
    # 步骤3: 创建测试数据
    print("\n步骤3: 创建测试密度场...")
    # 创建一个简单的高斯湍流场
    ntheta = 400
    nx = config.nx0
    nz = config.nz0
    
    density_3d = torch.randn(ntheta, nx, nz, device=device) * 0.1
    print(f"  密度场shape: {density_3d.shape}")
    print(f"  密度范围: [{density_3d.min():.4f}, {density_3d.max():.4f}]")
    
    # 步骤4: 执行正向模拟
    print("\n步骤4: 执行PCI正向模拟...")
    
    # 如果平衡态数据未加载，设置简化模式
    if config.PA is None:
        print("  警告: 平衡态数据未加载，使用简化几何")
        # 设置简化的等离子体轴心
        config.PA = torch.tensor([config.major_R * config.L_ref, 0.0], device=device)
        config.GAC = torch.ones(nx+1, ntheta+1, device=device) * config.trpeps * config.major_R * config.L_ref
    
    try:
        import time
        start_time = time.time()
        
        pci_image = forward_projection(
            density_3d,
            config,
            beam_config,
            device=device
        )
        
        elapsed = time.time() - start_time
        
        print(f"  成功！耗时: {elapsed*1000:.2f} ms")
        print(f"  输出shape: {pci_image.shape}")
        print(f"  信号范围: [{pci_image.min():.4f}, {pci_image.max():.4f}]")
        
        # 步骤5: 可视化（如果有matplotlib）
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            print("\n步骤5: 可视化结果...")
            
            pci_np = pci_image.cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # PCI图像
            im1 = axes[0].imshow(pci_np, origin='lower', cmap='RdBu_r')
            axes[0].set_title('PCI Detector Image')
            axes[0].set_xlabel('Toroidal detectors')
            axes[0].set_ylabel('Vertical detectors')
            plt.colorbar(im1, ax=axes[0], label='Integrated signal')
            
            # 密度场的一个切片
            density_slice = density_3d[:, nx//2, :].cpu().numpy()
            im2 = axes[1].imshow(density_slice.T, origin='lower', cmap='viridis', aspect='auto')
            axes[1].set_title('Density Field (mid-radial slice)')
            axes[1].set_xlabel('Poloidal (theta)')
            axes[1].set_ylabel('Toroidal (z)')
            plt.colorbar(im2, ax=axes[1], label='Density')
            
            plt.tight_layout()
            
            output_path = Path(__file__).parent / "example_output.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  结果已保存到: {output_path}")
            
        except ImportError:
            print("\n  (跳过可视化 - 需要matplotlib)")
    
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()



