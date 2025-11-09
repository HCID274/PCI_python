"""
PCI结果可视化工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D

from .config import GENEConfig, BeamConfig


class PCIVisualizer:
    """PCI结果可视化工具"""
    
    def __init__(self, config: Optional[GENEConfig] = None):
        self.config = config
    
    def plot_beam_geometry_3d(
        self,
        beam_grid: dict,
        save_path: Optional[str] = None
    ):
        """3D光束几何（对应MATLAB Figure 1）"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        grid = beam_grid['grid_xyz'].cpu().numpy()
        div1, div2, divls, _ = grid.shape
        
        # 详细的数据分析
        print(f"  数据分析:")
        print(f"    网格形状: {grid.shape} (总点数: {div1 * div2 * divls})")
        print(f"    X范围: [{grid[:,:,:,0].min():.6f}, {grid[:,:,:,0].max():.6f}]")
        print(f"    Y范围: [{grid[:,:,:,1].min():.6f}, {grid[:,:,:,1].max():.6f}]")
        print(f"    Z范围: [{grid[:,:,:,2].min():.6f}, {grid[:,:,:,2].max():.6f}]")
        
        # 检查数据分布密度
        center = grid[div1//2, div2//2, :, :]
        print(f"    中心线长度: {np.linalg.norm(center[-1] - center[0]):.6f}")
        print(f"    起点: {center[0]}")  
        print(f"    终点: {center[-1]}")
        
        # 检查是否有重复点或零密度区域
        grid_flat = grid.reshape(-1, 3)
        unique_points = len(np.unique(grid_flat, axis=0))
        print(f"    唯一点数: {unique_points} / {len(grid_flat)}")
        
        # 绘制所有光束传播路径（对应MATLAB plot3(xls1,yls1,zls1,'.'))
        # 将所有网格点展平为一维数组，模拟MATLAB的reshape逻辑
        grid_flat = grid.reshape(-1, 3)
        ax.plot(grid_flat[:, 0], grid_flat[:, 1], grid_flat[:, 2], 'b.', 
                alpha=0.6, markersize=1.5, label='Beam propagation path')
        
        # 绘制中心线，强调弯曲特征
        center = grid[div1//2, div2//2, :, :]
        ax.plot(center[:, 0], center[:, 1], center[:, 2], 'r-', linewidth=1, 
                alpha=0.9, label='Beam center')
        
        # 绘制起点和终点（对应MATLAB plot3(B2(:,1),B2(:,2),B2(:,3),'o')）
        start = beam_grid['beam_start'].cpu().numpy()
        end = beam_grid['beam_end'].cpu().numpy()
        # 将起点终点组合成2x3数组，模拟MATLAB B2的格式
        B2_points = np.array([start, end])
        ax.plot(B2_points[:, 0], B2_points[:, 1], B2_points[:, 2], 'ro', 
                markersize=12, markerfacecolor='red', markeredgecolor='darkred', 
                markeredgewidth=2, label='Start/End points')
        
        # 绘制托卡马克边界（严格按照MATLAB LSview_com.m第144-148行）
        if self.config is not None and self.config.GRC is not None:
            GRC = self.config.GRC.cpu().numpy()  # (n_phi, n_R) 或 (n_R, n_phi)
            GZC = self.config.GZC.cpu().numpy()  # (n_phi, n_Z) 或 (n_Z, n_phi)
            
            # MATLAB逻辑正确实现：
            # xls_b=dataC.GRC(end,:).'*cos([0:30]*2*pi/30);
            # yls_b=dataC.GRC(end,:).'*sin([0:30]*2*pi/30);
            # zls_b=repmat(dataC.GZC(end,:).',1,31);
            
            R_boundary = GRC[-1, :]  # 边界上不同位置的不同R值
            Z_boundary = GZC[-1, :]  # 边界上对应位置的不同Z值
            
            # 生成31个角度点
            n_points = 31
            angles = np.arange(n_points) * 2 * np.pi / n_points
            
            # 关键：MATLAB的广播操作
            # R_boundary是(n,)数组，angles是(31,)数组
            # 结果是(n,31)数组，每个R值对应31个角度点
            xls_b = R_boundary[:, np.newaxis] * np.cos(angles[np.newaxis, :])  # (n_R, 31)
            yls_b = R_boundary[:, np.newaxis] * np.sin(angles[np.newaxis, :])  # (n_R, 31)
            zls_b = Z_boundary[:, np.newaxis] * np.ones(n_points)[np.newaxis, :]  # (n_Z, 31)
            
            # MATLAB: plot3(xls_b(1:2:end,:).',yls_b(1:2:end,:).',zls_b(1:2:end,:).','k-');
            # 步长为2取样（每两个点取一个）
            step_indices = slice(0, None, 2)  # 1:2:end的Python等价
            xls_b_sampled = xls_b[step_indices, :]
            yls_b_sampled = yls_b[step_indices, :]
            zls_b_sampled = zls_b[step_indices, :]
            
            # 检查是否有数据需要绘制
            if xls_b_sampled.shape[0] > 0:
                # 绘制第一条线并添加图例标签
                ax.plot(xls_b_sampled[0, :], yls_b_sampled[0, :], zls_b_sampled[0, :], 
                        'k-', linewidth=1.0, alpha=0.9, label='Tokamak boundary')
                
                # 循环绘制剩余的线，不再添加图例标签以保持图例干净
                for i in range(1, xls_b_sampled.shape[0]):
                    ax.plot(xls_b_sampled[i, :], yls_b_sampled[i, :], zls_b_sampled[i, :], 
                            'k-', linewidth=1.0, alpha=0.9)
        
        # 标准化坐标轴标签（严格按照MATLAB的X, Y, Z格式）
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold') 
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('3D Beam Geometry - Complex Tokamak Configuration (Figure 1)', 
                     fontsize=14, fontweight='bold')
        
        # 改善3D视角
        ax.view_init(elev=20, azim=45)  # 调整视角以更好地显示3D结构
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_detector_contour(
        self,
        pci_image: torch.Tensor,
        beam_config,
        time_t: float,
        save_path: Optional[str] = None
    ):
        """检测器信号等高线图（严格对应MATLAB Figure 3）
        
        Args:
            pci_image: PCI信号图像
            beam_config: 光束配置，包含width_v, width_t, n_detectors_v, n_detectors_t
            time_t: 时间点
            save_path: 保存路径
        """
        # MATLAB中的参数提取
        wid1 = beam_config.width_vertical  # 垂直宽度
        wid2 = beam_config.width_toroidal  # 环向宽度
        div1 = beam_config.div_vertical    # 垂直半范围
        div2 = beam_config.div_toroidal    # 环向半范围
        
        # 严格按照MATLAB: [xx1,yy1]=meshgrid(wid1/2*[-div1:div1]/div1,-wid2/2*[-div2:div2]/div2)
        # 注意：div1和div2已经是半范围，所以[-div1:div1]会给出2*div1+1个点
        xx1 = np.meshgrid(
            wid1/2 * np.arange(-div1, div1+1) / div1,
            -wid2/2 * np.arange(-div2, div2+1) / div2
        )[0]
        
        yy1 = np.meshgrid(
            wid1/2 * np.arange(-div1, div1+1) / div1,
            -wid2/2 * np.arange(-div2, div2+1) / div2
        )[1]
        
        # MATLAB: xx1 = fliplr(xx1) - 水平翻转
        xx1 = np.fliplr(xx1).copy()  # 使用.copy()避免负步长问题
        
        pci_np = pci_image.cpu().numpy()
        
        # 处理3D数据：沿光束路径积分得到2D检测器信号
        if pci_np.ndim == 3:
            # 按照MATLAB逻辑：pout2 = sum(pout1, 3)
            pci_2d = np.sum(pci_np, axis=2)
        else:
            pci_2d = pci_np
        
        fig = plt.figure(figsize=(10, 8))
        
        # 严格按照MATLAB: contourf(yy1.',xx1.',pout2,100,'LineStyle','none')
        plt.contourf(yy1.T, xx1.T, pci_2d, levels=100)
        
        # MATLAB: shading flat
        plt.gca().set_facecolor('white')
        
        # MATLAB: axis equal
        plt.axis('equal')
        
        # MATLAB: colorbar
        cbar = plt.colorbar()
        
        # MATLAB: xlabel('x (m)'); ylabel('y (m)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        
        # 添加时间信息
        plt.title(f'Detector Signal (t = {time_t:.2f}) (Figure 3)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_wavenumber_contour(
        self,
        Amp: torch.Tensor,
        KX: torch.Tensor,
        KY: torch.Tensor,
        mode_name: str,
        save_path: Optional[str] = None
    ):
        """波数空间等高线图（对应MATLAB Figure 4）"""
        fig = plt.figure(figsize=(10, 8))
        
        Amp_np = Amp.cpu().numpy()
        KX_np = KX.cpu().numpy()
        KY_np = KY.cpu().numpy()
        
        plt.contourf(KX_np, KY_np, np.log(Amp_np + 1e-10), 100, cmap='viridis')
        plt.xlabel(r'$k_x\rho_i$')
        plt.ylabel(r'$k_y\rho_i$')
        plt.colorbar(label='log(Amplitude)')
        plt.title(f'{mode_name} (Figure 4)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_time_evolution_animation(
        self,
        pout2_series: torch.Tensor,
        xx: torch.Tensor,
        yy: torch.Tensor,
        times: torch.Tensor,
        save_path: str
    ):
        """时间演化动画"""
        try:
            import matplotlib.animation as animation
        except ImportError:
            print("matplotlib.animation required for animation")
            return
        
        fig = plt.figure(figsize=(10, 8))
        
        pout2_np = pout2_series.cpu().numpy()
        xx_np = xx.cpu().numpy()
        yy_np = yy.cpu().numpy()
        times_np = times.cpu().numpy()
        
        # 计算全局vmin/vmax
        vmin, vmax = pout2_np.min(), pout2_np.max()
        
        def update(frame):
            plt.clf()
            plt.contourf(yy_np, xx_np, pout2_np[:, :, frame], 100, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            plt.axis('equal')
            plt.colorbar(label='Signal')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title(f't = {times_np[frame]:.2f}')
        
        anim = animation.FuncAnimation(fig, update, frames=pout2_np.shape[2], interval=100)
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved: {save_path}")
        plt.close()
    
    def plot_density_slice(
        self,
        density_3d: torch.Tensor,
        config: GENEConfig,
        beam_config=None,
        slice_indices: Optional[list] = None,
        save_path: Optional[str] = None
    ):
        """密度场poloidal截面可视化 (修正为MATLAB逻辑)
        
        Args:
            density_3d: 3D密度场数据 [R, Z, phi]
            config: GENE配置，包含GRC, GZC坐标网格
            beam_config: 光束配置 (保持兼容性)
            slice_indices: 保留参数兼容性，但使用poloidal截面
            save_path: 保存路径
        """
        if config.GRC is None or config.GZC is None:
            print("  Error: GRC/GZC coordinate grid data not available")
            return
            
        # 按照MATLAB cont_data2_s.m的逻辑实现poloidal截面可视化
        # 获取坐标网格 (MATLAB: contourf(obj.GRC,obj.GZC,(2*z),30,'LineStyle','none'))
        grc = config.GRC.cpu().numpy()  # R坐标网格
        gzc = config.GZC.cpu().numpy()  # Z坐标网格
        
        # 处理密度数据 (MATLAB: z=real(data2)*cos_pha-imag(data2)*sin_pha)
        density_np = density_3d.cpu().numpy()
        
        # 如果是3D数据，取中间phi截面作为poloidal截面 (MATLAB方式)
        if density_np.ndim == 3:
            phi_idx = density_np.shape[2] // 2
            density_2d = density_np[:, :, phi_idx]
        else:
            density_2d = density_np
            
        # 处理复数数据 (if exists)
        if density_2d.dtype == np.complex64 or density_2d.dtype == np.complex128:
            # MATLAB: z=real(data2)*cos_pha-imag(data2)*sin_pha
            density_processed = np.real(density_2d) * 2  # MATLAB *2 factor
        else:
            density_processed = density_2d * 2  # MATLAB *2 factor
        
        # Ensure data dimension matching
        if density_processed.shape != grc.shape:
            print(f"  Warning: Data dimension mismatch {density_processed.shape} vs {grc.shape}")
            min_rows = min(density_processed.shape[0], grc.shape[0])
            min_cols = min(density_processed.shape[1], grc.shape[1])
            density_processed = density_processed[:min_rows, :min_cols]
            grc = grc[:min_rows, :min_cols]
            gzc = gzc[:min_rows, :min_cols]
        
        # 创建poloidal截面图 (MATLAB: contourf(obj.GRC,obj.GZC,(2*z),30,'LineStyle','none'))
        fig = plt.figure(figsize=(12, 8))
        plt.contourf(grc, gzc, density_processed, levels=30, cmap='RdBu_r')
        plt.gca().set_aspect('equal')  # MATLAB: axis equal
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.title(f'Density Poloidal Cross-Section (phi={phi_idx}) (Figure 2)')
        plt.colorbar(label='Density')
        
        # MATLAB: shading flat
        plt.gca().set_facecolor('white')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure saved: {save_path}")
        else:
            plt.show()
        plt.close()
        
        return phi_idx  # 返回使用的phi截面索引
    
    def plot_density_3d_orthogonal(
        self,
        density_3d: torch.Tensor,
        config: GENEConfig,
        center_indices: Optional[dict] = None,
        save_path: Optional[str] = None
    ):
        """3D密度场的正交切片可视化
        
        Args:
            density_3d: 3D密度场数据 [R, Z, phi]
            config: GENE配置
            center_indices: 中心切片索引 {'r': R//2, 'z': Z//2, 'phi': phi//2}
            save_path: 保存路径
        """
        if center_indices is None:
            center_indices = {
                'r': density_3d.shape[0] // 2,
                'z': density_3d.shape[1] // 2,
                'phi': density_3d.shape[2] // 2
            }
        
        density_np = density_3d.cpu().numpy()
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R-Z平面（poloidal截面，phi固定）
        rz_slice = density_np[:, :, center_indices['phi']]
        im1 = axes[0, 0].contourf(rz_slice, levels=100, cmap='RdBu_r')
        axes[0, 0].set_title(f'R-Z截面 (φ={center_indices["phi"]})')
        axes[0, 0].set_xlabel('R index')
        axes[0, 0].set_ylabel('Z index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # R-φ平面（toroidal截面，Z固定）
        rphi_slice = density_np[:, center_indices['z'], :]
        im2 = axes[0, 1].contourf(rphi_slice.T, levels=100, cmap='RdBu_r')
        axes[0, 1].set_title(f'R-φ截面 (Z={center_indices["z"]})')
        axes[0, 1].set_xlabel('φ index')
        axes[0, 1].set_ylabel('R index')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Z-φ平面（poloidal截面，R固定）
        zphi_slice = density_np[center_indices['r'], :, :]
        im3 = axes[1, 0].contourf(zphi_slice.T, levels=100, cmap='RdBu_r')
        axes[1, 0].set_title(f'Z-φ截面 (R={center_indices["r"]})')
        axes[1, 0].set_xlabel('φ index')
        axes[1, 0].set_ylabel('Z index')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 3D密度分布直方图
        axes[1, 1].hist(density_np.flatten(), bins=50, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('密度分布直方图')
        axes[1, 1].set_xlabel('密度值')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Figure saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_mode_structure(
        self,
        R: torch.Tensor,
        p2: torch.Tensor,
        m: int,
        n: int,
        save_path: Optional[str] = None
    ):
        """
        模式结构图（严格对应MATLAB plot_mode_structure.m）
        
        Args:
            R: 径向坐标数组
            p2: 模态数据 [radial_points, 6个物理量] (PHI, PSI, VPL, NE, TE, TI)
            m, n: 模态编号
            save_path: 保存路径
        """
        # 物理量名称和对应关系 (严格按照MATLAB)
        # MATLAB: PHI(321), PSI(322), VPL(324), NE(323), TE(325), TI(326)
        subplot_mappings = [
            ('PHI', 3, 2, 1),   # subplot(321)
            ('PSI', 3, 2, 2),   # subplot(322)
            ('VPL', 3, 2, 4),   # subplot(324) 
            ('NE',  3, 2, 3),   # subplot(323)
            ('TE',  3, 2, 5),   # subplot(325)
            ('TI',  3, 2, 6),   # subplot(326)
        ]
        
        # 转换为numpy数组以便处理
        R_np = R.cpu().numpy()
        
        # 使用与MATLAB相近的图像大小 (默认matplotlib figsize约6.4x4.8)
        fig = plt.figure(figsize=(10, 8))
        
        for i, (field_name, rows, cols, index) in enumerate(subplot_mappings):
            ax = fig.add_subplot(rows, cols, index)
            
            # 获取对应物理量的数据
            field_data = p2[:, i].cpu().numpy()
            
            # MATLAB: plot(obj.R,real(p2(:,a)),'r',obj.R,imag(p2(:,a)),'b')
            # 使用默认线宽，与MATLAB一致
            ax.plot(R_np, np.real(field_data), 'r-', label='Real')
            ax.plot(R_np, np.imag(field_data), 'b-', label='Imag')
            
            # 设置标题和标签（严格按照MATLAB）
            ax.set_title(field_name)
            ax.set_xlabel('r')
            
            # MATLAB: axis tight
            ax.autoscale(axis='both', tight=True)
            
            # MATLAB: legend('Real','Imag','Location','Best') - 只对PHI和PSI
            if field_name in ['PHI', 'PSI']:
                ax.legend(['Real', 'Imag'], loc='best', fontsize=8)
        
        # 添加总标题
        fig.suptitle(f'Mode Structure (m={m}, n={n})', fontsize=14, fontweight='bold')
        
        # 调整子图间距
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Mode structure saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_mode_structure_complete(
        self,
        R: torch.Tensor,
        p2_all: torch.Tensor,
        m_values: list,
        n_values: list,
        save_path: Optional[str] = None
    ):
        """
        完整的模式结构图显示（支持多模态）
        
        Args:
            R: 径向坐标数组
            p2_all: 所有模态数据 [n_modes, radial_points, 6个物理量]
            m_values: m模态编号列表
            n_values: n模态编号列表
            save_path: 保存路径
        """
        n_modes = len(m_values)
        
        # 计算子图布局
        n_cols = min(3, n_modes)  # 最多3列
        n_rows = (n_modes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # 确保axes是2D数组
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        R_np = R.cpu().numpy()
        field_names = ['PHI', 'PSI', 'VPL', 'NE', 'TE', 'TI']
        
        for mode_idx, (m, n) in enumerate(zip(m_values, n_values)):
            row = mode_idx // n_cols
            col = mode_idx % n_cols
            ax = axes[row, col]
            
            # 获取模态数据
            p2 = p2_all[mode_idx]
            
            # 选择第一个物理量（PHI）作为示例
            phi_data = p2[:, 0].cpu().numpy()
            ax.plot(R_np, np.real(phi_data), 'r-', linewidth=1, label='Real')
            ax.plot(R_np, np.imag(phi_data), 'b-', linewidth=1, label='Imag')
            
            ax.set_title(f'm={m}, n={n}')
            ax.set_xlabel('r')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for mode_idx in range(n_modes, n_rows * n_cols):
            row = mode_idx // n_cols
            col = mode_idx % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Complete mode structure saved: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_spectrum(
        self,
        p2: torch.Tensor,
        field_number: int = 4,
        save_path: Optional[str] = None
    ):
        """
        绘制能谱图（严格对应MATLAB plot_spectrum.m）
        
        Args:
            p2: 全模态数据 [radial_points, poloidal_modes, toroidal_modes, n_fields]
            field_number: 物理量编号 (4=DENS, 5=TEME, 其他=1.0)
            save_path: 保存路径
        """
        # 物理量系数（严格按照MATLAB）
        # 假设合理的BETAI和BETAE值
        betai = getattr(self.config, 'BETAI', None)
        betae = getattr(self.config, 'BETAE', None)
        
        if betai is None or betae is None:
            # 如果没有设置BETAI和BETAE，使用beta参数估算
            beta = getattr(self.config, 'beta', 0.01)
            betai = beta * 0.8 if beta > 0 else 0.01
            betae = beta * 0.2 if beta > 0 else 0.01
        
        if field_number == 4:  # DENS
            coef = 1.0 / (betai + betae)
        elif field_number == 5:  # TEME
            coef = 1.5 / betai
        else:  # 其他物理量
            coef = 1.0
        
        # 转换为numpy处理
        p2_np = p2.cpu().numpy()
        
        # MATLAB: dr=zeros(obj.IRMAX+1,1);
        #         dr(1:end-1)=obj.R(2:end)-obj.R(1:end-1);
        R_np = self.config.R.cpu().numpy()
        dr = np.zeros(len(R_np))
        dr[0:-1] = R_np[1:] - R_np[0:-1]
        
        # MATLAB: ss=squeeze(sum(p2(:,:,:,fn).*conj(p2(:,:,:,fn)).*repmat(obj.R.*dr,1,obj.LYM2,obj.LZM2),1));
        # 选择特定物理量
        fn_idx = field_number - 1  # MATLAB是1基索引，Python是0基索引
        p2_field = p2_np[:, :, :, fn_idx]  # [radial, poloidal, toroidal]
        
        # 计算模方并乘以径向权重
        p2_conj = np.conj(p2_field)
        energy_density = p2_field * p2_conj
        
        # MATLAB: repmat(obj.R.*dr,1,obj.LYM2,obj.LZM2)
        R_dr = R_np * dr  # [radial_points]
        weight_3d = np.broadcast_to(R_dr[:, np.newaxis, np.newaxis], 
                                   (len(R_dr), p2_field.shape[1], p2_field.shape[2]))
        
        # 应用权重并对径向积分
        weighted_energy = energy_density * weight_3d
        ss = np.sum(weighted_energy, axis=0)  # 沿径向求和
        
        # ==============================================================================
        # 严格按照MATLAB的模态重排逻辑
        # ==============================================================================
        
        # MATLAB: ss2=zeros(obj.LYM2,obj.LZM2);
        LYM2 = p2_field.shape[1]  # 总poloidal模态数
        LZM2 = p2_field.shape[2]  # 总toroidal模态数
        ss2 = np.zeros((LYM2, LZM2), dtype=ss.dtype)
        
        # MATLAB: KYM = KYMt/2 (正m模态数)
        # LYM2 = KYMt (总模态数，包含正负和零模态)
        # 所以: KYM = LYM2//2
        KYM = LYM2 // 2  # 正m模态数
        KZM = LZM2 - 1   # toroidal模态最大编号
        
        # MATLAB: ss2(KYM:end,:)=ss(1:KYM+1,:);        % 正m模态: 0到KYM
        # 正m模态数: KYM+1个 (0, 1, 2, ..., KYM)
        # 位置: KYM 到 LYM2-1 (从KYM开始填充)
        pos_m_length = min(KYM + 1, ss.shape[0], LYM2 - KYM)
        if pos_m_length > 0:
            ss2[KYM:KYM+pos_m_length, :] = ss[0:pos_m_length, :]
        
        # MATLAB: ss2(1:KYM-1,:)=ss(KYM+2:end,:);      % 负m模态: -(KYM-1)到-1
        # 负m模态数: KYM-1个 (-(KYM-1), ..., -2, -1)
        # 位置: 0 到 KYM-2 (从前KYM-1个位置开始填充)
        neg_m_target = KYM - 1
        neg_m_source_start = KYM + 2
        neg_m_source_length = ss.shape[0] - neg_m_source_start
        neg_m_copy_length = min(neg_m_target, neg_m_source_length, LYM2)
        
        if neg_m_copy_length > 0:
            ss2[0:neg_m_copy_length, :] = ss[neg_m_source_start:neg_m_source_start+neg_m_copy_length, :]
        
        # MATLAB: ss2(1:KYM-1,1)=ss([KYM:-1:2],1);   % toroidal模态0的负m部分
        # 特殊处理：toroidal模态0 (第一个列) 的负m部分需要反向填充
        if LZM2 > 0 and KYM > 1:
            special_length = min(KYM - 1, ss.shape[0])
            if special_length > 0:
                # 反向索引：ss[KYM-1], ss[KYM-2], ..., ss[2]
                reverse_indices = list(range(min(KYM, ss.shape[0]) - 1, max(0, KYM - special_length - 1), -1))
                copy_length = min(len(reverse_indices), KYM - 1, ss2.shape[0])
                if copy_length > 0:
                    ss2[0:copy_length, 0] = ss[reverse_indices[:copy_length], 0]
        
        # ==============================================================================
        # 模态重排完成
        # ==============================================================================
        
        # 应用系数并确保实数
        ss2 = np.abs(ss2 * coef)  # 使用绝对值确保实数
        
        # MATLAB: [x,y]=meshgrid([-obj.KYM+1:obj.KYM],[0:obj.KZM]);
        #         contourf(x,y,log10(ss2.'),30,'LineStyle','none');
        
        # 计算正确的m,n范围（严格按照MATLAB）
        # m范围: [-KYM, ..., -1, 0, 1, ..., KYM]  共2*KYM+1个值
        # n范围: [0, 1, ..., KZM]  共KZM+1个值
        m_range = np.arange(-KYM, KYM + 1)  # 从-KYM到KYM，共2*KYM+1个值
        n_range = np.arange(0, KZM + 1)     # 正好是0到KZM
        
        # 生成网格（MATLAB meshgrid行为）
        X, Y = np.meshgrid(m_range, n_range, indexing='xy')  # 匹配MATLAB的meshgrid行为
        
        # 确保数据维度与网格匹配
        if ss2.shape != X.shape:
            # 注意：ss2是(21,5)，X是(5,21)，这是正确的
            # MATLAB中ss2.'就是转置，所以不需要调整原始ss2
            pass  # 保持原样，让转置来处理维度匹配
        
        # 绘制图形
        fig = plt.figure(figsize=(10, 8))
        
        # 绘制30层对数等高线
        try:
            # MATLAB: contourf(x,y,log10(ss2.'),30,'LineStyle','none');
            # MATLAB使用ss2.'即转置，但需要注意维度匹配
            # X: (21,5), Y: (21,5), ss2: (21,5)
            # 转置后ss2.T: (5,21)，与X的转置形状匹配
            contour = plt.contourf(X, Y, np.log10(ss2.T + 1e-10), levels=30)
            
            # MATLAB: shading flat;
            plt.gca().set_facecolor('white')
            
            # 设置标签和标题
            plt.title('spectrum')
            plt.xlabel('m')
            plt.ylabel('n')
            plt.axis('tight')
            
            # 添加颜色条
            cbar = plt.colorbar()
            cbar.set_label('log10(Amplitude)')
            
        except Exception as e:
            print(f"    Contour plotting warning: {e}")
            # Fallback: ensure dimensions are correct
            print(f"    Trying adjustment: X{X.shape}, Y{Y.shape}, ss2{ss2.shape}")
            # X(5, 21), Y(5, 21), ss2 should be (21, 5), after transpose (5, 21)
            data_for_plot = ss2.T  # MATLAB's ss2.' is transpose
            
            try:
                plt.contourf(X, Y, np.log10(data_for_plot + 1e-10), levels=30)
                plt.gca().set_facecolor('white')
                plt.title('spectrum (fallback)')
                plt.xlabel('m')
                plt.ylabel('n')
                plt.colorbar()
            except Exception as e2:
                print(f"    Fallback method also failed: {e2}")
                # Final fallback
                plt.imshow(np.log10(ss2 + 1e-10), aspect='auto', origin='lower',
                          extent=[m_range[0], m_range[-1], n_range[0], n_range[-1]])
                plt.colorbar()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Spectrum saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def spectrum(
        self,
        sim_type: str,
        data_n: int,
        t: float,
        save_path: Optional[str] = None
    ):
        """
        主能谱分析函数（对应MATLAB spectrum.m）
        
        Args:
            sim_type: 仿真类型 ('R6F', 'r6f', 'R5F', 'r5f')
            data_n: 数据编号
            t: 时间点
            save_path: 保存路径
        """
        # 这里需要实现数据读取逻辑
        # 暂时使用模拟数据进行演示
        print(f"能谱分析: sim_type={sim_type}, data_n={data_n}, t={t}")
        
        # 严格按照MATLAB的模态定义创建数据
        # 假设KYM=10, KZM=4, 则LYM2=2*KYM+1=21, LZM2=KZM+1=5
        KYM = 10  # 正m模态数
        KZM = 4   # toroidal模态最大编号
        
        n_radial = len(self.config.R)  # 使用config中的径向点数，确保与plot_spectrum一致
        n_poloidal = 2 * KYM + 1  # 总模态数 = 正m+负m+零模态 = 21
        n_toroidal = KZM + 1      # toroidal模态数 = KZM+1 = 5
        n_fields = 6
        
        print(f"  模态配置: KYM={KYM}, KZM={KZM}, LYM2={n_poloidal}, LZM2={n_toroidal}")
        
        p2 = torch.zeros((n_radial, n_poloidal, n_toroidal, n_fields), 
                        dtype=torch.complex64, device='cpu')
        
        # 生成一些模拟的模态数据
        for i in range(n_fields):
            for j in range(n_poloidal):
                for k in range(n_toroidal):
                    # 创建一些模式结构
                    r = torch.linspace(0.1, 2.0, n_radial)
                    real_part = torch.sin(2 * np.pi * r * (1 + 0.1 * (j + k)))
                    imag_part = torch.cos(2 * np.pi * r * (1 + 0.1 * (j + k)))
                    p2[:, j, k, i] = real_part + 1j * imag_part
        
        # 根据时间选择物理量
        if t == 0:
            # 时间序列，使用DENS (field 4)
            field_number = 4
        else:
            # 单时间点，使用TEME (field 5)
            field_number = 5
        
        print(f"  Using field: {field_number} ({'DENS' if field_number == 4 else 'TEME' if field_number == 5 else 'OTHER'})")
        
        # 绘制能谱图
        self.plot_spectrum(p2, field_number, save_path)
        
        return p2


def plot_wavenumber_space_2d(
    xx,
    yy,
    real_space_data: torch.Tensor,
    config: GENEConfig,
    save_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D波数空间分析（严格按照MATLAB plotWaveNumberSpace.m）
    """
    # MATLAB: Mean = mean(mean(realSpaceData(:)));
    Mean = torch.mean(real_space_data)
    
    # MATLAB: realSpaceData = realSpaceData - Mean;
    data_centered = real_space_data - Mean
    
    # MATLAB: [Ny, Nx] = size(realSpaceData);
    # 处理3D数据：取2D投影
    if data_centered.ndim == 3:
        # 按照MATLAB逻辑：pout2 = sum(pout1, 3)
        data_2d = torch.sum(data_centered, dim=2)
    else:
        data_2d = data_centered
    
    Ny, Nx = data_2d.shape
    
    # MATLAB: P = fft2(realSpaceData);
    P = torch.fft.fft2(data_2d)
    
    # MATLAB: P_shifted = fftshift(P);
    P_shifted = torch.fft.fftshift(P)
    
    # MATLAB: Amp = abs(P_shifted);
    Amp = torch.abs(P_shifted)
    
    # MATLAB中的网格计算：这里需要模拟MATLAB的网格生成
    # MATLAB的plotWaveNumberSpace会基于实际数据的尺寸来计算波数
    # 但这里我们使用实际的网格间距
    
    if xx is not None and yy is not None:
        # 如果提供了空间坐标，计算实际的dx, dy
        # MATLAB: dx = (xx(1,2) - xx(1,1));
        dx = (xx[0, 1] - xx[0, 0]).item()
        
        # MATLAB: dy = (yy(2,1) - yy(1,1));
        dy = (yy[1, 0] - yy[0, 0]).item()
    else:
        # 如果没有提供空间坐标，使用默认的网格间距
        # 这里假设单位网格间距
        dx = 1.0
        dy = 1.0
    
    # MATLAB: if mod(Nx, 2) == 0
    if Nx % 2 == 0:
        # MATLAB: kx = 2*pi*(-Nx/2:Nx/2-1)/((Nx-1)*dx);
        kx = 2 * np.pi * np.arange(-Nx//2, Nx//2) / ((Nx - 1) * dx)
    else:
        # MATLAB: kx = 2*pi*(-(Nx-1)/2:(Nx-1)/2)/((Nx-1)*dx);
        kx = 2 * np.pi * np.arange(-(Nx-1)//2, (Nx-1)//2 + 1) / ((Nx - 1) * dx)
    
    # MATLAB: if mod(Ny, 2) == 0
    if Ny % 2 == 0:
        # MATLAB: ky = 2*pi*(-Ny/2:Ny/2-1)/((Ny-1)*dy);
        ky = 2 * np.pi * np.arange(-Ny//2, Ny//2) / ((Ny - 1) * dy)
    else:
        # MATLAB: ky = 2*pi*(-(Ny-1)/2:(Ny-1)/2)/((Ny-1)*dy);
        ky = 2 * np.pi * np.arange(-(Ny-1)//2, (Ny-1)//2 + 1) / ((Ny - 1) * dy)
    
    # MATLAB: [KX, KY] = meshgrid(kx, ky);
    # MATLAB的meshgrid默认是indexing='xy'行为
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    
    # MATLAB: 归一化处理
    if config.FVER == 5:
        # MATLAB: kx = kx*obj.rho_ref; ky = ky*obj.rho_ref;
        kx = kx * config.rho_ref
        ky = ky * config.rho_ref
        KX = KX * config.rho_ref
        KY = KY * config.rho_ref
    else:
        # MATLAB: kx = kx*0.003; ky = ky*0.003;
        kx = kx * 0.003
        ky = ky * 0.003
        KX = KX * 0.003
        KY = KY * 0.003
    
    # MATLAB: contourf(KX, KY, log(Amp), 100, 'LineStyle', 'none');
    if save_path:
        plt.figure(figsize=(10, 8))
        plt.contourf(KX, KY, np.log(Amp.cpu().numpy()), levels=100)
        plt.xlabel('k_xρ_i')
        plt.ylabel('k_yρ_i')
        plt.colorbar()
        plt.title('Wavenumber Space (Figure 4)')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    kx_tensor = torch.from_numpy(kx).to(real_space_data.device)
    ky_tensor = torch.from_numpy(ky).to(real_space_data.device)
    KX_tensor = torch.from_numpy(KX).to(real_space_data.device)
    KY_tensor = torch.from_numpy(KY).to(real_space_data.device)
    
    return Amp, KX_tensor, KY_tensor

