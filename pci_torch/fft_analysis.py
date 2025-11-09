"""
3D FFT频谱分析

对应MATLAB的FFT_3D.m
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, List
import scipy.io

from .config import GENEConfig
from .path_config import PathConfig


class FFT3DAnalyzer:
    """3D傅里叶变换分析器"""
    
    def __init__(self, config: GENEConfig, data_n: int, path_config: PathConfig, sim_type: str = 'GENE'):
        """
        初始化FFT分析器
        
        Args:
            config: GENE配置
            data_n: 数据编号
            path_config: 路径配置对象
            sim_type: 仿真类型 ('GENE' 或 'R5F')
        """
        self.config = config
        self.data_n = data_n
        self.path_config = path_config
        
        # 数据变量
        self.realSpaceData = None
        self.xx = None
        self.yy = None
        self.tt = None
        
        # FFT结果
        self.Amp = None
        self.kx = None
        self.ky = None
        self.f = None
        self.KX = None
        self.KY = None
        self.F = None
        
        # 数据格式标识
        if sim_type.upper() in ['R5F', 'R5F']:
            self.FVER = 3.5  # R5F格式
        else:
            self.FVER = 5   # GENE格式 (默认)
        
        self.time_files = None
        self.time_n = 0
        
        print(f"FFT分析器初始化: sim_type={sim_type}, FVER={self.FVER}")
    
    def load_time_series(self, device: str = 'cpu') -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        加载IntegratedSignal_overall.mat和时间/空间网格 (按MATLAB精确实现)
        
        Returns:
            (realSpaceData, (xx, yy), tt)
        """
        print("加载时间序列数据...")
        
        # 加载MAT文件（使用PathConfig）
        mat_path = self.path_config.get_integrated_signal_path(self.data_n)
        if not mat_path.exists():
            raise FileNotFoundError(f"找不到文件: {mat_path}")
        
        data = scipy.io.loadmat(str(mat_path))
        pout2 = data['pout2']  # (Ny, Nx, Nt)
        
        Ny, Nx, Nt = pout2.shape
        print(f"  数据shape: {pout2.shape}")
        
        # 转换为tensor
        self.realSpaceData = torch.from_numpy(pout2).to(device=device, dtype=torch.float64)
        
        # 加载光束配置获取空间网格（使用PathConfig）
        ls_condition_file = self.path_config.beam_config_file
        if not ls_condition_file.exists():
            raise FileNotFoundError(f"光束配置文件不存在: {ls_condition_file}")
        
        with open(ls_condition_file, 'r') as f:
            lines = f.readlines()
            # 跳过注释行，读取数据行
            data_lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
            
            # 第1行：pp1 (这行可能被跳过)
            # 第2行：wid1, wid2
            wid1, wid2 = map(float, data_lines[1].split(','))
            
            # 第3行：div1, div2, divls
            div1, div2, divls = map(int, data_lines[2].split(','))
        
        # 生成网格 (严格按MATLAB逻辑)
        # MATLAB: [xx1,yy1] = meshgrid(wid1/2*[-div1:div1]/div1,-wid2/2*[-div2:div2]/div2);
        x_coords = wid1/2 * torch.linspace(-div1, div1, 2*div1+1, device=device) / div1
        y_coords = -wid2/2 * torch.linspace(-div2, div2, 2*div2+1, device=device) / div2
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        xx = torch.fliplr(xx)  # 对应MATLAB的fliplr
        
        self.xx = xx
        self.yy = yy
        
        # 按MATLAB逻辑处理时间数据
        if self.FVER == 3.5:  # R5F
            # R5F: 使用纯数字.dat文件
            f_list = sorted(self.path_config.input_dir.glob('*.dat'))
            # 过滤只包含数字的文件名
            f_list = [f for f in f_list if f.stem.isdigit()]
            self.time_n = len(f_list)
        else:  # GENE
            # GENE: 使用TORUSIons_act_*.dat文件
            f_list = sorted(self.path_config.input_dir.glob('TORUSIons_act_*.dat'))
            self.time_n = len(f_list)
        
        print(f"  找到 {self.time_n} 个时间文件")
        self.time_files = f_list
        
        # 提取时间值 (按MATLAB逻辑)
        t_numbers = []
        for file_path in f_list:
            if self.FVER == 3.5:  # R5F
                t_string = file_path.stem  # 纯数字
            else:  # GENE
                # 提取 TORUSIons_act_123.dat 中的 123
                import re
                match = re.search(r'TORUSIons_act_(\d+)\.dat', file_path.name)
                if match:
                    t_string = match.group(1)
                else:
                    continue
            
            t_num = float(t_string) / 100.0  # MATLAB: t(i) = t(i)/100;
            t_numbers.append(t_num)
        
        if len(t_numbers) == 0:
            # 如果没有找到时间文件，使用默认
            t_numbers = list(range(self.time_n))
        
        # 按时间排序 (按MATLAB: [~, sortOrder] = sort(t_numbers))
        sort_order = np.argsort(t_numbers)
        sorted_t_numbers = [t_numbers[i] for i in sort_order]
        
        # 创建时间向量
        t = np.array(sorted_t_numbers)
        
        # 按MATLAB: t = repmat(t,1,size(t,1)); % 创建时间网格
        t = np.tile(t.reshape(-1, 1), (1, len(t)))
        
        # 转换为真实时间 (按MATLAB逻辑)
        if self.FVER == 5:  # GENE
            # MATLAB: t = t * (dataC.L_ref/sqrt(2*dataC.T_ref2/dataC.m_ref));
            time_norm_factor = self.config.L_ref / np.sqrt(2 * self.config.T_ref2 / self.config.m_ref)
            t = t * time_norm_factor
        
        # 转换为tensor
        self.tt = torch.from_numpy(t).to(device=device, dtype=torch.float64)
        
        print(f"  空间网格: {self.xx.shape}")
        print(f"  时间矩阵: {self.tt.shape}")
        print(f"  时间范围: [{t.min():.4f}, {t.max():.4f}]")
        
        return self.realSpaceData, (self.xx, self.yy), self.tt
    
    def compute_3d_fft(self) -> Tuple[torch.Tensor, ...]:
        """
        执行3D FFT (按MATLAB精确实现)
        
        Returns:
            (Amp, kx, ky, f, KX, KY, F)
        """
        print("\n执行3D FFT...")
        
        if self.realSpaceData is None:
            raise ValueError("请先调用load_time_series()加载数据")
        
        Ny, Nx, Nt = self.realSpaceData.shape
        
        # 3D FFT
        P = torch.fft.fftn(self.realSpaceData)
        P_shifted = torch.fft.fftshift(P)
        self.Amp = torch.abs(P_shifted)
        
        # 获取空间和时间分辨率 (按MATLAB逻辑)
        dx = (self.xx[0, 1] - self.xx[0, 0]).item()
        dy = (self.yy[1, 0] - self.yy[0, 0]).item()
        dt = (self.tt[1, 0] - self.tt[0, 0]).item() if len(self.tt) > 1 else 1.0
        
        # 生成波数和频率向量 (严格按MATLAB逻辑)
        if Nx % 2 == 0:
            # MATLAB: kx = 2*pi*(-Nx/2:Nx/2-1)/((Nx-1)*dx);
            kx = 2 * np.pi * np.arange(-Nx//2, Nx//2) / ((Nx - 1) * dx)
        else:
            # MATLAB: kx = 2*pi*(-(Nx-1)/2:(Nx-1)/2)/((Nx-1)*dx);
            kx = 2 * np.pi * np.arange(-(Nx-1)//2, (Nx-1)//2 + 1) / ((Nx - 1) * dx)
        
        if Ny % 2 == 0:
            # MATLAB: ky = 2*pi*(-Ny/2:Ny/2-1)/((Ny-1)*dy);
            ky = 2 * np.pi * np.arange(-Ny//2, Ny//2) / ((Ny - 1) * dy)
        else:
            # MATLAB: ky = 2*pi*(-(Ny-1)/2:(Ny-1)/2)/((Ny-1)*dy);
            ky = 2 * np.pi * np.arange(-(Ny-1)//2, (Ny-1)//2 + 1) / ((Ny - 1) * dy)
        
        if Nt % 2 == 0:
            # MATLAB: f = 2*pi*(-Nt/2:Nt/2-1)/(Nt*dt);
            f = 2 * np.pi * np.arange(-Nt//2, Nt//2) / (Nt * dt)
        else:
            # MATLAB: f = 2*pi*(-(Nt-1)/2:(Nt-1)/2)/(Nt*dt);
            f = 2 * np.pi * np.arange(-(Nt-1)//2, (Nt-1)//2 + 1) / (Nt * dt)
        
        # 创建3D网格 (按MATLAB: [KX, KY, F] = meshgrid(kx, ky, f))
        KY_np, KX_np, F_np = np.meshgrid(ky, kx, f, indexing='ij')
        
        # 归一化 (按FVER和MATLAB逻辑)
        if self.FVER == 5:  # GENE
            if self.config.rho_ref is not None:
                kx = kx * self.config.rho_ref
                ky = ky * self.config.rho_ref
                KX_np = KX_np * self.config.rho_ref
                KY_np = KY_np * self.config.rho_ref
            
            # 频率归一化
            time_norm = self.config.L_ref / np.sqrt(2 * self.config.T_ref2 / self.config.m_ref)
            f = f * time_norm
            F_np = F_np * time_norm
        else:  # R5F
            # MATLAB: kx = kx * 0.003; etc.
            kx = kx * 0.003
            ky = ky * 0.003
            KX_np = KX_np * 0.003
            KY_np = KY_np * 0.003
            # 频率归一化在R5F中未实现 (按MATLAB注释)
        
        # 转换为tensor
        self.kx = torch.from_numpy(kx).to(self.realSpaceData.device)
        self.ky = torch.from_numpy(ky).to(self.realSpaceData.device)
        self.f = torch.from_numpy(f).to(self.realSpaceData.device)
        self.KX = torch.from_numpy(KX_np).to(self.realSpaceData.device)
        self.KY = torch.from_numpy(KY_np).to(self.realSpaceData.device)
        self.F = torch.from_numpy(F_np).to(self.realSpaceData.device)
        
        print(f"  FFT完成 (FVER={self.FVER})")
        print(f"  kx范围: [{kx.min():.4f}, {kx.max():.4f}]")
        print(f"  ky范围: [{ky.min():.4f}, {ky.max():.4f}]")
        print(f"  f范围: [{f.min():.4f}, {f.max():.4f}]")
        
        return self.Amp, self.kx, self.ky, self.f, self.KX, self.KY, self.F
    
    def analyze_mode1_kxky(self, f_idx: int, save_path: Optional[str] = None):
        """模式1：kx-ky频谱（固定频率切片）"""
        import matplotlib.pyplot as plt
        
        if self.Amp is None:
            raise ValueError("请先调用compute_3d_fft()")
        
        print(f"\n模式1: kx-ky频谱 (f_idx={f_idx}, f={self.f[f_idx]:.4f})")
        
        slice_data = self.Amp[:, :, f_idx].cpu().numpy()
        KX_slice = self.KX[:, :, f_idx].cpu().numpy()
        KY_slice = self.KY[:, :, f_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(KX_slice, KY_slice, np.log(slice_data + 1e-10), 30, cmap='viridis')
        plt.xlabel(r'$k_x \rho_i$')
        plt.ylabel(r'$k_y \rho_i$')
        plt.title(f'Slice at f = {self.f[f_idx]:.4f}')
        plt.colorbar(label='log(Amplitude)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode2_wky(self, kx_idx: int, save_path: Optional[str] = None):
        """模式2：ω-ky频谱（固定kx切片）"""
        import matplotlib.pyplot as plt
        
        if self.Amp is None:
            raise ValueError("请先调用compute_3d_fft()")
        
        print(f"\n模式2: ω-ky频谱 (kx_idx={kx_idx}, kx={self.kx[kx_idx]:.4f})")
        
        slice_data = self.Amp[:, kx_idx, :].cpu().numpy()
        F_slice = self.F[:, kx_idx, :].cpu().numpy()
        KY_slice = self.KY[:, kx_idx, :].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(F_slice, KY_slice, np.log(slice_data + 1e-10), 30, cmap='viridis')
        plt.xlabel('f')
        plt.ylabel(r'$k_y \rho_i$')
        plt.title(f'Slice at k_x = {self.kx[kx_idx]:.4f}')
        plt.colorbar(label='log(Amplitude)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode3_wkx(self, ky_idx: int, save_path: Optional[str] = None):
        """模式3：ω-kx频谱（固定ky切片）"""
        import matplotlib.pyplot as plt
        
        if self.Amp is None:
            raise ValueError("请先调用compute_3d_fft()")
        
        print(f"\n模式3: ω-kx频谱 (ky_idx={ky_idx}, ky={self.ky[ky_idx]:.4f})")
        
        slice_data = self.Amp[ky_idx, :, :].cpu().numpy()
        F_slice = self.F[ky_idx, :, :].cpu().numpy()
        KX_slice = self.KX[ky_idx, :, :].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(F_slice, KX_slice, np.log(slice_data + 1e-10), 30, cmap='viridis')
        plt.xlabel('f')
        plt.ylabel(r'$k_x \rho_i$')
        plt.title(f'Slice at k_y = {self.ky[ky_idx]:.4f}')
        plt.colorbar(label='log(Amplitude)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode4_snapshot(self, t_idx: int, save_path: Optional[str] = None):
        """模式4：时间快照分析 (修正版，匹配MATLAB)"""
        import matplotlib.pyplot as plt
        
        if self.realSpaceData is None:
            raise ValueError("请先调用load_time_series()")
        
        print(f"\n模式4: 时间快照 (t_idx={t_idx}, t={self.tt[t_idx]:.4f})")
        
        # 获取时间转换因子 (用于显示)
        time_factor = self.config.L_ref / np.sqrt(2 * self.config.T_ref2 / self.config.m_ref)
        display_time = self.tt[t_idx] / time_factor
        
        snapshot = self.realSpaceData[:, :, t_idx].cpu().numpy()
        xx = self.xx.cpu().numpy()
        yy = self.yy.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, snapshot, 100, 'LineStyle', 'none')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f't = {display_time:.2f}')
        plt.colorbar()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode4_angle_fk_spectrum(self, theta: float, save_path: Optional[str] = None):
        """模式4扩展：角度相关的f-k谱分析"""
        import matplotlib.pyplot as plt
        
        if self.Amp is None:
            raise ValueError("请先调用compute_3d_fft()")
        
        print(f"\n模式4扩展: 角度f-k谱 (theta={theta}°)")
        
        # 按MATLAB逻辑实现角度相关分析
        KX2 = self.KX[1:(self.KX.shape[0]+1)//2, :, 0]
        KY2 = self.KY[1:(self.KY.shape[0]+1)//2, :, 0]
        KX3 = KX2[0, (self.KX.shape[1]+1)//2-1]
        KY3 = KY2[(self.KY.shape[0]+1)//2-1, 0]
        
        theta = 180 - theta  # 转换为MATLAB坐标系
        theta_rad = np.deg2rad(theta)
        
        Ny, Nx, Nt = self.Amp.shape
        fkAmp = np.zeros(((Ny+1)//2, Nt))
        FK = np.zeros(((Ny+1)//2, Nt))
        FF = np.zeros((Nt,))
        
        # 时间转换因子
        time_factor = self.config.L_ref / np.sqrt(2 * self.config.T_ref2 / self.config.m_ref)
        
        for j in range(Nt):
            for i in range((Ny+1)//2):
                x2 = (KY3/np.tan(theta_rad)) * (i-1)
                y2 = KY3 * (i-1)
                p2 = x2 / KX3
                pa2 = int(np.floor(p2))
                pb2 = p2 - pa2
                
                col_idx1 = (Nx+1)//2 + pa2
                col_idx2 = col_idx1 + 1
                
                if col_idx1 < 1 or col_idx1 > Nx or col_idx2 < 1 or col_idx2 > Nx:
                    continue
                
                fkAmp[i, j] = (1 - pb2) * self.Amp[Ny-i, col_idx1, j].item() + pb2 * self.Amp[Ny-i, col_idx2, j].item()
                FK[i, j] = np.sqrt((KY3*(i-1))**2 + x2**2)
                FF[j] = self.f[j] * time_factor  # 转换到真实频率
        
        plt.figure(figsize=(10, 8))
        plt.contourf(FF, FK, np.log(fkAmp[::-1, ::-1] + 1e-10), 100, 'LineStyle', 'none')
        plt.xlabel('f (kHz)')
        plt.ylabel('kρ_i')
        plt.colorbar()
        plt.title(f'f-k spectrum at θ = {theta}°')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode6_2dfft_detailed(self, save_path: Optional[str] = None):
        """模式6：详细的时间平均2D波数谱分析 (按MATLAB精确实现)"""
        import matplotlib.pyplot as plt
        
        if self.realSpaceData is None:
            raise ValueError("请先调用load_time_series()")
        
        print("\n模式6: 详细2D FFT时间平均分析")
        
        Ny, Nx, Nt = self.realSpaceData.shape
        
        # 计算每个时间点的2D FFT
        kxkyAmp = torch.zeros(Ny, Nx, Nt, device=self.realSpaceData.device)
        KX_3d = torch.zeros(Ny, Nx, Nt, device=self.realSpaceData.device)
        KY_3d = torch.zeros(Ny, Nx, Nt, device=self.realSpaceData.device)
        
        # 计算时间平均值
        Mean = torch.mean(self.realSpaceData, dim=2, keepdim=True)
        
        for i in range(self.time_n):
            data_i = self.realSpaceData[:, :, i]
            kxkyAmp_i, KX_i, KY_i = self.plotWaveNumberSpace_2(self.xx, self.yy, data_i, Mean[:, :, 0])
            kxkyAmp[:, :, i] = kxkyAmp_i
            KX_3d[:, :, i] = KX_i
            KY_3d[:, :, i] = KY_i
        
        # 计算平均
        kxkyAmp_ave = torch.mean(kxkyAmp, dim=2)
        KX_fixed = KX_3d[:, :, 0]  # 取第一个时间点的波数网格
        KY_fixed = KY_3d[:, :, 0]
        
        # 绘制2D波数谱
        plt.figure(figsize=(10, 8))
        plt.contourf(KX_fixed.cpu().numpy(), KY_fixed.cpu().numpy(), 
                    np.log(kxkyAmp_ave.cpu().numpy() + 1e-10), 100, 'LineStyle', 'none')
        plt.xlabel('k_xρ_i')
        plt.ylabel('k_yρ_i')
        plt.colorbar()
        plt.title('Time-averaged 2D wavenumber spectrum')
        
        if save_path:
            # 保存图像
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存图像到: {save_path}")
            
            # 保存MAT文件
            mat_save_path = str(Path(save_path).with_suffix('.mat'))
            scipy.io.savemat(mat_save_path, {'kxkyAmp_ave': kxkyAmp_ave.cpu().numpy()})
            print(f"  保存MAT到: {mat_save_path}")
        else:
            plt.show()
        plt.close()
        
        return kxkyAmp_ave, KX_fixed, KY_fixed
    
    def analyze_mode6_with_angle_cut(self, theta: float, kxkyAmp_ave: torch.Tensor, 
                                    KX_fixed: torch.Tensor, KY_fixed: torch.Tensor, 
                                    save_path: Optional[str] = None):
        """模式6扩展：带角度切割的1D波数谱分析"""
        import matplotlib.pyplot as plt
        
        Ny, Nx = kxkyAmp_ave.shape
        
        # 只取上半部分 (按MATLAB逻辑)
        kxkyAmp_ave2 = kxkyAmp_ave[1:(Ny+1)//2, :]
        KX2 = KX_fixed[1:(Ny+1)//2, :]
        KY2 = KY_fixed[1:(Ny+1)//2, :]
        
        # 绘制上半部分谱
        plt.figure(figsize=(10, 8))
        plt.contourf(KX2.cpu().numpy(), KY2.cpu().numpy(), 
                    np.log(kxkyAmp_ave2.cpu().numpy() + 1e-10), 100, 'LineStyle', 'none')
        plt.xlabel('k_xρ_i')
        plt.ylabel('k_yρ_i')
        plt.colorbar()
        plt.title(f'2D spectrum with angle cut θ = {theta}°')
        
        # 添加角度线
        KX3 = KX2[0, (Nx+1)//2-1]
        KY3 = KY2[(Ny+1)//2-1, 0]
        
        theta2 = theta  # 保存原始角度用于文件名
        theta = 180 - theta  # 转换为MATLAB坐标系
        
        if theta == 90:  # 垂直线
            kxkyAmp_ave3 = torch.zeros((Ny+1)//2)
            K = torch.zeros((Ny+1)//2)
            for i in range((Ny+1)//2):
                kxkyAmp_ave3[i] = kxkyAmp_ave2[-(i+1), (Nx+1)//2]
                K[i] = KY3 * i
        else:  # 任意角度
            theta_rad = np.deg2rad(theta)
            kxkyAmp_ave3 = torch.zeros((Ny+1)//2)
            K = torch.zeros((Ny+1)//2)
            x = torch.zeros((Ny+1)//2)
            y = torch.zeros((Ny+1)//2)
            
            for i in range((Ny+1)//2):
                x[i] = (KY3/np.tan(theta_rad)) * i
                y[i] = KY3 * i
                p = x[i] / KX3
                pa = int(np.floor(p))
                pb = p - pa
                
                col_idx1 = (Nx+1)//2 + pa
                col_idx2 = col_idx1 + 1
                
                if col_idx1 >= 1 and col_idx1 <= Nx and col_idx2 >= 1 and col_idx2 <= Nx:
                    kxkyAmp_ave3[i] = (1 - pb) * kxkyAmp_ave2[-(i+1), col_idx1-1] + pb * kxkyAmp_ave2[-(i+1), col_idx2-1]
                    K[i] = np.sqrt((KY3*i)**2 + x[i]**2)
        
        # 绘制角度切割线
        if theta != 90:
            plt.plot(-x.cpu().numpy(), y.cpu().numpy(), '-k', linewidth=2)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  保存到: {save_path}")
        else:
            plt.show()
        plt.close()
        
        # 绘制1D波数谱
        plt.figure(figsize=(10, 6))
        plt.plot(K.cpu().numpy(), np.log(kxkyAmp_ave3.cpu().numpy() + 1e-10), '-ok')
        plt.xlabel('kρ_i')
        plt.ylabel('Amplitude')
        plt.title(f'1D wavenumber spectrum at θ = {theta2}°')
        
        if save_path:
            cut_path = save_path.replace('.png', f'_cut_{theta2}deg.png')
            plt.savefig(cut_path, dpi=150, bbox_inches='tight')
            print(f"  保存1D谱到: {cut_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_mode7_multi_fluctuation(self, data_n_list: List[int], save_path: Optional[str] = None):
        """模式7：多波动分析 (简化版对应MATLAB模式9)"""
        import matplotlib.pyplot as plt
        
        print(f"\n模式7: 多波动分析 (数据列表: {data_n_list})")
        
        # 这里需要实际加载多个数据集
        # 暂时给出框架实现
        raise NotImplementedError("需要实现多数据集成分析功能")
    
    def analyze_mode8_video_animation(self, save_path: Optional[str] = None):
        """模式8：时间动画 (对应MATLAB模式16)"""
        import matplotlib.pyplot as plt
        import imageio
        
        if self.realSpaceData is None:
            raise ValueError("请先调用load_time_series()")
        
        print("\n模式8: 生成时间动画")
        
        frames = []
        Ny, Nx, Nt = self.realSpaceData.shape
        
        # 时间转换因子
        time_factor = self.config.L_ref / np.sqrt(2 * self.config.T_ref2 / self.config.m_ref)
        
        for i in range(min(Nt, 50)):  # 限制帧数避免过大文件
            fig, ax = plt.subplots(figsize=(10, 8))
            
            snapshot = self.realSpaceData[:, :, i].cpu().numpy()
            contour = ax.contourf(self.xx.cpu().numpy(), self.yy.cpu().numpy(), 
                                 snapshot, 100, 'LineStyle', 'none')
            ax.axis('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f't = {self.tt[i] / time_factor:.2f}')
            fig.colorbar(contour, ax=ax)
            
            # 保存帧
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)
        
        if save_path:
            video_path = save_path.replace('.png', '.gif') if save_path.endswith('.png') else save_path + '.gif'
            imageio.mimsave(video_path, frames, fps=10)
            print(f"  动画保存到: {video_path}")
        else:
            print("  动画已生成 (请提供save_path保存)")
    
    def analyze_all_modes_advanced(self, output_dir: str, enable_modes: List[int] = None):
        """批量生成所有分析图表 (包含高级模式)"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if enable_modes is None:
            enable_modes = [1, 2, 3, 4, 5, 6, 7, 8]
        
        print(f"\n批量生成所有分析图表到: {output_dir}")
        print(f"启用模式: {enable_modes}")
        
        if 1 in enable_modes:
            # 模式1: 几个代表性的频率切片
            for f_idx in [len(self.f)//4, len(self.f)//2, 3*len(self.f)//4]:
                self.analyze_mode1_kxky(f_idx, str(output_path / f'mode1_kxky_f{f_idx}.png'))
        
        if 2 in enable_modes:
            # 模式2: 几个代表性的kx切片
            for kx_idx in [len(self.kx)//4, len(self.kx)//2, 3*len(self.kx)//4]:
                self.analyze_mode2_wky(kx_idx, str(output_path / f'mode2_wky_kx{kx_idx}.png'))
        
        if 3 in enable_modes:
            # 模式3: 几个代表性的ky切片
            for ky_idx in [len(self.ky)//4, len(self.ky)//2, 3*len(self.ky)//4]:
                self.analyze_mode3_wkx(ky_idx, str(output_path / f'mode3_wkx_ky{ky_idx}.png'))
        
        if 4 in enable_modes:
            # 模式4: 几个时间快照
            Nt = self.realSpaceData.shape[2]
            for t_idx in range(0, Nt, max(1, Nt//10)):
                self.analyze_mode4_snapshot(t_idx, str(output_path / f'mode4_snapshot_t{t_idx}.png'))
            
            # 角度相关分析
            self.analyze_mode4_angle_fk_spectrum(45, str(output_path / 'mode4_angle_fk_45deg.png'))
        
        if 5 in enable_modes:
            # 模式5: 2D FFT
            self.analyze_mode5_2dfft(str(output_path / 'mode5_2dfft_average.png'))
        
        if 6 in enable_modes:
            # 模式6: 详细2D FFT分析
            kxkyAmp_ave, KX_fixed, KY_fixed = self.analyze_mode6_2dfft_detailed(
                str(output_path / 'mode6_2dfft_detailed.png'))
            
            # 带角度切割的分析
            self.analyze_mode6_with_angle_cut(45, kxkyAmp_ave, KX_fixed, KY_fixed,
                                             str(output_path / 'mode6_angle_cut_45deg.png'))
        
        if 7 in enable_modes:
            # 模式7: 多波动分析
            print("模式7需要多个数据集，暂未实现")
        
        if 8 in enable_modes:
            # 模式8: 动画
            self.analyze_mode8_video_animation(str(output_path / 'mode8_animation'))
        
        print("\n所有分析图表生成完成！")

    def plotWaveNumberSpace_2(self, xx: torch.Tensor, yy: torch.Tensor, 
                            realSpaceData: torch.Tensor, Mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D波数空间分析函数 (对应MATLAB的plotWaveNumberSpace_2)
        
        Args:
            xx: x坐标网格 (Ny, Nx)
            yy: y坐标网格 (Ny, Nx)
            realSpaceData: 输入数据 (Ny, Nx)
            Mean: 平均值用于去均值
            
        Returns:
            (Amp, KX, KY) - 振幅和波数网格
        """
        # 去除均值
        data_centered = realSpaceData - Mean
        Ny, Nx = data_centered.shape
        
        # 2D FFT
        P = torch.fft.fft2(data_centered)
        P_shifted = torch.fft.fftshift(P)
        Amp = torch.abs(P_shifted)
        
        # 计算空间分辨率
        dx = (xx[0, 1] - xx[0, 0]).item()
        dy = (yy[1, 0] - yy[0, 0]).item()
        
        # 生成波数向量 (严格按MATLAB逻辑)
        if Nx % 2 == 0:
            kx = 2 * np.pi * np.arange(-Nx//2, Nx//2) / ((Nx - 1) * dx)
        else:
            kx = 2 * np.pi * np.arange(-(Nx-1)//2, (Nx-1)//2 + 1) / ((Nx - 1) * dx)
        
        if Ny % 2 == 0:
            ky = 2 * np.pi * np.arange(-Ny//2, Ny//2) / ((Ny - 1) * dy)
        else:
            ky = 2 * np.pi * np.arange(-(Ny-1)//2, (Ny-1)//2 + 1) / ((Ny - 1) * dy)
        
        # 创建网格
        KY_np, KX_np = np.meshgrid(ky, kx, indexing='ij')
        KX = torch.from_numpy(KX_np).to(realSpaceData.device)
        KY = torch.from_numpy(KY_np).to(realSpaceData.device)
        
        # 归一化 (按FVER支持GENE和R5F)
        if self.FVER == 5:  # GENE
            if self.config.rho_ref is not None:
                kx = kx * self.config.rho_ref
                ky = ky * self.config.rho_ref
                KX = KX * self.config.rho_ref
                KY = KY * self.config.rho_ref
        else:  # R5F
            kx = kx * 0.003
            ky = ky * 0.003
            KX = KX * 0.003
            KY = KY * 0.003
        
        return Amp, KX, KY

