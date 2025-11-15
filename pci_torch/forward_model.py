"""
PCI正向模拟核心模型

整合所有模块，实现完整的PCI正向投影
"""

import torch
from typing import Optional, Tuple, Union, Dict
from .config import GENEConfig, BeamConfig
from .beam_geometry import compute_beam_grid
from .coordinates import cartesian_to_flux, cartesian_to_cylindrical
from .interpolation import probe_local_trilinear
from .utils import ensure_batch_dim, remove_batch_dim

def preprocess_density_field_for_probe(
    density_3d: torch.Tensor,
    config: GENEConfig,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    复刻 MATLAB probe_multi2 中的 p2 -> data3 预处理流程：
      1. 径向扩展：nx0 -> nx0+1（最后一列填 0）
      2. φ 方向周期边界：再加一层，复制第一层
      3. inside/outside 径向 padding：嵌入到 IRMAX+1 = nx0+inside+outside+1
      4. poloidal 重排：按照 NTGMAX/2 做循环移位
      5. poloidal 周期边界：再加一行，复制第一行

    输入:
      density_3d: (ntheta, nx0, nz) 或 (B, ntheta, nx0, nz)
    输出:
      processed: (ntheta+1, IRMAX+1, nz+1) 或 (B, ntheta+1, IRMAX+1, nz+1)
                 语义上对应 MATLAB 的 data3
    """
    density_3d = density_3d.to(device)

    # 统一成有 batch 维
    has_batch = (density_3d.dim() == 4)
    if not has_batch:
        density_3d = density_3d.unsqueeze(0)

    B, ntheta, nx0, nz = density_3d.shape

    inside = int(getattr(config, "inside", 0) or 0)
    outside = int(getattr(config, "outside", 0) or 0)
    KZMt = int(getattr(config, "KZMt", nz - 1))

    if nz != KZMt + 1:
        print(f"[WARN] preprocess_density_field_for_probe: "
              f"nz={nz} 与 KZMt+1={KZMt+1} 不一致，请检查 config.KZMt / fread_data_s")

    # === 1. 径向扩展：加一列 0 ===
    zero_col = torch.zeros(
        B, ntheta, 1, nz,
        dtype=density_3d.dtype, device=density_3d.device
    )
    p2_s = torch.cat([density_3d, zero_col], dim=2)  # (B, ntheta, nx0+1, nz)

    # === 2. φ 周期边界：最后一层复制第一层 ===
    first_phi = p2_s[..., :1]                         # (B, ntheta, nx0+1, 1)
    p2_s = torch.cat([p2_s, first_phi], dim=3)        # (B, ntheta, nx0+1, nz+1)

    # === 3. inside/outside 径向 padding ===
    nx_ext = nx0 + 1 + inside + outside               # IRMAX+1
    nz_ext = nz + 1
    data2 = torch.zeros(
        B, ntheta, nx_ext, nz_ext,
        dtype=density_3d.dtype, device=density_3d.device
    )
    # 对应 MATLAB: data2(:, inside+1:end-outside, :) = p2_s;
    data2[:, :, inside:inside + (nx0 + 1), :] = p2_s

    # === 4. poloidal 重排 ===
    # MATLAB:
    # for i = 1:NTGMAX
    #   md = mod(i + NTGMAX/2, NTGMAX);
    #   data3(md+1,:,:) = data2(i,:,:);
    # end
    NTGMAX = int(getattr(config, "NTGMAX", ntheta) or ntheta)

    if NTGMAX != ntheta:
        print(f"[WARN] preprocess_density_field_for_probe: "
              f"NTGMAX={NTGMAX} 与 ntheta={ntheta} 不一致，默认使用 ntheta")

        NTGMAX = ntheta

    idx_src = torch.arange(NTGMAX, device=density_3d.device)      # 0..NTGMAX-1，对应 MATLAB 的 i-1
    # md = mod(i + NTGMAX/2, NTGMAX)  (MATLAB, i=1..NTGMAX)
    # 0-based: j0 = (i0 + 1 + NTGMAX/2) mod NTGMAX
    idx_dst = (idx_src + NTGMAX // 2 + 1) % NTGMAX

    data3 = torch.zeros_like(data2)
    # data3[:, md, :, :] = data2[:, i, :, :]
    data3[:, idx_dst, :, :] = data2[:, idx_src, :, :]

    # === 5. poloidal 周期边界：最后一行复制第一行 ===
    first_theta = data3[:, :1, :, :]                  # (B, 1, nx_ext, nz_ext)
    data3 = torch.cat([data3, first_theta], dim=1)    # (B, ntheta+1, nx_ext, nz_ext)

    if not has_batch:
        data3 = data3[0]

    return data3


def _batch_probe_local_trilinear(
    density_3d: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    PHI: torch.Tensor,
    config,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    批量版本的probe_local_trilinear - 优化性能，保持MATLAB逻辑
    
    Args:
        density_3d: 密度场 (ntheta, nx, nz)
        R, Z, PHI: 坐标张量
        config: 配置对象
        device: 设备
    
    Returns:
        插值结果张量
    """
    N = len(R)
    results = torch.zeros(N, device=device, dtype=density_3d.dtype)
    
    # 批量处理：虽然逐点调用，但减少函数调用开销
    for i in range(N):
        results[i] = probe_local_trilinear(
            density_3d,
            R[i], Z[i], PHI[i],
            config
        )
    
    return results


def forward_projection(
    density_3d: torch.Tensor,
    config: GENEConfig,
    beam_config: BeamConfig,
    device: str = 'cuda',
    return_line_integral: bool = False,
    cache_beam_grid: Optional[dict] = None,
    return_debug_info: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    PCI正向投影：3D密度扰动 → 2D检测器信号
    
    完整流程：
    1. 生成光束采样网格（笛卡尔坐标）
    2. 转换到磁通坐标
    3. 从3D密度场插值采样
    4. 沿光束方向积分
    5. 返回2D检测器图像
    
    Args:
        density_3d: 3D密度场
            - shape: (B, ntheta, nx, nz) 批处理模式
            - 或: (ntheta, nx, nz) 单个场
        config: GENE配置（包含平衡态数据）
        beam_config: 光束配置
        device: PyTorch设备 ('cuda' 或 'cpu')
        return_line_integral: 如果True，返回未积分的3D数据
        cache_beam_grid: 预计算的光束网格（可选，用于加速）
        return_debug_info: 如果True，输出调试信息并保存Stage 1/2的调试文件（默认False）
    
    Returns:
        如果return_line_integral=False:
            - (B, n_detectors_v, n_detectors_t) 2D检测器图像
            - 或 (n_detectors_v, n_detectors_t) 如果输入无batch维度
        如果return_line_integral=True:
            - (B, n_detectors_v, n_detectors_t, n_beam_points) 
    
    完全可微分，支持批处理和自动微分
    """
    # === 0. 统一设备 & batch 维 ===
    density_3d = density_3d.to(device)
    density_3d, batch_added = ensure_batch_dim(density_3d)
    B, ntheta_raw, nx_raw, nz_raw = density_3d.shape
    
    # 预留一个 debug dict，后面一起返回
    extra_debug = {}
    
    # === 1. 断点1：保存"原始 p2" (MATLAB 的 p2) ===
    if return_debug_info:
        print("\n=== 断点1: 原始 3D 密度场 (p2) ===")
        
        import numpy as np
        from pathlib import Path
        import json
        
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        density_raw_np = density_3d.squeeze(0).cpu().numpy()
        np.savetxt(
            debug_dir / "stage1_3d_density_field_raw_p2.txt",
            density_raw_np.reshape(-1, nz_raw),
            fmt="%.8e"
        )
        
        density_raw_stats = {
            "stage": "1_raw_p2",
            "shape": [ntheta_raw, nx_raw, nz_raw],
            "min": float(density_raw_np.min()),
            "max": float(density_raw_np.max()),
            "mean": float(density_raw_np.mean()),
            "std": float(density_raw_np.std()),
            "nonzero_count": int(np.count_nonzero(density_raw_np)),
        }
        
        with open(debug_dir / "stage1_density_raw_stats.json", "w") as f:
            json.dump(density_raw_stats, f, indent=2)
        
        print(f"  shape: {ntheta_raw} × {nx_raw} × {nz_raw}")
        print(f"  range: [{density_raw_stats['min']:.3e}, {density_raw_stats['max']:.3e}]")
        
        # 也放进 debug_info 返回给上层 .mat/.npz
        extra_debug["density_raw_p2"] = density_3d.detach().clone()
    
    # === 1.5 NEW: 做 MATLAB 等价的预处理，得到 processed_density_field_py ===
    density_proc = preprocess_density_field_for_probe(
        density_3d, config, device=device
    )
    B, ntheta, nx, nz = density_proc.shape  # 注意这里的 nz 已经是 KZMt+2
    
    # 保存 processed 版本，方便对比 MATLAB data3
    if return_debug_info:
        import numpy as np
        from pathlib import Path
        import json
        
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        density_proc_np = density_proc.squeeze(0).cpu().numpy()
        # 为了看清楚 φ 方向结构，用 (ntheta+1)*nx 行，每行 nz 列
        np.savetxt(
            debug_dir / "stage1b_processed_density_field_py.txt",
            density_proc_np.reshape(-1, nz),
            fmt="%.8e"
        )
        
        density_proc_stats = {
            "stage": "1b_processed_density_py",
            "shape": [ntheta, nx, nz],
            "min": float(density_proc_np.min()),
            "max": float(density_proc_np.max()),
            "mean": float(density_proc_np.mean()),
            "std": float(density_proc_np.std()),
            "nonzero_count": int(np.count_nonzero(density_proc_np)),
        }
        
        with open(debug_dir / "stage1b_density_processed_stats.json", "w") as f:
            json.dump(density_proc_stats, f, indent=2)
        
        print(f"=== 断点1b: 预处理后的 3D 密度场 (Python data3 等价) ===")
        print(f"  shape: {ntheta} × {nx} × {nz}")
        print(f"  range: [{density_proc_stats['min']:.3e}, {density_proc_stats['max']:.3e}]")
        
        # 这两个变量会被 run_single_time 存进 .mat / .npz 里
        extra_debug["processed_density_field_py"] = density_proc.detach().clone()
        extra_debug["processed_density_field_py_shape"] = [ntheta, nx, nz]
    
    # 后续几何 / 插值都用预处理后的场
    density_3d = density_proc
    
    # === 原有"断点2: 几何计算阶段"保持不变 ===
    if cache_beam_grid is None:
        beam_grid = compute_beam_grid(beam_config, config=config, device=device, debug=return_debug_info)
    else:
        beam_grid = cache_beam_grid
    
    if return_debug_info:
        print("\n=== 断点2: 几何计算阶段 ===")
        
        import numpy as np
        from pathlib import Path
        import json
        
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        grid_xyz = beam_grid['grid_xyz']  # (n_det_v, n_det_t, n_beam, 3)
        n_det_v, n_det_t, n_beam, _ = grid_xyz.shape
        
        # 保存光束几何数据
        grid_xyz_numpy = grid_xyz.cpu().numpy()
        
        # 保存坐标数据
        x_coords = grid_xyz_numpy[:,:,:,0].flatten()  # X坐标
        y_coords = grid_xyz_numpy[:,:,:,1].flatten()  # Y坐标  
        z_coords = grid_xyz_numpy[:,:,:,2].flatten()  # Z坐标
        
        np.savetxt(debug_dir / "stage2_beam_geometry_x.txt", x_coords, fmt='%.8e')
        np.savetxt(debug_dir / "stage2_beam_geometry_y.txt", y_coords, fmt='%.8e')
        np.savetxt(debug_dir / "stage2_beam_geometry_z.txt", z_coords, fmt='%.8e')
        
        # 保存光束几何统计信息
        beam_stats = {
            'stage': '2_beam_geometry_completed',
            'shape': [n_det_v, n_det_t, n_beam, 3],
            'x_range': [float(x_coords.min()), float(x_coords.max())],
            'y_range': [float(y_coords.min()), float(y_coords.max())],
            'z_range': [float(z_coords.min()), float(z_coords.max())],
            'total_points': int(len(x_coords))
        }
        
        with open(debug_dir / "stage2_beam_stats.json", 'w') as f:
            json.dump(beam_stats, f, indent=2)
        
        print(f"✓ 断点2: 光束几何数据已保存")
        print(f"  - 网格形状: {n_det_v} × {n_det_t} × {n_beam} × 3")
        print(f"  - 总采样点数: {len(x_coords)}")
        print(f"  - X范围: [{beam_stats['x_range'][0]:.3f}, {beam_stats['x_range'][1]:.3f}]")
        print(f"  - Y范围: [{beam_stats['y_range'][0]:.3f}, {beam_stats['y_range'][1]:.3f}]")
        print(f"  - Z范围: [{beam_stats['z_range'][0]:.3f}, {beam_stats['z_range'][1]:.3f}]")
        print("=== 退出: 几何计算阶段完成 ===\n")

    # === 插值阶段 ===
    grid_xyz = beam_grid["grid_xyz"]               # (n_det_v, n_det_t, n_beam, 3)
    n_det_v, n_det_t, n_beam, _ = grid_xyz.shape
    
    R_coords = grid_xyz[:, :, :, 0].flatten()
    Z_coords = grid_xyz[:, :, :, 1].flatten()
    PHI_coords = grid_xyz[:, :, :, 2].flatten()
    
    from .interpolation import probe_local_trilinear
    
    interpolated_values = probe_local_trilinear(
        density_3d,       # 注意这里已经是 processed 版本
        R_coords, Z_coords, PHI_coords,
        config
    )
    
    pout1 = interpolated_values.reshape(n_det_v, n_det_t, n_beam)
    pout2 = torch.sum(pout1, dim=2)
    
    if return_line_integral:
        # 你的原有 dims_info 构造代码...
        dims_info = {
            "n_det_v": n_det_v,
            "n_det_t": n_det_t,
            "n_beam": n_beam,
            "total_points": len(interpolated_values),
            "return_line_integral": True,
        }
        if return_debug_info:
            dims_info.update(extra_debug)
        debug_info = dims_info
        result = pout1
    else:
        base_info = {
            "n_det_v": n_det_v,
            "n_det_t": n_det_t,
            "n_beam": n_beam,
            "pout1_shape": list(pout1.shape),
            "pout2_shape": list(pout2.shape),
        }
        if return_debug_info:
            base_info.update(extra_debug)
        debug_info = base_info
        result = pout2
    
    if return_debug_info:
        print(f"✓ forward_projection 完成, 结果 shape = {tuple(result.shape)}")
    
    return result, debug_info


def forward_projection_with_preprocessing(
    density_3d: torch.Tensor,
    config: GENEConfig,
    beam_config: BeamConfig,
    device: str = 'cuda',
    subtract_mean: bool = True,
    normalize: bool = False
) -> Tuple[torch.Tensor, dict]:
    """
    带预处理的正向投影
    
    Args:
        density_3d: 3D密度场
        config: GENE配置
        beam_config: 光束配置
        device: 设备
        subtract_mean: 是否减去平均值（湍流涨落）
        normalize: 是否归一化输出
    
    Returns:
        (pci_image, metadata):
            - pci_image: 2D检测器图像
            - metadata: 包含统计信息的字典
    """
    # 预处理：减去平均值（提取涨落分量）
    if subtract_mean:
        mean_density = torch.mean(density_3d, dim=(-3, -2, -1), keepdim=True)
        density_fluctuation = density_3d - mean_density
    else:
        density_fluctuation = density_3d
        mean_density = None
    
    # 正向投影
    pci_result, metadata = forward_projection(
        density_fluctuation,
        config,
        beam_config,
        device=device
    )
    
    # 后处理：归一化
    if normalize:
        pci_min = pci_result.min()
        pci_max = pci_result.max()
        if pci_max > pci_min:
            pci_result = (pci_result - pci_min) / (pci_max - pci_min)
    
    # 收集元数据
    metadata = {
        'mean_density': mean_density,
        'pci_min': pci_result.min().item(),
        'pci_max': pci_result.max().item(),
        'pci_mean': pci_result.mean().item(),
        'pci_std': pci_result.std().item(),
    }
    
    return pci_result, metadata


def batch_forward_projection(
    density_3d_list: list,
    config: GENEConfig,
    beam_config: BeamConfig,
    device: str = 'cuda',
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """
    批量处理多个密度场
    
    Args:
        density_3d_list: 密度场列表 [tensor1, tensor2, ...]
        config: GENE配置
        beam_config: 光束配置
        device: 设备
        batch_size: 批大小（None表示一次处理全部）
    
    Returns:
        (N, n_det_v, n_det_t) 所有结果堆叠
    """
    # 预计算光束网格（所有批次共享）
    beam_grid = compute_beam_grid(beam_config, device=device)
    
    results = []
    
    if batch_size is None:
        # 一次处理全部
        density_batch = torch.stack(density_3d_list, dim=0)
        result = forward_projection(
            density_batch,
            config,
            beam_config,
            device=device,
            cache_beam_grid=beam_grid
        )
        return result
    else:
        # 分批处理
        n_samples = len(density_3d_list)
        for i in range(0, n_samples, batch_size):
            batch = density_3d_list[i:i+batch_size]
            density_batch = torch.stack(batch, dim=0)
            
            result = forward_projection(
                density_batch,
                config,
                beam_config,
                device=device,
                cache_beam_grid=beam_grid
            )
            results.append(result)
        
        return torch.cat(results, dim=0)


def compute_fft_spectrum(
    pci_images: torch.Tensor,
    spatial_grid: Tuple[torch.Tensor, torch.Tensor],
    config: GENEConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算PCI图像的2D傅里叶频谱
    
    对应MATLAB的plotWaveNumberSpace函数
    
    Args:
        pci_images: PCI图像 (B, H, W) 或 (H, W)
        spatial_grid: (yy, xx) 空间网格坐标
        config: GENE配置
    
    Returns:
        (kx, ky, spectrum):
            - kx: x方向波数 (Nx,)
            - ky: y方向波数 (Ny,)
            - spectrum: 功率谱 (B, Ny, Nx) 或 (Ny, Nx)
    """
    # 确保有batch维度
    pci_images, batch_added = ensure_batch_dim(pci_images)
    B, Ny, Nx = pci_images.shape
    
    # 减去平均值
    mean_val = torch.mean(pci_images, dim=(-2, -1), keepdim=True)
    pci_centered = pci_images - mean_val
    
    # 2D FFT
    fft_result = torch.fft.fft2(pci_centered)
    fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
    spectrum = torch.abs(fft_shifted)
    
    # 计算波数轴
    yy, xx = spatial_grid
    dx = (xx[0, 1] - xx[0, 0]).item()
    dy = (yy[1, 0] - yy[0, 0]).item()
    
    if Nx % 2 == 0:
        kx = 2 * torch.pi * torch.arange(-Nx//2, Nx//2, device=pci_images.device) / ((Nx-1) * dx)
    else:
        kx = 2 * torch.pi * torch.arange(-(Nx-1)//2, (Nx-1)//2+1, device=pci_images.device) / ((Nx-1) * dx)
    
    if Ny % 2 == 0:
        ky = 2 * torch.pi * torch.arange(-Ny//2, Ny//2, device=pci_images.device) / ((Ny-1) * dy)
    else:
        ky = 2 * torch.pi * torch.arange(-(Ny-1)//2, (Ny-1)//2+1, device=pci_images.device) / ((Ny-1) * dy)
    
    # 归一化波数（使用回旋半径）
    if config.rho_ref is not None:
        kx = kx * config.rho_ref
        ky = ky * config.rho_ref
    
    # 移除batch维度（如果原本没有）
    spectrum = remove_batch_dim(spectrum, batch_added)
    
    return kx, ky, spectrum


def differentiable_forward_projection(
    density_3d: torch.Tensor,
    config: GENEConfig,
    beam_config: BeamConfig,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    可微分的正向投影（确保梯度传播）
    
    用于逆问题求解和神经网络训练
    
    Args:
        density_3d: 3D密度场 (requires_grad=True)
        config: GENE配置
        beam_config: 光束配置
        device: 设备
    
    Returns:
        PCI图像（支持反向传播）
    """
    # 确保输入需要梯度
    if not density_3d.requires_grad:
        density_3d = density_3d.clone().requires_grad_(True)
    
    # 调用标准正向投影
    pci_image = forward_projection(
        density_3d,
        config,
        beam_config,
        device=device
    )
    
    return pci_image

