"""
PCI正向模拟核心模型

整合所有模块，实现完整的PCI正向投影
"""

import torch
from typing import Optional, Tuple
from .config import GENEConfig, BeamConfig
from .beam_geometry import compute_beam_grid
from .coordinates import cartesian_to_flux, cartesian_to_cylindrical
from .interpolation import probe_local_trilinear
from .utils import ensure_batch_dim, remove_batch_dim


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
    cache_beam_grid: Optional[dict] = None
) -> torch.Tensor:
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
    
    Returns:
        如果return_line_integral=False:
            - (B, n_detectors_v, n_detectors_t) 2D检测器图像
            - 或 (n_detectors_v, n_detectors_t) 如果输入无batch维度
        如果return_line_integral=True:
            - (B, n_detectors_v, n_detectors_t, n_beam_points) 
    
    完全可微分，支持批处理和自动微分
    """
    # 确保density_3d在正确的设备上
    density_3d = density_3d.to(device)
    
    # 处理batch维度
    density_3d, batch_added = ensure_batch_dim(density_3d)
    B, ntheta, nx, nz = density_3d.shape
    
    # 步骤1: 生成或使用缓存的光束网格
    if cache_beam_grid is None:
        beam_grid = compute_beam_grid(beam_config, device=device)
    else:
        beam_grid = cache_beam_grid
    
    grid_xyz = beam_grid['grid_xyz']  # (n_det_v, n_det_t, n_beam, 3)
    n_det_v, n_det_t, n_beam, _ = grid_xyz.shape
    
    # 步骤2: 转换光束网格到柱坐标
    # Reshape为 (N, 3) 进行批量转换
    grid_flat = grid_xyz.reshape(-1, 3)
    x, y, z = grid_flat[:, 0], grid_flat[:, 1], grid_flat[:, 2]
    
    # 使用精确的probe_local_trilinear（对应MATLAB的probeEQ_local_s）
    R, Z, phi = cartesian_to_cylindrical(x, y, z)
    
    # 步骤3: 使用probe_local_trilinear进行精确插值（单点版本）
    # 优化版本：保持MATLAB逻辑的同时提升性能
    sampled_values_list = []
    for b in range(B):
        density_single = density_3d[b]  # (ntheta, nx, nz)
        
        # 批量处理插值（保持MATLAB逐点逻辑，提升计算效率）
        sampled_flat = _batch_probe_local_trilinear(
            density_single, R, Z, phi, config, device
        )
        
        # Reshape回网格形状
        sampled_grid = sampled_flat.reshape(n_det_v, n_det_t, n_beam)
        sampled_values_list.append(sampled_grid)
    
    sampled_values = torch.stack(sampled_values_list, dim=0)
    # shape: (B, n_det_v, n_det_t, n_beam)
    
    if return_line_integral:
        result = sampled_values
    else:
        # 步骤4: 沿光束方向积分（求和）
        result = torch.sum(sampled_values, dim=-1)
        # shape: (B, n_det_v, n_det_t)
    
    # 移除batch维度（如果原本没有）
    result = remove_batch_dim(result, batch_added)
    
    return result


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
    pci_image = forward_projection(
        density_fluctuation,
        config,
        beam_config,
        device=device
    )
    
    # 后处理：归一化
    if normalize:
        pci_min = pci_image.min()
        pci_max = pci_image.max()
        if pci_max > pci_min:
            pci_image = (pci_image - pci_min) / (pci_max - pci_min)
    
    # 收集元数据
    metadata = {
        'mean_density': mean_density,
        'pci_min': pci_image.min().item(),
        'pci_max': pci_image.max().item(),
        'pci_mean': pci_image.mean().item(),
        'pci_std': pci_image.std().item(),
    }
    
    return pci_image, metadata


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

