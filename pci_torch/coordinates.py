"""
坐标系统转换模块

严格对应MATLAB中的坐标转换实现:
- cartesian_to_cylindrical: 笛卡尔坐标 → 柱坐标 (R, Z, phi)
- cylindrical_to_cartesian: 柱坐标 → 笛卡尔坐标
- cylindrical_to_flux: 柱坐标 → 磁通坐标 (rho, theta, phi) 
- cartesian_to_flux: 笛卡尔坐标 → 磁通坐标

对应MATLAB:
- LSview_com.m (第67-69行): 柱坐标到笛卡尔坐标
- probeEQ_local_s.m: 局部插值和坐标转换
"""

import torch
import numpy as np
from typing import Tuple, Optional


def cartesian_to_cylindrical(
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    笛卡尔坐标转换为柱坐标
    
    对应MATLAB的逆向转换 (LSview_com.m 第67-69行):
    MATLAB: 
        B2(:,1)=B1(:,1).*cos(2*pi*B1(:,3));  % x = R*cos(2π*phi)
        B2(:,2)=B1(:,1).*sin(2*pi*B1(:,3));  % y = R*sin(2π*phi)
        B2(:,3)=B1(:,2);                      % z = Z
    
    Args:
        x, y, z: 笛卡尔坐标张量
        
    Returns:
        R, Z, phi: 柱坐标张量
        - R: 径向距离 (sqrt(x² + y²))
        - Z: 轴向坐标 
        - phi: 角度坐标 [0, 1] (对应MATLAB中的0到2π范围)
    """
    # 计算径向距离
    R = torch.sqrt(x**2 + y**2)
    
    # 轴向坐标保持不变
    Z = z
    
    # 计算角度，转换到[0,1]范围以匹配MATLAB
    phi = torch.atan2(y, x) / (2 * np.pi)
    
    # 确保phi在[0,1]范围内
    phi = torch.remainder(phi, 1.0)
    
    return R, Z, phi


def cylindrical_to_cartesian(
    R: torch.Tensor, 
    Z: torch.Tensor, 
    phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    柱坐标转换为笛卡尔坐标
    
    对应MATLAB (LSview_com.m 第67-69行):
        B2(:,1)=B1(:,1).*cos(2*pi*B1(:,3));  % x = R*cos(2π*phi)
        B2(:,2)=B1(:,1).*sin(2*pi*B1(:,3));  % y = R*sin(2π*phi)
        B2(:,3)=B1(:,2);                      % z = Z
    
    Args:
        R, Z, phi: 柱坐标张量 (phi在[0,1]范围)
        
    Returns:
        x, y, z: 笛卡尔坐标张量
    """
    # 将phi转换到弧度
    phi_rad = phi * 2 * np.pi
    
    # 计算笛卡尔坐标
    x = R * torch.cos(phi_rad)
    y = R * torch.sin(phi_rad)
    z = Z
    
    return x, y, z


def cylindrical_to_flux(
    R: torch.Tensor, 
    Z: torch.Tensor, 
    phi: torch.Tensor, 
    config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    柱坐标转换为磁通坐标
    
    对应MATLAB: probeEQ_local_s.m 中的坐标转换逻辑
    
    Args:
        R, Z, phi: 柱坐标张量
        config: 包含equilibrium数据的配置对象
        
    Returns:
        rho, theta, phi_flux: 磁通坐标张量
        - rho: 归一化径向坐标 [0,1]
        - theta: 角度坐标 [-1,1] (对应MATLAB中的-pi到pi范围)
        - phi_flux: 环向坐标 [0,1]
    """
    # 如果没有equilibrium数据，使用简化转换
    if config.PA is None or config.GAC is None:
        # 简化模式：直接归一化
        R_max = config.R if hasattr(config, 'R') else torch.max(R)
        rho = R / R_max
        theta = (phi - 0.5) * 2  # 转换到[-1,1]范围
        phi_flux = phi
        return rho, theta, phi_flux
    
    # 相对于plasma axis的坐标
    PA = config.PA  # (2,) [R_axis, Z_axis]
    r_rel = torch.sqrt((R - PA[0])**2 + (Z - PA[1])**2)
    
    # 计算相对于axis的theta
    theta_raw = torch.atan2(Z - PA[1], R - PA[0])
    # 转换到[-1,1]范围以匹配MATLAB的th0参数
    theta = theta_raw / np.pi
    
    # 计算rho (径向归一化坐标)
    # 使用二分查找找到对应的flux surface
    rho = torch.zeros_like(r_rel)
    
    # 找到每个theta对应的最大R值作为边界
    for i in range(len(r_rel)):
        r_target = r_rel[i]
        theta_val = theta[i]
        
        # 找到最接近的theta索引
        if hasattr(config, 'GTC_c') and config.GTC_c is not None:
            theta_grid = config.GTC_c[-1, :]  # 最外层 flux surface的theta网格
            theta_idx = torch.argmin(torch.abs(theta_grid - theta_raw[i]))
            r_boundary = config.GAC[-1, theta_idx]  # 对应的边界R值
            rho[i] = r_rel[i] / r_boundary
        else:
            # 如果没有equilibrium数据，使用简化计算
            rho[i] = r_rel[i] / torch.max(r_rel) if torch.max(r_rel) > 0 else 0.0
    
    # 确保rho在[0,1]范围内
    rho = torch.clamp(rho, 0.0, 1.0)
    
    # phi_flux保持原始的phi值
    phi_flux = phi
    
    return rho, theta, phi_flux


def cartesian_to_flux(
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor, 
    config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    笛卡尔坐标转换为磁通坐标
    
    组合上述两个转换函数:
    1. cartesian_to_cylindrical: (x,y,z) → (R,Z,phi)
    2. cylindrical_to_flux: (R,Z,phi) → (rho,theta,phi_flux)
    
    对应MATLAB: 集成在probeEQ_*系列函数中
    
    Args:
        x, y, z: 笛卡尔坐标张量
        config: 包含equilibrium数据的配置对象
        
    Returns:
        rho, theta, phi_flux: 磁通坐标张量
    """
    # 步骤1: 笛卡尔坐标 → 柱坐标
    R, Z, phi = cartesian_to_cylindrical(x, y, z)
    
    # 步骤2: 柱坐标 → 磁通坐标
    rho, theta, phi_flux = cylindrical_to_flux(R, Z, phi, config)
    
    return rho, theta, phi_flux

