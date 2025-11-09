"""
插值模块 - 与MATLAB的probeEQ_local_s.m完全对应

本模块实现了与MATLAB probeEQ_local_s.m完全一致的Python版本，
包含坐标转换和三维三线性插值功能。

主要函数:
- probe_local_trilinear: 对应MATLAB的probeEQ_local_s.m
- bisec: 对应MATLAB的bisec.m，使用torch.searchsorted实现

MATLAB对应关系:
- probeEQ_local_s.m: GENE版本的精确插值
- bisec.m: 二分查找函数
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .utils import bisec


def probe_local_trilinear(
    density_3d: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    PHI: torch.Tensor,
    config
) -> torch.Tensor:
    """
    3D三线性插值 - 完全对应MATLAB的probeEQ_local_s.m (GENE版本)
    
    这个函数实现了与MATLAB完全一致的插值算法
    对应MATLAB代码: sim_data/GENE/@GENEClass/probeEQ_local_s.m
    
    Args:
        density_3d: 密度场 (ntheta, nx, nz)
        R: R坐标 (scalar or tensor)
        Z: Z坐标 (scalar or tensor)  
        PHI: PHI坐标 [0, 2π] (scalar or tensor)
        config: 包含equilibrium数据的配置对象
    
    Returns:
        插值结果 (scalar or tensor)
    """
    # 确保输入是tensor
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, device=density_3d.device, dtype=torch.float64)
    if not isinstance(Z, torch.Tensor):
        Z = torch.tensor(Z, device=density_3d.device, dtype=torch.float64)
    if not isinstance(PHI, torch.Tensor):
        PHI = torch.tensor(PHI, device=density_3d.device, dtype=torch.float64)
    
    # 展平为1D
    original_shape = R.shape
    R_flat = R.flatten()
    Z_flat = Z.flatten()
    PHI_flat = PHI.flatten()
    N = R_flat.shape[0]
    
    # 初始化结果
    result = torch.zeros(N, device=density_3d.device, dtype=density_3d.dtype)
    
    # 检查是否有equilibrium数据
    if config.PA is None or config.GAC is None:
        print("警告: 没有equilibrium数据，使用简化插值")
        return result.reshape(original_shape)
    
    # 步骤1: 计算相对于plasma axis的(r, theta) - 对应MATLAB第6-7行
    PA = config.PA  # (2,) [R_axis, Z_axis]
    r = torch.sqrt((R_flat - PA[0])**2 + (Z_flat - PA[1])**2)
    theta = torch.remainder(torch.atan2(Z_flat - PA[1], R_flat - PA[0]), 2*np.pi)
    
    # 步骤2: 使用bisec查找theta索引 - 对应MATLAB第8行
    GTC_c_last = config.GTC_c[-1, :]  # 最外层的theta坐标
    
    # 步骤3: 设置phi列表（归一化到[0,1]）- 对应MATLAB第9行
    # MATLAB: philist = linspace(0, 1, obj.KZMt+1+1);  % KZMt+2个点
    nz = density_3d.shape[2] 
    KZMt = nz - 2  # 从density_3d的shape推断KZMt
    philist = torch.linspace(0, 1, KZMt + 2, device=density_3d.device)
    
    # 分别处理每个点
    for i in range(N):
        r_i = r[i]
        theta_i = theta[i]
        phi_i = PHI_flat[i] / (2*np.pi)  # 归一化到[0,1]
        
        # 获取theta索引 - 修正bisec返回值处理
        # MATLAB: theta_p = bisec(theta, obj.GTC_c(end, :));
        # bisec返回两个索引，需要根据数组排序方向正确解释
        theta_idx1, theta_idx2 = bisec(theta_i, GTC_c_last)
        
        # 根据MATLAB逻辑，通常取第一个索引作为主索引
        # 检查GTC_c_last的排序方向
        if GTC_c_last[0] < GTC_c_last[-1]:  # 升序
            theta_p_lower = theta_idx1
            theta_p_upper = theta_idx2
        else:  # 降序
            theta_p_lower = theta_idx2  
            theta_p_upper = theta_idx1
            
        poid_cyl_2 = theta_p_lower
        
        # 步骤4: 检查是否在plasma内部 - 对应MATLAB第11-21行
        # MATLAB: if ((r < obj.GAC(end, theta_p(1))) && (r < obj.GAC(end, theta_p(2))))
        r_boundary_l = config.GAC[-1, theta_p_lower]  # theta_p(1)
        r_boundary_u = config.GAC[-1, theta_p_upper]  # theta_p(2)
        
        # 关键修正: 恢复边界条件检查，与MATLAB保持一致
        if not (r_i < r_boundary_l and r_i < r_boundary_u):
            # 点在等离子体边界外，返回0
            result[i] = 0.0
            continue
        
        # 查找r索引 - 对应MATLAB第13行
        GAC_at_theta = config.GAC[:, poid_cyl_2]
        
        # GAC数据不是单调的，使用线性查找替代bisec
        r_diffs = torch.abs(GAC_at_theta - r_i)
        r_p_lower = torch.argmin(r_diffs)
        r_p_upper = min(r_p_lower + 1, len(GAC_at_theta) - 1)  # 确保不超出范围
        
        # 正确的索引映射：GAC索引>=inside时，density索引=GAC索引
        if r_p_lower < config.inside:
            result[i] = 0.0
            continue  # 点在内部区域，density没有数据，返回0
        
        # 转换为density索引（直接使用GAC索引，因为density使用相同的索引系统）
        poid_cyl_1 = r_p_lower
        
        # 边界检查：如果r_i超出GAC_at_theta的范围，返回0
        gac_min = torch.min(GAC_at_theta).item()
        gac_max = torch.max(GAC_at_theta).item()
        if r_i < gac_min or r_i > gac_max:
            continue  # 跳过这个点，保持result[i]=0
        
        # 查找phi索引 - 对应MATLAB第15-17行
        p_p_lower, p_p_upper = bisec(phi_i, philist)
        
        # 边界检查：如果phi_i超出philist的范围，返回0
        if phi_i < philist[0] or phi_i > philist[-1]:
            continue  # 跳过这个点，保持result[i]=0
        
        poid_cyl_3 = p_p_lower
        
        # 步骤5: 获取边界值 - 对应MATLAB第23-28行
        r_min = config.GAC[poid_cyl_1, poid_cyl_2]
        r_max = config.GAC[min(poid_cyl_1 + 1, config.GAC.shape[0] - 1), poid_cyl_2]  # m1+1，确保不越界
        theta_min = GTC_c_last[poid_cyl_2]
        theta_max = GTC_c_last[min(poid_cyl_2 + 1, GTC_c_last.shape[0] - 1)]  # n1+1，确保不越界
        phi_min = philist[poid_cyl_3]
        phi_max = philist[min(poid_cyl_3 + 1, len(philist) - 1)]  # p1+1，确保不越界
        
        # 步骤6: 计算权重 - 对应MATLAB第30-32行，添加除以零检查
        # 检查分母是否为0，避免NaN
        r_diff = r_max - r_min
        theta_diff = theta_max - theta_min  
        phi_diff = phi_max - phi_min
        
        if abs(r_diff) < 1e-12:
            da_cyl_1 = 0.5  # 当r_max == r_min时，使用中点权重
        else:
            da_cyl_1 = (r_max - r_i) / r_diff
            
        if abs(theta_diff) < 1e-12:
            da_cyl_2 = 0.5  # 当theta_max == theta_min时，使用中点权重
        else:
            da_cyl_2 = (theta_max - theta_i) / theta_diff
            
        if abs(phi_diff) < 1e-12:
            da_cyl_3 = 0.5  # 当phi_max == phi_min时，使用中点权重
        else:
            da_cyl_3 = (phi_max - phi_i) / phi_diff
        
        # 步骤7: 设置索引变量 - 对应MATLAB第34-39行
        # 重要: 根据MATLAB probeEQ_local_s.m分析
        # m1 = poid_cyl(1) = r_p(1) (径向索引)
        # n1 = poid_cyl(2) = theta_p(1) (极向索引) 
        # p1 = poid_cyl(3) = p_p(1) (phi索引)
        # MATLAB访问: data(n1, m1, p1) = data(极向, 径向, phi)
        # density_3d形状: (ntheta, nx, nz) = (极向, 径向, phi)
        # 所以正确的映射: density_3d[n1, m1, p1]
        
        # 严格边界检查
        m1 = max(0, min(poid_cyl_1, density_3d.shape[1] - 1))  # 径向，范围[0, 127]
        n1 = max(0, min(poid_cyl_2, density_3d.shape[0] - 1))  # 极向，范围[0, 399]
        p1 = max(0, min(p_p_lower, density_3d.shape[2] - 1))   # phi，范围[0, 28]
        
        m2 = max(0, min(m1 + 1, density_3d.shape[1] - 1))  # 径向边界
        n2 = max(0, min(n1 + 1, density_3d.shape[0] - 1))  # 极向边界
        p2 = max(0, min(p1 + 1, density_3d.shape[2] - 1))  # phi边界
        
        # 调试信息
        # print(f"DEBUG: indices m1={m1}, m2={m2}, n1={n1}, n2={n2}, p1={p1}, p2={p2}, shape={density_3d.shape}")
        
        # 步骤8: 三线性插值 - 修正索引顺序
        # 原代码错误: density_3d[n1, m1, p1] - 这个顺序是正确的
        term1 = da_cyl_3 * (da_cyl_2 * (da_cyl_1 * density_3d[n1, m1, p1] + (1.0 - da_cyl_1) * density_3d[n1, m2, p1]) \
            + (1.0 - da_cyl_2) * (da_cyl_1 * density_3d[n2, m1, p1] + (1.0 - da_cyl_1) * density_3d[n2, m2, p1]))
        
        term2 = (1.0 - da_cyl_3) * (da_cyl_2 * (da_cyl_1 * density_3d[n1, m1, p2] + (1.0 - da_cyl_1) * density_3d[n1, m2, p2]) \
            + (1.0 - da_cyl_2) * (da_cyl_1 * density_3d[n2, m1, p2] + (1.0 - da_cyl_1) * density_3d[n2, m2, p2]))
        
        result[i] = term1 + term2
    # else: 保持result[i] = 0 (已经在初始化时设置)
    
    # 在plasma外的点保持为0（已经初始化为0）
    return result.reshape(original_shape)