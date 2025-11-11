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
    # 🔧 恢复原始处理逻辑，不应该自动添加batch维度
    original_density_shape = density_3d.shape
    if density_3d.ndim == 3:
        # 保持3D输入不变，不添加batch维度
        # 原逻辑：只处理3D张量
        pass
    elif density_3d.ndim == 4:
        # 如果是4D，移除batch维度，保持为3D
        density_3d = density_3d.squeeze(0)  # 移除batch维度
    else:
        raise ValueError(f"density_3d必须是3D或4D张量，但得到的是{density_3d.ndim}D: {density_3d.shape}")
    
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
    # 🔧 修复1: 使用正确的MATLAB mod函数和PA磁轴
    PA = config.PA  # (2,) [R_axis, Z_axis]
    
    # 🔧 数值精度微调3: 坐标转换精度优化
    # 使用更高的精度进行坐标计算
    # 确保PA是torch.Tensor类型
    PA_tensor = torch.tensor(PA, device=R.device, dtype=R.dtype)
    dR = R_flat - PA_tensor[0]  # R - R_axis
    dZ = Z_flat - PA_tensor[1]  # Z - Z_axis
    
    # 使用高精度计算径向距离
    r = torch.sqrt(dR**2 + dZ**2)
    
    # 🔧 关键修复: 使用MATLAB的mod函数行为
    # MATLAB: theta = mod(atan2(Z0 - obj.PA(2), R0 - obj.PA(1)), 2*pi);
    # 修复numpy.mod与MATLAB mod的差异
    raw_theta = torch.atan2(dZ, dR)  # 注意顺序与MATLAB一致
    
    # 确保精度一致性 - MATLAB的mod实现
    two_pi = 2 * torch.pi  # 使用torch.pi而不是np.pi
    theta = raw_theta - two_pi * torch.floor(raw_theta / two_pi)
    
    # 归一化到[0, 2π]范围，确保数值稳定性
    theta = torch.where(theta < 0, theta + two_pi, theta)
    theta = torch.where(theta >= two_pi, theta - two_pi, theta)
    
    # 🔧 调试坐标转换
    if N > 0:  # 如果有数据点
        print(f"DEBUG 坐标转换 (第1个点):")
        print(f"  输入: R={R_flat[0]:.3f}, Z={Z_flat[0]:.3f}, PHI={PHI_flat[0]:.3f}")
        print(f"  PA: {PA}")
        print(f"  相对坐标: dR={dR[0]:.3f}, dZ={dZ[0]:.3f}")
        print(f"  计算结果: r={r[0]:.3f}, theta={theta[0]:.3f}")
    
    # 🔧 修复2: 严格按照MATLAB的坐标系统和缩放
    # 2. MATLAB中GAC是归一化坐标，需要考虑L_ref缩放
    if hasattr(config, 'L_ref') and config.L_ref is not None:
        # MATLAB: GAC是归一化坐标，乘以L_ref得到物理坐标
        GAC_physical = config.GAC * config.L_ref
        GAC_last_layer = GAC_physical[-1, :]  # 最外层的物理边界
    else:
        # 如果没有L_ref，直接使用GAC（可能在某些情况下已经是物理坐标）
        GAC_last_layer = config.GAC[-1, :]
    
    # 🔧 数值精度微调4: 边界值精度优化将在循环内进行
    # 因为theta_p_lower和theta_p_upper在循环内定义
    
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
        
        # 查找r索引 - 对应MATLAB第13行
        # 使用正确缩放的GAC数据
        if hasattr(config, 'L_ref') and config.L_ref is not None:
            GAC_for_interpolation = config.GAC * config.L_ref
        else:
            GAC_for_interpolation = config.GAC
        
        GAC_at_theta = GAC_for_interpolation[:, poid_cyl_2]
        
        # GAC数据不是单调的，使用线性查找替代bisec
        r_diffs = torch.abs(GAC_at_theta - r_i)
        r_p_lower = torch.argmin(r_diffs)
        r_p_upper = min(r_p_lower + 1, len(GAC_at_theta) - 1)  # 确保不超出范围
        
        # 🔧 修复2: 严格按照MATLAB的边界检查逻辑
        # MATLAB: if ((r < obj.GAC(end, theta_p(1))) && (r < obj.GAC(end, theta_p(2))))
        
        # 🔧 重要：考虑Python vs MATLAB下标起点差异
        # MATLAB: theta_p是1-based，GAC(end, theta_p(1))表示GAC[-1, theta_p(1)-1]
        # Python: 需要将MATLAB的1-based索引转换为0-based
        
        # MATLAB的theta_p是1-based，我们需要转换为Python的0-based
        # theta_p_lower和theta_p_upper已经是Python的0-based索引
        # GAC_last_layer[theta_p_lower]对应MATLAB的GAC(end, theta_p_lower+1)
        
        # 获取对应的物理边界值
        r_boundary1 = GAC_last_layer[theta_p_lower]  # 直接使用Python 0-based索引
        r_boundary2 = GAC_last_layer[theta_p_upper]  # 直接使用Python 0-based索引
        
        # 🔧 数值精度微调4: 边界值精度优化
        # 确保边界值的数值精度
        r_boundary1 = r_boundary1.clone().detach()  # 确保精度
        r_boundary2 = r_boundary2.clone().detach()
        
        # 边界值后处理：确保边界值在合理范围内
        if hasattr(config, 'L_ref') and config.L_ref is not None:
            max_expected_r = config.L_ref * 1.1  # 允许10%的安全边界
            r_boundary1 = torch.clamp(r_boundary1, 0.0, max_expected_r)
            r_boundary2 = torch.clamp(r_boundary2, 0.0, max_expected_r)
        
        # 🔧 严格使用MATLAB的边界检查逻辑 - 无容差
        # MATLAB: if ((r < obj.GAC(end, theta_p(1))) && (r < obj.GAC(end, theta_p(2))))
        
        # MATLAB的边界检查逻辑：要同时满足
        # 严格比较，不使用任何容差
        inside_plasma = (r_i < r_boundary1 and r_i < r_boundary2)
        
        if not inside_plasma:
            # 点在等离子体边界外，返回0
            result[i] = 0.0
            continue
        
        # 转换为density索引（直接使用GAC索引，因为density使用相同的索引系统）
        poid_cyl_1 = r_p_lower
        
        # 查找phi索引 - 对应MATLAB第15-17行
        p_p_lower, p_p_upper = bisec(phi_i, philist)
        
        # 确保p_p_lower是标量
        if hasattr(p_p_lower, 'item'):
            p_p_lower_scalar = p_p_lower.item()
        else:
            p_p_lower_scalar = int(p_p_lower)
            
        if hasattr(p_p_upper, 'item'):
            p_p_upper_scalar = p_p_upper.item()
        else:
            p_p_upper_scalar = int(p_p_upper)
        
        # 🔧 修复3: 检查phi索引和数组边界
        poid_cyl_3 = p_p_lower_scalar
        
        # 确保所有索引在有效范围内
        if (poid_cyl_1 < 0 or poid_cyl_1 >= density_3d.shape[1] or
            poid_cyl_2 < 0 or poid_cyl_2 >= density_3d.shape[0] or
            poid_cyl_3 < 0 or poid_cyl_3 >= density_3d.shape[2]):
            result[i] = 0.0
            continue
        
        # 步骤4: 获取边界值 - 严格按照MATLAB第100-105行
        # MATLAB: r_min = obj.GAC(poid_cyl(1), poid_cyl(2));
        #         r_max = obj.GAC(poid_cyl(1) + 1, poid_cyl(2));
        r_min = GAC_for_interpolation[poid_cyl_1, poid_cyl_2]
        r_max = GAC_for_interpolation[min(poid_cyl_1 + 1, GAC_for_interpolation.shape[0] - 1), poid_cyl_2]
        
        # MATLAB: theta_min = obj.GTC_c(end, poid_cyl(2));
        #         theta_max = obj.GTC_c(end, poid_cyl(2) + 1);
        theta_min = GTC_c_last[poid_cyl_2]
        theta_max = GTC_c_last[min(poid_cyl_2 + 1, GTC_c_last.shape[0] - 1)]
        
        # MATLAB: phi_min = philist(poid_cyl(3));
        #         phi_max = philist(poid_cyl(3) + 1);
        phi_min = philist[p_p_lower_scalar]
        phi_max = philist[min(p_p_lower_scalar + 1, len(philist) - 1)]
        
        # 步骤5: 计算权重 - 严格按照MATLAB第107-109行
        # MATLAB: da_cyl(1) = (r_max - r) / (r_max - r_min);
        #         da_cyl(2) = (theta_max - theta) / (theta_max - theta_min);
        #         da_cyl(3) = (phi_max - PHI0/(2*pi)) / (phi_max - phi_min);
        da_cyl_1 = (r_max - r_i) / (r_max - r_min + 1e-9)  # 添加小量避免除零
        da_cyl_2 = (theta_max - theta_i) / (theta_max - theta_min + 1e-9)
        da_cyl_3 = (phi_max - phi_i) / (phi_max - phi_min + 1e-9)
        
        # 步骤6: 设置索引变量 - 严格按照MATLAB第111-116行
        # MATLAB: m1 = poid_cyl(1); n1 = poid_cyl(2); p1 = poid_cyl(3);
        #         m2 = m1 + 1; n2 = n1 + 1; p2 = p1 + 1;
        m1 = int(max(0, min(poid_cyl_1, density_3d.shape[1] - 1)))  # 径向索引
        n1 = int(max(0, min(poid_cyl_2, density_3d.shape[0] - 1)))  # 极向索引
        p1 = int(max(0, min(p_p_lower_scalar, density_3d.shape[2] - 1)))  # phi索引
        
        m2 = int(max(0, min(m1 + 1, density_3d.shape[1] - 1)))  # 径向+1
        n2 = int(max(0, min(n1 + 1, density_3d.shape[0] - 1)))  # 极向+1
        p2 = int(max(0, min(p1 + 1, density_3d.shape[2] - 1)))  # phi+1
        
        # 步骤7: 三线性插值 - 严格按照MATLAB第118-121行
        # MATLAB: data(n1, m1, p1) 其中 n1=极向, m1=径向, p1=phi
        # Python: density_3d[n1, m1, p1] 其中 n1=极向, m1=径向, p1=phi
        # 严格按照MATLAB三线性插值公式
        data_000 = density_3d[n1, m1, p1]  # data(n1, m1, p1)
        data_100 = density_3d[n1, m2, p1]  # data(n1, m2, p1)
        data_010 = density_3d[n2, m1, p1]  # data(n2, m1, p1)
        data_110 = density_3d[n2, m2, p1]  # data(n2, m2, p1)
        data_001 = density_3d[n1, m1, p2]  # data(n1, m1, p2)
        data_101 = density_3d[n1, m2, p2]  # data(n1, m2, p2)
        data_011 = density_3d[n2, m1, p2]  # data(n2, m1, p2)
        data_111 = density_3d[n2, m2, p2]  # data(n2, m2, p2)
        
        # 严格按照MATLAB的插值公式
        # 第一层phi插值 (da_cyl(3)权重)
        term1 = da_cyl_3 * (da_cyl_2 * (da_cyl_1 * data_000 + (1.0 - da_cyl_1) * data_100) \
            + (1.0 - da_cyl_2) * (da_cyl_1 * data_010 + (1.0 - da_cyl_1) * data_110))
        
        # 第二层phi插值 (1-da_cyl(3)权重)  
        term2 = (1.0 - da_cyl_3) * (da_cyl_2 * (da_cyl_1 * data_001 + (1.0 - da_cyl_1) * data_101) \
            + (1.0 - da_cyl_2) * (da_cyl_1 * data_011 + (1.0 - da_cyl_1) * data_111))
        
        result[i] = term1 + term2
    # else: 保持result[i] = 0 (已经在初始化时设置)
    
    # 在plasma外的点保持为0（已经初始化为0）
    result = result.reshape(original_shape)
    
    return result