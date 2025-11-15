"""
插值模块 - 与MATLAB的probeEQ_local_s.m完全对应

本模块实现了与MATLAB probeEQ_local_s.m完全一致的Python版本，
包含坐标转换和三维三线性插值功能。

主要函数:
- probe_local_trilinear: 对应MATLAB的probeEQ_local_s.m
- probe_local_trilinear_vectorized: GPU优化的向量化版本
- bisec: 对应MATLAB的bisec.m，使用torch.searchsorted实现

MATLAB对应关系:
- probeEQ_local_s.m: GENE版本的精确插值
- bisec.m: 二分查找函数
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .utils import bisec


def _to_numpy(x):
    """安全地把 torch.Tensor / numpy / list 统一成 numpy.ndarray"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _bisec_scalar_numpy(xx: float, data: np.ndarray):
    """
    1D 二分查找，尽量模仿 MATLAB 的 bisec.m（返回 0-based 索引 (i_left, i_right)）

    MATLAB 行为（升序时）大致是：
      - 在 data 中找到 ya, yb，使得 data[ya] <= xx <= data[yb]
      - 并且 |ya - yb| == 1
    这里用相同思路实现。
    """
    data = np.asarray(data, dtype=np.float64)
    m = data.size
    if m < 2:
        return 0, 0

    # 判断单调方向
    ascending = data[0] < data[-1]
    if ascending:
        ya, yb = 0, m - 1
    else:
        ya, yb = m - 1, 0

    for _ in range(40):
        yt = (ya + yb) // 2
        ymid = data[yt]

        if (ascending and ymid <= xx) or ((not ascending) and ymid >= xx):
            ya = yt
        else:
            yb = yt

        if abs(ya - yb) <= 1:
            i1, i2 = sorted((ya, yb))
            # 保护一下边界：确保有 i1 < i2 且在 [0, m-1]
            if i1 == i2:
                if i1 == 0:
                    i2 = 1
                elif i1 == m - 1:
                    i1 = m - 2
                else:
                    i2 = min(i1 + 1, m - 1)
            return i1, i2

    # fallback：就近夹一个 cell
    i1 = max(0, min(ya, m - 2))
    i2 = i1 + 1
    return i1, i2


def batch_bisec_search(values: torch.Tensor, reference_array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量二分查找 - 向量化的bisec实现
    
    Args:
        values: 要查找的值数组 (N,)
        reference_array: 参考数组 (M,)
    
    Returns:
        (indices1, indices2): 查找结果，与MATLAB bisec一致
    """
    # 确保reference_array是1D tensor
    ref_array = reference_array.flatten()
    
    # 检查排序方向
    is_ascending = ref_array[0] < ref_array[-1]
    
    # 使用torch.searchsorted进行批量查找
    if is_ascending:
        # 升序
        indices = torch.searchsorted(ref_array, values, side='right')
    else:
        # 降序，需要特殊处理
        # 反转数组进行搜索，然后调整结果
        ref_reversed = ref_array.flip(0)
        indices_reversed = torch.searchsorted(ref_reversed, values, side='right')
        indices = len(ref_array) - indices_reversed
    
    # 确保索引在有效范围内
    indices = torch.clamp(indices, 1, len(ref_array))  # MATLAB 1-based索引
    
    # bisec返回两个索引
    indices1 = torch.clamp(indices - 1, 1, len(ref_array))
    indices2 = indices
    
    return indices1, indices2


def batch_trilinear_interpolate(
    density_3d: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    theta_indices: Tuple[torch.Tensor, torch.Tensor],
    phi_indices: Tuple[torch.Tensor, torch.Tensor],
    GAC_physical: torch.Tensor,
    GTC_c_last: torch.Tensor,
    philist: torch.Tensor
) -> torch.Tensor:
    """
    批量三线性插值 - GPU优化的向量化实现
    
    Args:
        density_3d: 密度场 (ntheta, nx, nz)
        r: 径向坐标 (N,)
        theta: 极向角度 (N,)
        phi: 环向角度 (N,) 归一化到[0,1]
        theta_indices: (theta_lower, theta_upper) 索引
        phi_indices: (phi_lower, phi_upper) 索引
        GAC_physical: 物理坐标的GAC数据
        GTC_c_last: 最外层的theta坐标
        philist: phi网格
    
    Returns:
        插值结果 (N,)
    """
    N = r.shape[0]
    device = density_3d.device
    dtype = density_3d.dtype
    
    theta_lower, theta_upper = theta_indices
    phi_lower, phi_upper = phi_indices
    
    # 确保索引是整数类型
    theta_lower = theta_lower.long()
    theta_upper = theta_upper.long()
    phi_lower = phi_lower.long()
    phi_upper = phi_upper.long()
    
    # 边界检查：确保所有索引在有效范围内
    ntheta, nx, nz = density_3d.shape
    max_theta_idx = len(GTC_c_last)
    
    # 限制theta索引到有效范围 (MATLAB 1-based: 1到401)
    theta_lower = torch.clamp(theta_lower, 1, max_theta_idx)
    theta_upper = torch.clamp(theta_upper, 1, max_theta_idx)
    
    # 转换为Python 0-based索引
    theta_lower_0based = theta_lower - 1
    theta_upper_0based = theta_upper - 1
    
    # 限制phi索引
    phi_lower = torch.clamp(phi_lower, 0, len(philist) - 2)
    phi_upper = torch.clamp(phi_lower + 1, 0, len(philist) - 1)
    
    # 计算每个点对应的theta角度
    theta_at_lower = GTC_c_last[theta_lower_0based]
    theta_at_upper = GTC_c_last[theta_upper_0based]
    
    # 确保theta_max > theta_min (处理边界情况)
    theta_min = torch.minimum(theta_at_lower, theta_at_upper)
    theta_max = torch.maximum(theta_at_lower, theta_at_upper)
    
    # 防止theta_min == theta_max
    theta_diff = theta_max - theta_min
    theta_eps = 1e-6
    theta_min = torch.where(torch.abs(theta_diff) < theta_eps, 
                           theta_min - theta_eps, theta_min)
    theta_max = torch.where(torch.abs(theta_diff) < theta_eps, 
                           theta_max + theta_eps, theta_max)
    
    # 计算phi边界值
    phi_min = philist[phi_lower]
    phi_max = philist[phi_upper]
    
    # 防止phi_min == phi_max
    phi_diff = phi_max - phi_min
    phi_min = torch.where(torch.abs(phi_diff) < theta_eps, 
                         phi_min - theta_eps, phi_min)
    phi_max = torch.where(torch.abs(phi_diff) < theta_eps, 
                         phi_max + theta_eps, phi_max)
    
    # 对每个theta角度，找到对应的r边界值
    # 使用GAC数据的最后一层（最外层）
    r_boundary_lower = GAC_physical[-1, theta_lower_0based]  # shape: (N,)
    r_boundary_upper = GAC_physical[-1, theta_upper_0based]  # shape: (N,)
    
    # 边界检查：确保点在等离子体内部
    tolerance = 1e-2
    r_boundary_min = torch.minimum(r_boundary_lower, r_boundary_upper)
    r_boundary_max = torch.maximum(r_boundary_lower, r_boundary_upper)
    
    # 点在等离子体内部的条件
    inside_plasma = (r >= 0.0) & (r <= r_boundary_max + tolerance)
    
    # 对不在等离子体内部的点，返回0
    result = torch.zeros(N, device=device, dtype=dtype)
    
    if not inside_plasma.any():
        return result
    
    # 只对等离子体内部的点进行插值
    valid_indices = inside_plasma.nonzero(as_tuple=False).squeeze(-1)
    
    if len(valid_indices) == 0:
        return result
    
    # 为有效点准备数据
    r_valid = r[valid_indices]
    theta_valid = theta[valid_indices]
    phi_valid = phi[valid_indices]
    theta_lower_valid = theta_lower_0based[valid_indices]
    theta_upper_valid = theta_upper_0based[valid_indices]
    phi_lower_valid = phi_lower[valid_indices]
    phi_upper_valid = phi_upper[valid_indices]
    
    # 计算r边界值（对于有效点）
    r_boundary_min_valid = r_boundary_min[valid_indices]
    r_boundary_max_valid = r_boundary_max[valid_indices]
    
    # 计算径向索引（线性查找）
    r_indices = torch.zeros(len(valid_indices), device=device, dtype=torch.long)
    
    for i, idx in enumerate(valid_indices):
        r_i = r_valid[i]
        theta_idx = theta_lower_valid[i]  # 使用theta_lower对应的GAC列
        
        # 在该theta角度的GAC列中查找最接近的r值
        GAC_at_theta = GAC_physical[:, theta_idx]
        r_diffs = torch.abs(GAC_at_theta - r_i)
        r_p_lower = torch.argmin(r_diffs)
        r_p_upper = min(r_p_lower + 1, len(GAC_at_theta) - 1)
        
        r_indices[i] = r_p_lower
    
    # 确保r索引在有效范围内
    r_indices = torch.clamp(r_indices, 0, nx - 2)
    
    # 计算权重
    r_min = torch.gather(GAC_physical[:, theta_lower_valid], 0, r_indices.unsqueeze(0)).squeeze(0)
    r_max = torch.gather(GAC_physical[:, theta_upper_valid], 0, (r_indices + 1).unsqueeze(0)).squeeze(0)
    
    # 确保r_max > r_min
    r_min_final = torch.minimum(r_min, r_max)
    r_max_final = torch.maximum(r_min, r_max)
    
    # 防止除零
    r_diff = r_max_final - r_min_final
    r_diff = torch.where(torch.abs(r_diff) < 1e-6, 
                        torch.sign(r_diff) * 1e-6, r_diff)
    
    theta_min_valid = theta_min[valid_indices]
    theta_max_valid = theta_max[valid_indices]
    phi_min_valid = phi_min[valid_indices]
    phi_max_valid = phi_max[valid_indices]
    
    # 计算权重
    da_cyl_1 = (r_max_final - r_valid) / r_diff
    da_cyl_2 = (theta_max_valid - theta_valid) / (theta_max_valid - theta_min_valid)
    da_cyl_3 = (phi_max_valid - phi_valid) / (phi_max_valid - phi_min_valid)
    
    # 确保权重在合理范围内
    da_cyl_1 = torch.clamp(da_cyl_1, 0.0, 1.0)
    da_cyl_2 = torch.clamp(da_cyl_2, 0.0, 1.0)
    da_cyl_3 = torch.clamp(da_cyl_3, 0.0, 1.0)
    
    # 设置最终索引
    m1 = r_indices                                    # 径向索引
    n1 = theta_lower_valid                           # 极向索引
    p1 = phi_lower_valid                            # phi索引
    
    m2 = torch.clamp(m1 + 1, 0, nx - 1)             # 径向+1
    n2 = torch.clamp(n1 + 1, 0, ntheta - 1)         # 极向+1  
    p2 = torch.clamp(p1 + 1, 0, nz - 1)             # phi+1
    
    # 批量提取8个角点的数据
    data_000 = density_3d[n1, m1, p1]  # (theta, radial, phi)
    data_100 = density_3d[n1, m2, p1]
    data_010 = density_3d[n2, m1, p1]
    data_110 = density_3d[n2, m2, p1]
    data_001 = density_3d[n1, m1, p2]
    data_101 = density_3d[n1, m2, p2]
    data_011 = density_3d[n2, m1, p2]
    data_111 = density_3d[n2, m2, p2]
    
    # 批量三线性插值计算
    term1 = da_cyl_3 * (da_cyl_2 * (da_cyl_1 * data_000 + (1.0 - da_cyl_1) * data_100) \
        + (1.0 - da_cyl_2) * (da_cyl_1 * data_010 + (1.0 - da_cyl_1) * data_110))
    
    term2 = (1.0 - da_cyl_3) * (da_cyl_2 * (da_cyl_1 * data_001 + (1.0 - da_cyl_1) * data_101) \
        + (1.0 - da_cyl_2) * (da_cyl_1 * data_011 + (1.0 - da_cyl_1) * data_111))
    
    # 计算最终结果
    valid_result = term1 + term2
    
    # 放入结果数组的对应位置
    result[valid_indices] = valid_result
    
    return result


def probe_local_trilinear_vectorized(
    density_3d: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    PHI: torch.Tensor,
    config
) -> torch.Tensor:
    """
    3D三线性插值 - GPU优化的向量化版本
    
    这个函数实现了与probe_local_trilinear完全相同的功能，
    但使用向量化计算消除Python循环，显著提升性能。
    
    Args:
        density_3d: 密度场 (ntheta, nx, nz) 或 (1, ntheta, nx, nz)
        R: R坐标 (scalar or tensor)
        Z: Z坐标 (scalar or tensor)  
        PHI: PHI坐标 [0, 2π] (scalar or tensor)
        config: 包含equilibrium数据的配置对象
    
    Returns:
        插值结果 (与输入shape相同)
    """
    # 🔧 处理维度兼容性 - 与probe_local_trilinear保持一致
    original_density_shape = density_3d.shape
    if density_3d.ndim == 3:
        # 保持3D输入不变
        pass
    elif density_3d.ndim == 4:
        # 如果是4D，移除batch维度，保持为3D
        density_3d = density_3d.squeeze(0)  # 移除batch维度
    else:
        raise ValueError(f"density_3d必须是3D或4D张量，但得到的是{density_3d.ndim}D: {density_3d.shape}")
    
    # 确保输入是tensor
    if not isinstance(R, torch.Tensor):
        R = torch.as_tensor(R, device=density_3d.device, dtype=torch.float64).clone()
    if not isinstance(Z, torch.Tensor):
        Z = torch.as_tensor(Z, device=density_3d.device, dtype=torch.float64).clone()
    if not isinstance(PHI, torch.Tensor):
        PHI = torch.as_tensor(PHI, device=density_3d.device, dtype=torch.float64).clone()
    
    # 展平为1D
    original_shape = R.shape
    R_flat = R.flatten()
    Z_flat = Z.flatten()
    PHI_flat = PHI.flatten()
    N = R_flat.shape[0]
    
    # 检查是否有equilibrium数据
    if config.PA is None or config.GAC is None:
        return torch.zeros(original_shape, device=density_3d.device, dtype=density_3d.dtype)
    
    # 坐标转换：计算(r, theta)
    PA_tensor = config.PA.to(device=R.device, dtype=R.dtype).clone()
    dR = R_flat - PA_tensor[0]
    dZ = Z_flat - PA_tensor[1]
    
    # 计算径向距离和角度
    r = torch.sqrt(dR**2 + dZ**2)
    raw_theta = torch.atan2(dZ, dR)
    
    # 使用MATLAB的mod函数行为
    two_pi = 2 * torch.pi
    theta = raw_theta - two_pi * torch.floor(raw_theta / two_pi)
    theta = torch.where(theta < 0, theta + two_pi, theta)
    theta = torch.where(theta >= two_pi, theta - two_pi, theta)
    
    # 处理GAC坐标缩放
    if hasattr(config, 'L_ref') and config.L_ref is not None:
        GAC_physical = config.GAC * config.L_ref
    else:
        GAC_physical = config.GAC
    
    # 设置phi列表（归一化到[0,1]）
    nz = density_3d.shape[2]
    KZMt = nz - 2
    philist = torch.linspace(0, 1, KZMt + 2, device=density_3d.device)
    
    # 批量查找theta索引
    GTC_c_last = config.GTC_c[-1, :]
    theta_lower, theta_upper = batch_bisec_search(theta, GTC_c_last)
    
    # 批量查找phi索引
    phi_normalized = PHI_flat / (2 * torch.pi)
    phi_lower, phi_upper = batch_bisec_search(phi_normalized, philist)
    
    # 执行批量三线性插值
    result = batch_trilinear_interpolate(
        density_3d, r, theta, phi_normalized,
        (theta_lower, theta_upper),
        (phi_lower, phi_upper),
        GAC_physical, GTC_c_last, philist
    )
    
    return result.reshape(original_shape)


def probe_local_trilinear(
    density_3d: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    PHI: torch.Tensor,
    config
) -> torch.Tensor:
    """
    严格对应 MATLAB: sim_data/GENE/@GENEClass/probeEQ_local_s.m

    MATLAB 调用关系:
        z = probeEQ_local_s(obj, R0, Z0, PHI0, data3)

    这里先实现一个纯 CPU / numpy 的标量版本，
    保证数值和 MATLAB 对齐，再考虑后续向量化优化。
    """
    # === 1. 统一 density_3d 形状为 (ntheta, nx, nz) ===
    density_orig = density_3d
    if density_3d.ndim == 4:
        # (B, ntheta, nx, nz) -> 假定 B=1
        density_3d = density_3d.squeeze(0)
    elif density_3d.ndim != 3:
        raise ValueError(f"density_3d 维度必须是 3 或 4，目前是 {density_3d.ndim}: {density_3d.shape}")

    ntheta, nx, nz = density_3d.shape
    device = density_orig.device
    dtype = density_orig.dtype

    # === 2. 把 equilibrium 与密度场都搬到 numpy 上，方便精确索引 ===
    density_np   = _to_numpy(density_3d)           # (ntheta, nx, nz) = (theta, radial, phi)
    GAC_np       = _to_numpy(config.GAC)           # (nx, ntheta)
    GTC_last_np  = _to_numpy(config.GTC_c[-1, :])  # (ntheta+?); 实际长度应与 GAC 第二维一致
    PA_np        = _to_numpy(config.PA)            # (2,) [R_axis, Z_axis]

    # φ 方向网格，MATLAB: philist = linspace(0, 1, obj.KZMt+1+1);
    if hasattr(config, "KZMt") and config.KZMt is not None:
        KZMt = int(config.KZMt)
        philist_np = np.linspace(0.0, 1.0, KZMt + 2, dtype=np.float64)
    else:
        # 直接根据 data3 第三维推断：nz = KZMt+2
        philist_np = np.linspace(0.0, 1.0, nz, dtype=np.float64)
        KZMt = nz - 2

    # 容错：确保长度一致（理论上 philist_np.size == nz）
    if philist_np.size != nz:
        philist_np = np.linspace(0.0, 1.0, nz, dtype=np.float64)
        KZMt = nz - 2

    # 一致性检查
    if GAC_np.shape[0] != nx:
        raise ValueError(f"GAC 第一维({GAC_np.shape[0]}) 应该等于 nx={nx}")
    if GAC_np.shape[1] != ntheta:
        raise ValueError(f"GAC 第二维({GAC_np.shape[1]}) 应该等于 ntheta={ntheta}")
    if GTC_last_np.size != ntheta:
        raise ValueError(f"GTC_c(end,:) 长度({GTC_last_np.size}) 应该等于 ntheta={ntheta}")

    # === 3. 展平成 1D 坐标数组，在 numpy 上循环 ===
    R_arr   = _to_numpy(R).ravel()
    Z_arr   = _to_numpy(Z).ravel()
    PHI_arr = _to_numpy(PHI).ravel()
    N = R_arr.size

    result_np = np.zeros(N, dtype=density_np.dtype)

    two_pi = 2.0 * np.pi

    for i in range(N):
        R0   = float(R_arr[i])
        Z0   = float(Z_arr[i])
        PHI0 = float(PHI_arr[i])

        # --- 3.1 坐标变换 (对应 MATLAB 第6-7行) ---
        dR = R0 - PA_np[0]
        dZ = Z0 - PA_np[1]
        r = np.sqrt(dR * dR + dZ * dZ)
        theta = np.arctan2(dZ, dR)
        # MATLAB: mod(atan2(...), 2*pi)
        theta = theta % two_pi

        # --- 3.2 θ 方向二分查找 (对应 MATLAB: theta_p = bisec(theta, obj.GTC_c(end,:));) ---
        th_lo, th_hi = _bisec_scalar_numpy(theta, GTC_last_np)  # 0-based
        # th_lo ∈ [0, ntheta-2], th_hi = th_lo+1 ∈ [1, ntheta-1]（正常情况）

        # --- 3.3 等离子体边界检查 (对应 MATLAB if ((r < GAC(end,theta_p(1))) && (r < GAC(end,theta_p(2)))) ) ---
        r_b1 = float(GAC_np[-1, th_lo])
        r_b2 = float(GAC_np[-1, th_hi])

        if not ((r < r_b1) and (r < r_b2)):
            # 在等离子体外 -> 直接返回 0
            result_np[i] = 0.0
            continue

        # --- 3.4 r 方向二分查找 (对应 MATLAB: r_p = bisec(r, GAC(:, poid_cyl(2)).') ) ---
        r_col = GAC_np[:, th_lo]  # 固定在 θ = th_lo 这一列
        r_lo, r_hi = _bisec_scalar_numpy(r, r_col)
        # r_lo ∈ [0, nx-2], r_hi = r_lo+1 ∈ [1, nx-1]

        # --- 3.5 φ 方向二分查找 (对应 MATLAB: p_p = bisec(PHI0/(2*pi), philist); ) ---
        phi_norm = (PHI0 / two_pi) % 1.0  # 归一化到 [0,1)
        p_lo, p_hi = _bisec_scalar_numpy(phi_norm, philist_np)
        # p_lo ∈ [0, nz-2], p_hi = p_lo+1 ∈ [1, nz-1]

        # --- 3.6 计算 cell 边界值 (对应 MATLAB 第23-28行) ---
        r_min = float(GAC_np[r_lo, th_lo])
        r_max = float(GAC_np[r_hi, th_lo])
        theta_min = float(GTC_last_np[th_lo])
        theta_max = float(GTC_last_np[th_hi])
        phi_min = float(philist_np[p_lo])
        phi_max = float(philist_np[p_hi])

        # 避免分母为 0 的极端情况
        if r_max == r_min:
            # r 恰好落在网格点上，就直接用该点值
            n1 = min(max(th_lo, 0), ntheta - 1)
            m1 = min(max(r_lo,  0), nx - 1)
            p1 = min(max(p_lo,  0), nz - 1)
            result_np[i] = density_np[n1, m1, p1]
            continue
        if theta_max == theta_min:
            theta_max += 1e-12
        if phi_max == phi_min:
            phi_max += 1e-12

        # --- 3.7 权重 da_cyl (对应 MATLAB 第30-32行) ---
        da1 = (r_max   - r)        / (r_max   - r_min)
        da2 = (theta_max - theta)  / (theta_max - theta_min)
        da3 = (phi_max - phi_norm) / (phi_max - phi_min)

        # （可选）夹一下 0~1，防止数值抖动
        # da1 = np.clip(da1, 0.0, 1.0)
        # da2 = np.clip(da2, 0.0, 1.0)
        # da3 = np.clip(da3, 0.0, 1.0)

        # --- 3.8 设置 8 个角点索引 (对应 MATLAB 第34-40行) ---
        n1 = min(max(th_lo, 0), ntheta - 1)
        n2 = min(n1 + 1, ntheta - 1)
        m1 = min(max(r_lo,  0), nx - 1)
        m2 = min(m1 + 1, nx - 1)
        p1 = min(max(p_lo,  0), nz - 1)
        p2 = min(p1 + 1, nz - 1)

        d000 = density_np[n1, m1, p1]
        d100 = density_np[n1, m2, p1]
        d010 = density_np[n2, m1, p1]
        d110 = density_np[n2, m2, p1]
        d001 = density_np[n1, m1, p2]
        d101 = density_np[n1, m2, p2]
        d011 = density_np[n2, m1, p2]
        d111 = density_np[n2, m2, p2]

        # --- 3.9 三线性插值 (对应 MATLAB 最后的 z = ...) ---
        term1 = da3 * (
            da2 * (da1 * d000 + (1.0 - da1) * d100) +
            (1.0 - da2) * (da1 * d010 + (1.0 - da1) * d110)
        )
        term2 = (1.0 - da3) * (
            da2 * (da1 * d001 + (1.0 - da1) * d101) +
            (1.0 - da2) * (da1 * d011 + (1.0 - da1) * d111)
        )

        result_np[i] = term1 + term2

    # === 4. 把结果 reshape 回原始形状，并转回 torch.Tensor ===
    result_np = result_np.reshape(_to_numpy(R).shape)
    result = torch.as_tensor(result_np, device=device, dtype=dtype)
    return result