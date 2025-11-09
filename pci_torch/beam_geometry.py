"""
光束几何计算

生成PCI光束的采样网格和光路计算
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .config import BeamConfig, GENEConfig


def compute_beam_grid(
    beam_config: BeamConfig,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    计算光束网格的所有采样点（笛卡尔坐标）
    
    严格按照MATLAB LSview_com.m 行62-133实现
    
    Args:
        beam_config: 光束配置
        device: PyTorch设备
    
    Returns:
        字典包含:
            - 'grid_xyz': (div1*2+1, div2*2+1, divls+1, 3) 网格点笛卡尔坐标
            - 'grid_flat': (N, 3) 展平的坐标，N = (2*div1+1)*(2*div2+1)*(divls+1)
            - 'beam_vector': (3,) 光束方向单位向量
            - 'perpendicular_vectors': (2, 3) 两个垂直向量
    """
    # MATLAB 第62-70行: 坐标转换
    # B1(1,:) = pp1(1:3) - 起点 (R[mm], Z[mm], phi[0-1])
    # B1(2,:) = pp1(4:6) - 终点
    # 从 BeamConfig 获取原始值 (已经是米单位)
    B1_start = np.array([
        beam_config.injection_point[0],  # R [m]
        beam_config.injection_point[1],  # Z [m]
        beam_config.injection_point[2]   # phi [0-1]
    ])
    B1_end = np.array([
        beam_config.detection_point[0],   # R [m]
        beam_config.detection_point[1],   # Z [m]
        beam_config.detection_point[2]    # phi [0-1]
    ])
    
    # B2(:,1) = B1(:,1).*cos(2*pi*B1(:,3)) - X坐标
    # B2(:,2) = B1(:,1).*sin(2*pi*B1(:,3)) - Y坐标
    # B2(:,3) = B1(:,2) - Z坐标
    # B2 = B2/1000.0 - 单位转换（毫米到米）
    B2_start = np.array([
        B1_start[0] * np.cos(2 * np.pi * B1_start[2]),
        B1_start[0] * np.sin(2 * np.pi * B1_start[2]),
        B1_start[1]
    ]) / 1000.0
    
    B2_end = np.array([
        B1_end[0] * np.cos(2 * np.pi * B1_end[2]),
        B1_end[0] * np.sin(2 * np.pi * B1_end[2]),
        B1_end[1]
    ]) / 1000.0
    
    # 转换为torch tensor
    B2_start = torch.tensor(B2_start, dtype=torch.float64, device=device)
    B2_end = torch.tensor(B2_end, dtype=torch.float64, device=device)
    
    # MATLAB 第71-74行: 计算光束长度
    # b2ls = sqrt((B2(1,1)-B2(2,1))^2 + (B2(1,2)-B2(2,2))^2 + (B2(1,3)-B2(2,3))^2)
    b2ls = torch.sqrt(
        (B2_start[0] - B2_end[0])**2 +
        (B2_start[1] - B2_end[1])**2 +
        (B2_start[2] - B2_end[2])**2
    )
    
    # MATLAB 第76-78行: 计算光束方向向量
    # p1 = B2(1,:) - B2(2,:)  从终点指向起点
    p1 = torch.zeros(3, dtype=torch.float64, device=device)
    p1[0] = B2_start[0] - B2_end[0]
    p1[1] = B2_start[1] - B2_end[1]
    p1[2] = B2_start[2] - B2_end[2]
    
    # 计算单位向量
    p1_unit = p1 / torch.norm(p1)
    
    # MATLAB 第80-102行: 计算垂直向量
    xl = torch.zeros(2, 3, dtype=torch.float64, device=device)
    wid1 = beam_config.width_vertical
    wid2 = beam_config.width_toroidal
    
    # 使用原始 phi 值 B1(1,3)，范围 [0-1]
    phi_raw = B1_start[2]  # 原始 phi 值，范围 [0-1]
    
    # 检查光束是否垂直（p1(1)==0 && p1(2)==0）
    if torch.abs(p1[0]) < 1e-10 and torch.abs(p1[1]) < 1e-10:
        # MATLAB 第81-86行: 垂直光束的情况
        phi_rad = 2 * np.pi * phi_raw
        xl[0, 0] = wid1 / 2.0 * np.cos(phi_rad)
        xl[0, 1] = wid1 / 2.0 * np.sin(phi_rad)
        xl[0, 2] = 0.0
        xl[1, 0] = -wid2 / 2.0 * np.sin(phi_rad)
        xl[1, 1] = wid2 / 2.0 * np.cos(phi_rad)
        xl[1, 2] = 0.0
    else:
        # MATLAB 第87-101行: 一般情况
        phi_rad = 2 * np.pi * phi_raw
        tan_phi = np.tan(phi_rad)
        tan_phi_t = torch.tensor(tan_phi, dtype=torch.float64, device=device)
        
        # 第一个垂直向量（MATLAB第88-94行）
        xl[0, 0] = p1[2]
        xl[0, 1] = p1[2] * tan_phi_t
        xl[0, 2] = -(p1[0] + p1[1] * tan_phi_t)
        xl0 = 1.0 / torch.norm(xl[0]) * (wid1 / 2.0)
        xl[0, 0] = xl[0, 0] * xl0
        xl[0, 1] = xl[0, 1] * xl0
        xl[0, 2] = xl[0, 2] * xl0
        
        # 第二个垂直向量（MATLAB第95-101行）
        xl[1, 0] = p1[0] * p1[1] + (p1[1]**2 + p1[2]**2) * tan_phi_t
        xl[1, 1] = -p1[0]**2 - p1[2]**2 - p1[0] * p1[1] * tan_phi_t
        xl[1, 2] = p1[1] * p1[2] - p1[0] * p1[2] * tan_phi_t
        xl0 = 1.0 / torch.norm(xl[1]) * (wid2 / 2.0)
        xl[1, 0] = xl[1, 0] * xl0
        xl[1, 1] = xl[1, 1] * xl0
        xl[1, 2] = xl[1, 2] * xl0
    
    # 计算单位向量（用于返回）
    xl_unit = torch.zeros_like(xl)
    xl_unit[0] = xl[0] / torch.norm(xl[0])
    xl_unit[1] = xl[1] / torch.norm(xl[1])
    
    # MATLAB 第103-107行: 网格参数
    div1 = beam_config.div_vertical
    div2 = beam_config.div_toroidal
    divls = beam_config.div_beam
    divls_2 = divls + 1
    div1_2 = 2 * div1 + 1
    div2_2 = 2 * div2 + 1
    
    # MATLAB 第107行: b2ls = b2ls/divls (这是步长，不是总长度)
    # 注意：MATLAB 中 b2ls 被重新赋值为步长
    b2ls_step = b2ls / divls
    
    # MATLAB 第108-111行: 初始化网格（从检测器端点开始）
    xls = torch.ones(div1_2, div2_2, divls_2, device=device) * B2_end[0]
    yls = torch.ones(div1_2, div2_2, divls_2, device=device) * B2_end[1]
    zls = torch.ones(div1_2, div2_2, divls_2, device=device) * B2_end[2]
    
    # MATLAB 第113-118行: 添加垂直方向1的偏移
    # MATLAB: for j=1:div1_2, replix(j,:,:)=ones(div2_2,divls_2)*(real(j-1)-div1)/div1
    # Python: j 从 0 开始，所以 (j - div1) / div1 等价于 MATLAB 的 (real(j-1)-div1)/div1
    for j in range(div1_2):
        offset = (j - div1) / div1  # 对应 MATLAB 的 (real(j-1)-div1)/div1
        xls[j, :, :] = xls[j, :, :] + offset * xl[0, 0]
        yls[j, :, :] = yls[j, :, :] + offset * xl[0, 1]
        zls[j, :, :] = zls[j, :, :] + offset * xl[0, 2]
    
    # MATLAB 第119-124行: 添加垂直方向2的偏移
    # MATLAB: for j=1:div2_2, replix(:,j,:)=ones(div1_2,divls_2)*(real(j-1)-div2)/div2
    for j in range(div2_2):
        offset = (j - div2) / div2  # 对应 MATLAB 的 (real(j-1)-div2)/div2
        xls[:, j, :] = xls[:, j, :] + offset * xl[1, 0]
        yls[:, j, :] = yls[:, j, :] + offset * xl[1, 1]
        zls[:, j, :] = zls[:, j, :] + offset * xl[1, 2]
    
    # MATLAB 第125-130行: 添加光束方向的偏移
    # MATLAB: for j=1:divls_2, replix(:,:,j)=ones(div1_2,div2_2)*real(j-1)/divls
    # MATLAB 中 j 从 1 开始，所以 real(j-1)/divls 当 j=1 时为 0，当 j=divls_2 时为 divls/divls=1
    # Python 中 j 从 0 开始，所以 j/divls 当 j=0 时为 0，当 j=divls_2-1 时为 (divls_2-1)/divls = divls/divls=1
    # 注意：divls_2 = divls + 1，所以 j 的范围是 [0, divls]，最后一个 j=divls 时 offset=divls/divls=1
    for j in range(divls_2):
        offset = j / divls  # 对应 MATLAB 的 real(j-1)/divls，其中 MATLAB 的 j 从 1 开始
        xls[:, :, j] = xls[:, :, j] + offset * p1[0]
        yls[:, :, j] = yls[:, :, j] + offset * p1[1]
        zls[:, :, j] = zls[:, :, j] + offset * p1[2]
    
    # MATLAB 第131-133行: 展平
    # MATLAB: xls1=reshape(xls,div1_2*div2_2*divls_2,1)
    xls1 = xls.reshape(div1_2 * div2_2 * divls_2)
    yls1 = yls.reshape(div1_2 * div2_2 * divls_2)
    zls1 = zls.reshape(div1_2 * div2_2 * divls_2)
    
    # 堆叠成网格
    grid_xyz = torch.stack([xls, yls, zls], dim=-1)  # (div1_2, div2_2, divls_2, 3)
    
    # 展平为 (N, 3)
    grid_flat = torch.stack([xls1, yls1, zls1], dim=-1)  # (N, 3)
    
    return {
        'grid_xyz': grid_xyz,
        'grid_flat': grid_flat,
        'beam_vector': p1_unit,
        'perpendicular_vectors': xl_unit,
        'beam_start': B2_start,
        'beam_end': B2_end,
        'beam_length': b2ls,  # 总长度
        'beam_step': b2ls_step,  # 步长
    }


def compute_beam_path_center(
    beam_config: BeamConfig,
    device: str = 'cuda',
    beam_grid: Dict[str, torch.Tensor] = None
) -> torch.Tensor:
    """
    计算光束中心路径的采样点
    
    严格按照MATLAB LSview_com.m 行155-158实现
    从网格中提取中心路径: xls(div1+1, div2+1, :)
    
    Args:
        beam_config: 光束配置
        device: PyTorch设备
        beam_grid: compute_beam_grid的输出（可选，如果不提供则计算）
    
    Returns:
        (divls+1, 3) 中心路径坐标
    """
    # 如果没有提供beam_grid，则计算它
    if beam_grid is None:
        beam_grid = compute_beam_grid(beam_config, device=device)
    
    # MATLAB 第155-158行: 从网格中提取中心路径
    # xls_c(:,1)=squeeze(xls(div1+1,div2+1,:))
    # xls_c(:,2)=squeeze(yls(div1+1,div2+1,:))
    # xls_c(:,3)=squeeze(zls(div1+1,div2+1,:))
    div1 = beam_config.div_vertical
    div2 = beam_config.div_toroidal
    
    grid_xyz = beam_grid['grid_xyz']  # (div1_2, div2_2, divls_2, 3)
    
    # MATLAB 索引从1开始，所以 div1+1 对应 Python 的 div1
    # 因为 div1_2 = 2*div1+1，所以中心索引是 div1
    center_path = grid_xyz[div1, div2, :, :]  # (divls_2, 3)
    
    return center_path


def get_detector_positions(
    beam_config: BeamConfig,
    device: str = 'cuda',
    beam_grid: Dict[str, torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取检测器阵列的位置
    
    严格按照MATLAB LSview_com.m 行212-213实现
    [xx1,yy1]=meshgrid(wid1/2*[-div1:div1]/div1,-wid2/2*[-div2:div2]/div2);
    xx1 = fliplr(xx1);
    
    根据MATLAB代码逻辑，检测器的3D位置应该从光束网格中提取：
    - 光束网格从检测器端点（B2_end）开始初始化
    - 添加垂直方向的偏移后，在光束方向偏移为0的位置就是检测器位置
    - 即：detector_coords = grid_xyz[:, :, 0]（光束方向的第一个索引）
    
    Args:
        beam_config: 光束配置
        device: PyTorch设备
        beam_grid: compute_beam_grid的输出（可选，如果不提供则计算）
    
    Returns:
        (detector_coords, detector_grid):
            - detector_coords: (div1*2+1, div2*2+1, 3) 检测器3D坐标
            - detector_grid: (div1*2+1, div2*2+1, 2) 检测器网格 (yy1, xx1_flipped)
    """
    div1 = beam_config.div_vertical
    div2 = beam_config.div_toroidal
    wid1 = beam_config.width_vertical
    wid2 = beam_config.width_toroidal
    
    # MATLAB 第212行: meshgrid(wid1/2*[-div1:div1]/div1, -wid2/2*[-div2:div2]/div2)
    # wid1/2*[-div1:div1]/div1 生成从 -wid1/2 到 wid1/2 的数组，共 2*div1+1 个点
    # -wid2/2*[-div2:div2]/div2 生成从 wid2/2 到 -wid2/2 的数组（注意负号），共 2*div2+1 个点
    
    # 生成 x 坐标（对应 MATLAB 的第一个参数）
    x_coords = torch.tensor([wid1/2.0 * (i - div1) / div1 for i in range(2*div1+1)], 
                            dtype=torch.float64, device=device)
    # 生成 y 坐标（对应 MATLAB 的第二个参数，注意负号）
    y_coords = torch.tensor([-wid2/2.0 * (i - div2) / div2 for i in range(2*div2+1)], 
                           dtype=torch.float64, device=device)
    
    # MATLAB 的 meshgrid: [xx1, yy1] = meshgrid(x, y)
    # 其中 x 是列向量，y 是行向量
    # xx1 的每一行都是 x，yy1 的每一列都是 y
    # 在 Python 中，使用 indexing='xy' 来匹配 MATLAB 的行为
    xx1, yy1 = torch.meshgrid(x_coords, y_coords, indexing='xy')
    
    # MATLAB 第213行: xx1 = fliplr(xx1) - 左右翻转
    xx1_flipped = torch.flip(xx1, dims=[1])
    
    # 堆叠成网格 (div1_2, div2_2, 2)
    detector_grid = torch.stack([yy1, xx1_flipped], dim=-1)
    
    # 检测器的3D位置：从光束网格中提取
    # 根据MATLAB代码逻辑（第108-130行）：
    # - 网格从检测器端点（B2_end）开始初始化
    # - 添加垂直方向的偏移
    # - 光束方向的偏移从0开始（j=1，对应Python的j=0）
    # 所以检测器位置 = grid_xyz[:, :, 0]（光束方向的第一个索引）
    if beam_grid is None:
        beam_grid = compute_beam_grid(beam_config, device=device)
    
    grid_xyz = beam_grid['grid_xyz']  # (div1_2, div2_2, divls_2, 3)
    
    # 提取检测器位置：光束方向的第一个索引（offset=0）
    detector_coords = grid_xyz[:, :, 0, :]  # (div1_2, div2_2, 3)
    
    return detector_coords, detector_grid


def visualize_beam_geometry(
    beam_grid: Dict[str, torch.Tensor],
    config: GENEConfig = None,
    save_path: str = None
):
    """
    可视化光束几何（用于调试）
    
    Args:
        beam_grid: compute_beam_grid的输出
        config: GENE配置（可选，用于显示托卡马克边界）
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("需要matplotlib进行可视化")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制光束路径
    grid = beam_grid['grid_xyz'].cpu().numpy()
    div1, div2, divls, _ = grid.shape
    
    # 绘制中心线
    center = grid[div1//2, div2//2, :, :]
    ax.plot(center[:, 0], center[:, 1], center[:, 2], 'r-', linewidth=2, label='Beam center')
    
    # 绘制起点和终点
    start = beam_grid['beam_start'].cpu().numpy()
    end = beam_grid['beam_end'].cpu().numpy()
    ax.scatter([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
               c='red', s=100, marker='o', label='Start/End')
    
    # 采样一些光束线
    for i in range(0, div1, max(1, div1//2)):
        for j in range(0, div2, max(1, div2//2)):
            line = grid[i, j, ::10, :]  # 每10个点采样一次
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b.', alpha=0.3, markersize=1)
    
    # 如果有配置，绘制托卡马克边界
    if config is not None and config.GRC is not None:
        GRC = config.GRC.cpu().numpy()
        GZC = config.GZC.cpu().numpy()
        
        # 绘制几条poloidal截面
        n_phi = 8
        for i_phi in range(n_phi):
            phi = i_phi * 2 * np.pi / n_phi
            x_torus = GRC[-1, :] * np.cos(phi)
            y_torus = GRC[-1, :] * np.sin(phi)
            z_torus = GZC[-1, :]
            ax.plot(x_torus, y_torus, z_torus, 'k-', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('PCI Beam Geometry')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

