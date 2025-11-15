"""
数据加载器

读取GENE参数文件、平衡态数据、光束配置和仿真数据
"""

import re
import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Optional, List
from .config import GENEConfig, BeamConfig
from .utils import to_tensor


def parse_fortran_namelist(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    解析Fortran namelist格式的参数文件
    
    Args:
        file_path: parameters.dat文件路径
    
    Returns:
        嵌套字典，外层key是namelist名称，内层是参数
    """
    namelists = {}
    current_namelist = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            
            # 检测namelist开始 (如 &parallelization)
            if line.startswith('&'):
                current_namelist = line[1:].strip()
                namelists[current_namelist] = {}
                continue
            
            # 检测namelist结束 (/)
            if line == '/':
                current_namelist = None
                continue
            
            # 解析参数行 (如 n_procs_s = 1)
            if current_namelist and '=' in line:
                # 移除注释
                if '!' in line:
                    line = line[:line.index('!')]
                
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # 解析值
                    value = parse_value(value_str)
                    namelists[current_namelist][key] = value
    
    return namelists


def parse_value(value_str: str) -> Any:
    """解析Fortran参数值"""
    value_str = value_str.strip()
    
    # 移除尾部逗号
    if value_str.endswith(','):
        value_str = value_str[:-1].strip()
    
    # 布尔值
    if value_str.upper() in ('T', '.TRUE.', 'TRUE'):
        return True
    if value_str.upper() in ('F', '.FALSE.', 'FALSE'):
        return False
    
    # 字符串（带引号）
    if value_str.startswith("'") or value_str.startswith('"'):
        return value_str.strip("'\"")
    
    # 科学计数法 (如 0.25000000E-01)
    if 'E' in value_str.upper() or 'D' in value_str.upper():
        value_str = value_str.upper().replace('D', 'E')
        try:
            return float(value_str)
        except ValueError:
            return value_str
    
    # 数组（空格分隔）
    if ' ' in value_str:
        parts = value_str.split()
        try:
            return [parse_value(p) for p in parts]
        except:
            return value_str
    
    # 整数
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # 浮点数
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # 返回原始字符串
    return value_str


def load_gene_config(
    parameters_file: str,
    equilibrium_dir: str = None,
    device: str = 'cuda'
) -> GENEConfig:
    """
    从parameters.dat和平衡态数据加载GENE配置
    
    注意：此函数内部调用 load_gene_config_from_parameters 以确保
    m_ref 和 n_ref 的单位处理与MATLAB一致。
    
    Args:
        parameters_file: parameters.dat文件路径
        equilibrium_dir: 平衡态数据目录（包含equdata_*文件）
        device: PyTorch设备
    
    Returns:
        GENEConfig对象
    """
    # 统一使用 load_gene_config_from_parameters 以确保单位处理一致
    # 这样可以避免两套逻辑导致 m_ref / n_ref / rho_ref 不一致的问题
    return load_gene_config_from_parameters(parameters_file, equilibrium_dir, device)


def load_beam_config(ls_condition_file: str) -> BeamConfig:
    """
    从LS_condition_JT60SA.txt加载光束配置
    
    Args:
        ls_condition_file: LS_condition文件路径
    
    Returns:
        BeamConfig对象
    """
    with open(ls_condition_file, 'r') as f:
        lines = f.readlines()
    
    # 过滤掉注释和空行
    data_lines = [line.strip() for line in lines 
                  if line.strip() and not line.strip().startswith('#')]
    
    # 第1行: 入射点和检测点 (R[mm], Z[mm], phi[0-1])
    coords = [float(x.strip()) for x in data_lines[0].split(',')]
    # 关键修正: 添加单位转换 (mm -> m)，与MATLAB B2=B2/1000.0对应
    injection_point = (coords[0] / 1000.0, coords[1] / 1000.0, coords[2])
    detection_point = (coords[3] / 1000.0, coords[4] / 1000.0, coords[5])
    
    # 第2行: 宽度 (wid1[m], wid2[m])
    widths = [float(x.strip()) for x in data_lines[1].split(',')]
    width_vertical = widths[0]
    width_toroidal = widths[1]
    
    # 第3行: 网格点数 (div1, div2, divls)
    divs = [int(x.strip()) for x in data_lines[2].split(',')]
    div_vertical = divs[0]
    div_toroidal = divs[1]
    div_beam = divs[2]
    
    return BeamConfig(
        injection_point=injection_point,
        detection_point=detection_point,
        width_vertical=width_vertical,
        width_toroidal=width_toroidal,
        div_vertical=div_vertical,
        div_toroidal=div_toroidal,
        div_beam=div_beam
    )


def load_gene_data(
    file_path: str,
    config: GENEConfig,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    加载GENE仿真数据文件（TORUSIons_act_XXXX.dat）
    
    Args:
        file_path: 数据文件路径
        config: GENE配置
        device: PyTorch设备
    
    Returns:
        密度场张量 (ntheta, nx, nz)
    """
    # 读取二进制文件
    data = np.fromfile(file_path, dtype=np.float64)
    
    # Reshape数据
    # 数据格式: (KYMt, cols) 其中 KYMt = 400 (poloidal), cols = nx * nz
    if config.KYMt is None:
        # 根据数据推断维度
        # 假设KYMt = 400（标准poloidal网格数）
        config.KYMt = 400
        config.KZMt = len(data) // (400 * config.nx0) - 1
        config.LYM2 = config.KYMt
        config.LZM2 = config.KZMt + 1
    
    # 重构数据
    total_size = len(data)
    print(f"DEBUG: total_size={total_size}, KYMt={config.KYMt}, nx0={config.nx0}, KZMt={config.KZMt}")
    
    # 计算每个z层面的数据量
    data_per_z = config.KYMt * config.nx0
    n_z_actual = total_size // data_per_z
    print(f"DEBUG: data_per_z={data_per_z}, n_z_actual={n_z_actual}")
    
    # Reshape到2D: (KYMt, nx0 * n_z)
    data_2d = data.reshape(config.KYMt, total_size // config.KYMt)
    print(f"DEBUG: data_2d shape={data_2d.shape}")
    
    # 进一步reshape到 (ntheta, nx, nz)
    # 假设数据排列为：每个z层面连续排列
    nx = config.nx0
    nz = n_z_actual
    
    # 检查是否能整除
    if data_2d.shape[1] % nx != 0:
        print(f"Warning: 无法整除，data_2d.shape[1]={data_2d.shape[1]}, nx={nx}")
        # 使用实际的除法结果
        nx = data_2d.shape[1] // nz
    
    print(f"DEBUG: 最终形状: ({config.KYMt}, {nx}, {nz})")
    data_3d = data_2d.reshape(config.KYMt, nx, nz)
    
    # 确保数据形状匹配MATLAB: (ntheta, nx, nz) = (极向, 径向, phi)
    # 已经是正确的形状，直接使用
    # data_3d = data_3d.transpose(1, 0, 2)  # 不需要转置了
    final_data = data_3d
    
    # 转换为PyTorch张量
    tensor = to_tensor(final_data, device=device)
    
    return tensor


def load_equdata_BZ(
    file_path: str,
    config: 'GENEConfig',
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    读取equdata_BZ文件（平衡态坐标数据）
    
    基于MATLAB的fread_EQcod3.m
    
    Args:
        file_path: equdata_BZ文件路径
        config: GENE配置对象（需要nx0, KZMt等参数）
        device: PyTorch设备
    
    Returns:
        包含GRC, GZC, GFC, PA, GAC, GTC_f, GTC_c等的字典
    """
    import struct
    
    # 以Little-Endian格式读取二进制文件
    with open(file_path, 'rb') as f:
        # 读取网格维度
        NSGMAX = struct.unpack('<i', f.read(4))[0]
        NTGMAX = struct.unpack('<i', f.read(4))[0]
        
        # 确定径向和极向网格数
        if hasattr(config, 'nx0') and config.nx0 is not None:
            IRMAX = config.nx0 + config.inside + config.outside
            # LYM2和KZMt可能还未设置（在读取时间数据前）
            if config.LYM2 is not None and config.KZMt is not None:
                ITGMAX = config.LYM2 // (config.KZMt + 1)
            else:
                ITGMAX = 400  # 默认poloidal网格数
        else:
            IRMAX = 128
            ITGMAX = 128
        
        # 读取坐标数据
        GRC2 = np.fromfile(f, dtype='<f8', count=NSGMAX * NTGMAX)
        GZC2 = np.fromfile(f, dtype='<f8', count=NSGMAX * NTGMAX)
        GFC2 = np.fromfile(f, dtype='<f8', count=NSGMAX * NTGMAX)
        GQPS = np.fromfile(f, dtype='<f8', count=NSGMAX)
    
    # Reshape坐标数据（MATLAB按列主序存储，需要使用Fortran顺序）
    GRCt = np.zeros((NSGMAX, NTGMAX + 1))
    GZCt = np.zeros((NSGMAX, NTGMAX + 1))
    GFCt = np.zeros((NSGMAX, NTGMAX + 1))
    
    GRCt[:, :NTGMAX] = GRC2.reshape(NSGMAX, NTGMAX, order='F')
    GZCt[:, :NTGMAX] = GZC2.reshape(NSGMAX, NTGMAX, order='F')
    GFCt[:, :NTGMAX] = GFC2.reshape(NSGMAX, NTGMAX, order='F')
    
    # 径向插值：从NSGMAX到IRMAX
    GRCt2 = np.zeros((IRMAX + 1, NTGMAX + 1))
    GZCt2 = np.zeros((IRMAX + 1, NTGMAX + 1))
    GFCt2 = np.zeros((IRMAX + 1, NTGMAX + 1))
    
    NSG = NSGMAX - 1  # MATLAB: NSG=NSGMAX-1;
    s2 = -1  # MATLAB: s2=0; (Python索引从0开始，所以初始值是-1)
    # MATLAB: for a=1:NSG (Python中a从0到NSG-1，对应MATLAB的1到NSG)
    for a in range(NSG):
        s1 = s2 + 1  # MATLAB: s1=s2+1; (第一次迭代：s1=0在Python中)
        # MATLAB: s2=fix(IRMAX/NSG*a)+1; 
        # MATLAB索引从1开始需要+1，Python从0开始不需要
        s2 = int(np.fix(IRMAX / NSG * (a + 1)))
        
        # 确保s2不超过IRMAX
        if s2 > IRMAX:
            s2 = IRMAX
        
        # MATLAB: [s1:s2] 包含s1和s2，共(s2-s1+1)个元素
        # Python: s1:s2+1 包含s1到s2，共(s2-s1+1)个元素
        n_pts = s2 - s1 + 1
        if n_pts > 0:
            # MATLAB: ([s1:s2]-1) 从1-based索引转换为0-based索引
            # MATLAB中s1和s2是从1开始的，[s1:s2]是[s1, s1+1, ..., s2]
            # ([s1:s2]-1)是[s1-1, s1, ..., s2-1]，这是0-based索引
            # Python中s1和s2已经是0-based，所以np.arange(s1, s2+1)已经对应MATLAB的([s1:s2]-1)
            # 不需要再减1，否则会整体平移一格
            rows = np.arange(s1, s2 + 1)  # 0-based，对应MATLAB的([s1:s2]-1)
            aM = a + 1  # MATLAB的a（1-based）
            
            # MATLAB: w1=(a-([s1:s2]-1)*NSG/IRMAX).'*ones(1,NTGMAX+1);
            w1 = (aM - rows * NSG / IRMAX)[:, np.newaxis]
            # MATLAB: w2=-(a-1-([s1:s2]-1)*NSG/IRMAX).'*ones(1,NTGMAX+1);
            w2 = (-(aM - 1 - rows * NSG / IRMAX))[:, np.newaxis]
            
            # MATLAB: GRCt2(s1:s2,:)=ones(s2-s1+1,1)*GRCt(a,:).*w1 + ones(s2-s1+1,1)*GRCt(a+1,:).*w2;
            GRCt2[s1:s2 + 1, :] = w1 * GRCt[a, :] + w2 * GRCt[a + 1, :]
            GZCt2[s1:s2 + 1, :] = w1 * GZCt[a, :] + w2 * GZCt[a + 1, :]
            GFCt2[s1:s2 + 1, :] = w1 * GFCt[a, :] + w2 * GFCt[a + 1, :]
    
    GRCt2[0, :] = GRCt[0, :]
    GRCt2[-1, :] = GRCt[-1, :]
    GZCt2[0, :] = GZCt[0, :]
    GZCt2[-1, :] = GZCt[-1, :]
    GFCt2[0, :] = GFCt[0, :]
    GFCt2[-1, :] = GFCt[-1, :]
    GRCt2[:, NTGMAX] = GRCt2[:, 0]
    GZCt2[:, NTGMAX] = GZCt2[:, 0]
    GFCt2[:, NTGMAX] = GFCt2[:, 0]
    
    # 计算plasma axis（磁轴）
    PA = np.array([GRCt2[0, 0], GZCt2[0, 0]])
    
    # 计算theta坐标
    GTC_f_t = np.tile(np.linspace(0, 2*np.pi, NTGMAX + 1), (IRMAX + 1, 1))
    GTC_c_t = np.mod(np.arctan2(GZCt2 - PA[1], GRCt2 - PA[0]), 2*np.pi)
    GTC_c_t[:, -1] = 2.0 * np.pi
    
    # 处理theta不连续性
    for b in range(1, IRMAX + 1):
        ev_r1 = (GTC_c_t[b, 1:-1] - GTC_c_t[b, :-2]) * (GTC_c_t[b, 2:] - GTC_c_t[b, 1:-1])
        ev_r2 = np.where(ev_r1 < 0.0)[0]
        if len(ev_r2) > 0:
            GTC_c_t[b, :ev_r2[0] + 1] -= 2.0 * np.pi
    
    GTC_c_t[0, :] = GTC_c_t[1, :]
    
    # 极向插值：从NTGMAX到ITGMAX
    # 完全对应MATLAB代码第84-105行
    GRC = np.zeros((IRMAX + 1, ITGMAX + 1))
    GZC = np.zeros((IRMAX + 1, ITGMAX + 1))
    GFC = np.zeros((IRMAX + 1, ITGMAX + 1))
    GTC_f = np.zeros((IRMAX + 1, ITGMAX + 1))
    GTC_c = np.tile(np.linspace(0, 2*np.pi, ITGMAX + 1), (IRMAX + 1, 1))
    
    dtheta1 = 2.0 * np.pi / NTGMAX
    dtheta2 = 2.0 * np.pi / ITGMAX
    NSG = NTGMAX
    
    # MATLAB: for b=1:IRMAX+1
    for b in range(IRMAX + 1):
        s2 = -1  # MATLAB: s2=0; (Python索引从0开始，所以初始值是-1)
        # MATLAB: for a=1:NSG
        for a in range(NSG):
            s1 = s2 + 1  # MATLAB: s1=s2+1; (第一次迭代：s1=0在Python中)
            # MATLAB: s2=fix(GTC_c_t(b,a+1)/dtheta2)+1;
            # MATLAB索引从1开始需要+1，Python从0开始不需要
            s2_new = int(np.fix(GTC_c_t[b, a + 1] / dtheta2))
            
            # GTC_c_t可能有负值（处理不连续性时），导致s1或s2为负或超出范围
            # 如果有效，才进行赋值（等同于MATLAB的空切片赋值）
            if s1 >= 0 and s2_new >= 0 and s1 <= ITGMAX and s2_new <= ITGMAX and s2_new >= s1:
                n_pts = s2_new - s1 + 1
                if n_pts > 0:
                    # MATLAB: ([s1:s2]-1) 从1-based索引转换为0-based索引
                    # MATLAB中s1和s2是从1开始的，[s1:s2]是[s1, s1+1, ..., s2]
                    # ([s1:s2]-1)是[s1-1, s1, ..., s2-1]，这是0-based索引
                    # Python中s1和s2_new已经是0-based，所以np.arange(s1, s2_new+1)已经对应MATLAB的([s1:s2]-1)
                    # 不需要再减1，否则会整体平移一格
                    cols = np.arange(s1, s2_new + 1)  # 0-based，对应MATLAB的([s1:s2]-1)
                    
                    # 避免除零错误
                    denom = GTC_c_t[b, a + 1] - GTC_c_t[b, a]
                    if abs(denom) >= 1e-10:
                        # MATLAB: w1=(GTC_c_t(b,a+1)-([s1:s2]-1)*dtheta2)/(GTC_c_t(b,a+1)-GTC_c_t(b,a));
                        w1 = (GTC_c_t[b, a + 1] - cols * dtheta2) / denom
                        # MATLAB: w2=-(GTC_c_t(b,a)-([s1:s2]-1)*dtheta2)/(GTC_c_t(b,a+1)-GTC_c_t(b,a));
                        w2 = -(GTC_c_t[b, a] - cols * dtheta2) / denom
                        
                        # MATLAB: GRC(b,s1:s2)=GRCt2(b,a)*ones(1,s2-s1+1).*w1 + GRCt2(b,a+1)*ones(1,s2-s1+1).*w2;
                        GRC[b, s1:s2_new + 1] = GRCt2[b, a] * w1 + GRCt2[b, a + 1] * w2
                        GZC[b, s1:s2_new + 1] = GZCt2[b, a] * w1 + GZCt2[b, a + 1] * w2
                        GFC[b, s1:s2_new + 1] = GFCt2[b, a] * w1 + GFCt2[b, a + 1] * w2
                        GTC_f[b, s1:s2_new + 1] = GTC_f_t[b, a] * w1 + GTC_f_t[b, a + 1] * w2
            
            # 更新s2用于下一次迭代
            s2 = s2_new
    
    GRC[:, 0] = GRCt2[:, 0]
    GRC[:, -1] = GRCt2[:, -1]
    GZC[:, 0] = GZCt2[:, 0]
    GZC[:, -1] = GZCt2[:, -1]
    GFC[:, 0] = GFCt2[:, 0]
    GFC[:, -1] = GFCt2[:, -1]
    GTC_c[:, -1] = 2.0 * np.pi
    
    # 计算minor radius - 直接使用最终的GRC和GZC，与MATLAB一致
    # MATLAB: obj.GAC = sqrt((GRC-PA(1)).^2 + (GZC-PA(2)).^2);
    GAC = np.sqrt((GRC - PA[0])**2 + (GZC - PA[1])**2)
    
    # 转换为Tensor
    result = {
        'NSGMAX': NSGMAX,
        'NTGMAX': ITGMAX,
        'GRC': to_tensor(GRC, device),
        'GZC': to_tensor(GZC, device),
        'GFC': to_tensor(GFC, device),
        'PA': to_tensor(PA, device),
        'GAC': to_tensor(GAC, device),
        'GTC_f': to_tensor(GTC_f, device),
        'GTC_c': to_tensor(GTC_c, device),
        'Rmax': float(np.max(GRC)),
        'Rmin': float(np.min(GRC)),
        'Zmax': float(np.max(GZC)),
        'Zmin': float(np.min(GZC)),
    }
    
    return result


def load_equdata_be(
    file_path: str,
    PA: np.ndarray,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    读取equdata_be文件（平衡态磁场数据）
    
    基于MATLAB的fread_EQmag.m
    
    Args:
        file_path: equdata_be文件路径
        PA: Plasma axis位置 [R, Z]
        device: PyTorch设备
    
    Returns:
        包含磁场分量GBPR, GBPZ, GBTP, GBPP和网格信息的字典
    """
    import struct
    
    NPHIGM_set = 68  # 默认toroidal网格数
    
    # 以Big-Endian格式读取二进制文件
    with open(file_path, 'rb') as f:
        # 读取网格维度
        NRGM = struct.unpack('>i', f.read(4))[0]
        NZGM = struct.unpack('>i', f.read(4))[0]
        NPHIGM = struct.unpack('>i', f.read(4))[0]
        
        # 读取网格参数
        RG = np.fromfile(f, dtype='>f8', count=6)
        DR = np.fromfile(f, dtype='>f8', count=3)
        
        # 读取磁场分量
        GBPR2 = np.fromfile(f, dtype='>f8', count=NRGM * NZGM * NPHIGM)
        GBPZ2 = np.fromfile(f, dtype='>f8', count=NRGM * NZGM * NPHIGM)
        GBTP2 = np.fromfile(f, dtype='>f8', count=NRGM * NZGM * NPHIGM)
        GBPP2 = np.fromfile(f, dtype='>f8', count=NRGM * NZGM * NPHIGM)
    
    # 处理NPHIGM=1的特殊情况
    if NPHIGM == 1:
        NPHIGM = NPHIGM_set
        GBPR2 = np.tile(GBPR2, NPHIGM)
        GBPZ2 = np.tile(GBPZ2, NPHIGM)
        GBTP2 = np.tile(GBTP2, NPHIGM)
        GBPP2 = np.tile(GBPP2, NPHIGM)
        RG[4] = 0.0
        DR[2] = 2.0 * np.pi / (NPHIGM - 1)
    
    # Reshape为3D数组（Fortran order）
    GBPR = GBPR2.reshape(NRGM, NZGM, NPHIGM, order='F')
    GBPZ = GBPZ2.reshape(NRGM, NZGM, NPHIGM, order='F')
    GBTP = GBTP2.reshape(NRGM, NZGM, NPHIGM, order='F')
    GBPP = GBPP2.reshape(NRGM, NZGM, NPHIGM, order='F')
    
    # 转换为native字节序（PyTorch不支持非native字节序）
    GBPR = GBPR.astype(np.float64, copy=False)
    GBPZ = GBPZ.astype(np.float64, copy=False)
    GBTP = GBTP.astype(np.float64, copy=False)
    GBPP = GBPP.astype(np.float64, copy=False)
    RG = RG.astype(np.float64, copy=False)
    DR = DR.astype(np.float64, copy=False)
    
    # 计算B0（plasma axis处的磁场强度）
    # 使用简化方法：取toroidal磁场的中心值
    B0 = float(np.mean(GBTP[:, :, 0]))
    
    # 转换为Tensor
    result = {
        'NRGM': NRGM,
        'NZGM': NZGM,
        'NPHIGM': NPHIGM,
        'RG1': RG[:2].tolist(),
        'RG2': RG[2:4].tolist(),
        'RG3': RG[4:6].tolist(),
        'DR1': float(DR[0]),
        'DR2': float(DR[1]),
        'DR3': float(DR[2]),
        'GBPR_3d': to_tensor(GBPR, device),
        'GBPZ_3d': to_tensor(GBPZ, device),
        'GBTP_3d': to_tensor(GBTP, device),
        'GBPP_3d': to_tensor(GBPP, device),
        'GBPR_2d': to_tensor(GBPR[:, :, 0], device),
        'GBPZ_2d': to_tensor(GBPZ[:, :, 0], device),
        'GBTP_2d': to_tensor(GBTP[:, :, 0], device),
        'GBPP_2d': to_tensor(GBPP[:, :, 0], device),
        'B0': B0,
    }
    
    return result


def load_equilibrium_data(
    equilibrium_dir: str,
    config: 'GENEConfig',
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    加载所有平衡态数据
    
    Args:
        equilibrium_dir: 平衡态数据目录
        config: GENE配置对象
        device: PyTorch设备
    
    Returns:
        包含所有平衡态张量的字典
    """
    eq_data = {}
    
    eq_dir = Path(equilibrium_dir)
    
    # 加载equdata_BZ（坐标数据）
    equdata_bz_path = eq_dir / 'equdata_BZ'
    if equdata_bz_path.exists():
        print(f"加载 {equdata_bz_path}...")
        bz_data = load_equdata_BZ(str(equdata_bz_path), config, device)
        eq_data.update(bz_data)
    else:
        print(f"警告: 未找到 {equdata_bz_path}")
    
    # 加载equdata_be（磁场数据）
    equdata_be_path = eq_dir / 'equdata_be'
    if equdata_be_path.exists() and 'PA' in eq_data:
        print(f"加载 {equdata_be_path}...")
        PA_np = eq_data['PA'].cpu().numpy()
        be_data = load_equdata_be(str(equdata_be_path), PA_np, device)
        eq_data.update(be_data)
    else:
        if not equdata_be_path.exists():
            print(f"警告: 未找到 {equdata_be_path}")
    
    return eq_data


def peek_parameters_header(path: str):
    """
    调试函数：检查parameters.dat文件头部
    
    用于判断文件开头是否有二进制int（RET），对应MATLAB的：
    RET = fread(fid, 1, 'int');
    
    Args:
        path: parameters.dat文件路径
    """
    with open(path, 'rb') as f:
        raw4 = f.read(4)
        print("前4字节 raw =", raw4)
        print("前4字节 hex =", raw4.hex())
        if len(raw4) == 4:
            # 尝试解析为int32 (little-endian和big-endian)
            int_le = int.from_bytes(raw4, byteorder='little', signed=True)
            int_be = int.from_bytes(raw4, byteorder='big', signed=True)
            print(f"解析为int32 (little-endian) = {int_le}")
            print(f"解析为int32 (big-endian) = {int_be}")
            # 尝试解析为ASCII
            try:
                ascii_str = raw4.decode('ascii', errors='replace')
                print(f"解析为ASCII = {repr(ascii_str)}")
            except:
                print("无法解析为ASCII")
    
    with open(path, 'r', errors='replace') as f:
        first_line = f.readline()
        print("第一行(文本模式) =", repr(first_line))


def parse_parameters_dat(file_path: str) -> Dict[str, any]:
    """
    解析parameters.dat文件（GENE namelist格式）
    
    基于MATLAB的fread_param2.m
    注意：MATLAB版本会先读取一个4字节的int（RET），然后才开始读取文本
    
    Args:
        file_path: parameters.dat文件路径
    
    Returns:
        包含所有参数的字典
    """
    param_dict = {}
    current_section = ''
    
    # 尝试检测并跳过可能的二进制int头部（MATLAB的RET）
    # 先以二进制模式读取前4字节，检查是否是文本
    skip_bytes = 0
    with open(file_path, 'rb') as f_bin:
        first_4_bytes = f_bin.read(4)
        # 检查前4字节是否看起来像文本（ASCII可打印字符或&开头）
        if len(first_4_bytes) == 4:
            try:
                # 尝试解码为ASCII
                text_start = first_4_bytes.decode('ascii', errors='strict')
                # 如果第一个字符是&或可打印ASCII，说明是文本，不需要跳过
                if text_start[0] == '&' or (32 <= ord(text_start[0]) <= 126):
                    skip_bytes = 0
                else:
                    # 不是文本开头，可能是二进制int，需要跳过
                    skip_bytes = 4
            except:
                # 解码失败，可能是二进制int，需要跳过
                skip_bytes = 4
    
    # 以文本模式打开，如果需要则跳过前4字节
    with open(file_path, 'r', errors='replace') as f:
        if skip_bytes > 0:
            f.seek(skip_bytes)  # 跳过前4字节（对应MATLAB的RET）
        
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('!') or line == '/':
                continue
            
            # 检测section（&开头）
            if line.startswith('&'):
                current_section = line[1:].strip()
                continue
            
            # 解析键值对
            if '=' in line:
                # 移除行末注释
                if '!' in line:
                    line = line.split('!')[0].strip()
                
                key_value = line.split('=', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value_str = key_value[1].strip().rstrip(',')
                    
                    # 尝试转换为数值
                    # 先清理字符串：处理Fortran的D指数（D -> E）
                    value_str_clean = value_str.strip().upper().replace('D', 'E')
                    try:
                        # 尝试整数
                        if '.' not in value_str_clean and 'E' not in value_str_clean:
                            value = int(value_str_clean)
                        else:
                            # 浮点数（支持E指数，包括从D转换来的）
                            value = float(value_str_clean)
                    except ValueError:
                        # 保持为字符串（去除引号）
                        value = value_str.strip('"').strip("'")
                    
                    # 存储为section.key格式
                    if current_section:
                        full_key = f'{current_section}.{key}'
                    else:
                        full_key = key
                    
                    param_dict[full_key] = value
    
    return param_dict


def load_gene_config_from_parameters(
    parameters_file: str,
    equilibrium_dir: Optional[str] = None,
    device: str = 'cuda'
) -> GENEConfig:
    """
    从parameters.dat和equilibrium数据完整加载GENE配置
    
    Args:
        parameters_file: parameters.dat文件路径
        equilibrium_dir: equilibrium数据目录（可选）
        device: PyTorch设备
    
    Returns:
        完整的GENEConfig对象
    """
    # 解析parameters.dat
    param_dict = parse_parameters_dat(parameters_file)
    
    # 创建GENEConfig对象并填充参数
    config = GENEConfig()
    
    # 映射parameters.dat的参数到config
    param_mapping = {
        # 并行化
        'parallelization.n_procs_s': 'n_procs_s',
        'parallelization.n_procs_v': 'n_procs_v',
        'parallelization.n_procs_w': 'n_procs_w',
        'parallelization.n_procs_x': 'n_procs_x',
        'parallelization.n_procs_y': 'n_procs_y',
        'parallelization.n_procs_z': 'n_procs_z',
        'parallelization.n_procs_sim': 'n_procs_sim',
        
        # box
        'box.n_spec': 'n_spec',
        'box.nx0': 'nx0',
        'box.nky0': 'nky0',
        'box.nz0': 'nz0',
        'box.nv0': 'nv0',
        'box.nw0': 'nw0',
        'box.kymin': 'kymin',
        'box.lv': 'lv',
        'box.lw': 'lw',
        'box.lx': 'lx',
        'box.nexc': 'nexc',
        
        # geometry
        'geometry.q0': 'q0',
        'geometry.shat': 'shat',
        'geometry.trpeps': 'trpeps',
        'geometry.major_R': 'major_R',
        
        # general
        'general.beta': 'beta',
        'general.debye2': 'debye2',
        
        # species - 第一个物种
        'species_1.name': 'name1',
        'species_1.omn': 'omn1',
        'species_1.omt': 'omt1',
        'species_1.mass': 'mass1',
        'species_1.temp': 'temp1',
        'species_1.dens': 'dens1',
        'species_1.charge': 'charge1',
        
        # species - 第二个物种（如果有）
        'species_2.name': 'name2',
        'species_2.omn': 'omn2',
        'species_2.omt': 'omt2',
        'species_2.mass': 'mass2',
        'species_2.temp': 'temp2',
        'species_2.dens': 'dens2',
        'species_2.charge': 'charge2',
        
        # units
        'units.Bref': 'B_ref',
        'units.Tref': 'T_ref',
        'units.nref': 'n_ref',
        'units.Lref': 'L_ref',
        'units.mref': 'm_ref',
    }
    
    for param_key, config_attr in param_mapping.items():
        if param_key in param_dict:
            value = param_dict[param_key]
            # 特殊处理m_ref：MATLAB中是units.mref * 1.6726232e-27
            if param_key == 'units.mref':
                # 物理常数
                m_proton = 1.6726232e-27  # 质子质量 [kg]
                value = value * m_proton  # 转换为物理质量
            # 特殊处理n_ref：MATLAB中是units.nref * 1e19
            elif param_key == 'units.nref':
                # 检查是否需要单位修正（对于氢等离子体，约8.5）
                if 8.0 <= value <= 9.0:
                    value = value * 1e19  # 单位修正
            setattr(config, config_attr, value)
    
    # 重要：重新计算物理参数，因为可能使用了错误的m_ref初始值
    config.compute_physics_params()

    # ===== 关键：按照 MATLAB fread_param2 计算 inside / outside / IRMAX =====
    # 对应 MATLAB：
    # minor_r = 1.18
    # IN  = (trpeps*major_R*L_ref/(rho_ref*lx) - 1/2) * nx0;
    # OUT = ((minor_r - trpeps*major_R*L_ref)/(rho_ref*lx) - 1/2) * nx0;
    # inside  = round(IN); outside = round(OUT); IRMAX = nx0;
    try:
        needed = [config.trpeps, config.major_R, config.L_ref,
                  config.rho_ref, config.lx, config.nx0]
        if all(v is not None for v in needed):
            minor_r = 1.18
            IN = (config.trpeps * config.major_R * config.L_ref /
                  (config.rho_ref * config.lx) - 0.5) * config.nx0
            OUT = ((minor_r - config.trpeps * config.major_R * config.L_ref) /
                   (config.rho_ref * config.lx) - 0.5) * config.nx0

            config.inside = int(round(IN))
            config.outside = int(round(OUT))
            config.IRMAX = config.nx0
        else:
            # 如果有什么缺的，保持默认值（inside/outside=0）
            pass
    except Exception as e:
        print(f"[WARN] failed to compute inside/outside: {e}")

    # 加载equilibrium数据（如果提供）
    if equilibrium_dir:
        eq_data = load_equilibrium_data(equilibrium_dir, config, device)
        config.update_from_dict(eq_data)
    
    return config


def separate_torusdata(
    input_file: str,
    output_dir: str,
    time_n: int,
    tol_n: int
) -> List[str]:
    """
    分割大型TORUSIons_act.dat为时间序列文件
    
    基于MATLAB的separate_torusdata.m
    
    Args:
        input_file: 输入文件路径（TORUSIons_act.dat）
        output_dir: 输出目录
        time_n: 时间快照数量
        tol_n: 环面角度切片数量+1
    
    Returns:
        生成的文件路径列表
    """
    import os
    
    output_files = []
    
    with open(input_file, 'r') as fid:
        # 跳过前5行头部
        for _ in range(5):
            fid.readline()
        
        for l in range(time_n):
            # 重复处理tol_n次
            for k in range(tol_n):
                if not fid:
                    break
                
                # 跳过空行
                line = fid.readline()
                while line and not line.strip():
                    line = fid.readline()
                
                if not line:
                    break
                
                # 处理第一行（phi和t值）
                if 'phi =' in line and 't =' in line:
                    # 提取t值
                    t_match = re.search(r't\s*=\s*([\d\.\-\+eE]+)', line)
                    if t_match:
                        t = float(t_match.group(1))
                        time_int = int(np.floor(t * 100))
                        
                        # 创建或打开输出文件
                        outfile = os.path.join(output_dir, f'TORUSIons_act_{time_int}.dat')
                        
                        if k == 0:
                            mode = 'w'  # 第一个phi切面，写模式
                        else:
                            mode = 'a'  # 后续phi切面，追加模式
                        
                        with open(outfile, mode) as fout:
                            # 写入phi和t行
                            fout.write(line)
                            
                            # 读取并写入接下来的400行数值数据
                            for i in range(400):
                                data_line = fid.readline()
                                if not data_line:
                                    break
                                fout.write(data_line)
                        
                        if k == 0:
                            output_files.append(outfile)
            
            if output_files:
                print(f'Generated: {output_files[-1]}')
    
    return output_files


def generate_timedata(
    config: GENEConfig,
    text_file: str,
    time_t: float,
    output_dir: str
) -> str:
    """
    将文本格式的TORUSIons数据转换为二进制double格式
    
    基于MATLAB的generate_timedata.m
    
    Args:
        config: GENE配置对象
        text_file: 输入文本文件路径
        time_t: 时间值
        output_dir: 输出目录
    
    Returns:
        生成的二进制文件路径
    """
    import os
    
    # 读取文本文件，忽略#开头的行
    data_lines = []
    with open(text_file, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line and not line.startswith('#'):
                # 解析数值
                values = [float(x) for x in line.split()]
                data_lines.append(values)
    
    # 转换为numpy数组
    data = np.array(data_lines, dtype=np.float64)
    rows, cols = data.shape
    
    # 更新config参数
    config.KYMt = rows
    config.KZMt = rows // 400 - 1  # 400是poloidal mesh数量
    
    # 生成输出文件名
    filename = f'{int(time_t * 100):08d}.dat'
    output_path = os.path.join(output_dir, filename)
    
    # 以二进制格式写入 - 关键修正: 使用行主序与MATLAB的fwrite兼容
    with open(output_path, 'wb') as fid:
        # MATLAB的fwrite使用行主序（C order），Python也应该一致
        fid.write(data.tobytes(order='C'))  # 使用行主序，与MATLAB的fwrite一致
    
    print(f'Generated binary file: {output_path}')
    return output_path


def fread_data_1(
    f_n: int,
    config: GENEConfig,
    binary_file: str,
    m: int,
    n: int,
    device: str = 'hip'
) -> torch.Tensor:
    """
    读取单模态数据（严格对应MATLAB的fread_data_1.m）
    
    Args:
        f_n: 字段数量（通常为6个物理量：PHI, PSI, VPL, NE, TE, TI）
        config: GENE配置对象  
        binary_file: 二进制数据文件路径
        m: 模态编号m
        n: 模态编号n
        device: PyTorch设备
    
    Returns:
        单模态数据张量 (IRMAX+1, f_n) - 复数类型
    """
    # 检查必要的参数
    if config.LKY is None or config.LKZ is None:
        print("警告: LKY和LKZ参数未设置，使用默认模态")
        # 使用默认模态
        config.LKY = torch.arange(0, 6, device=device)
        config.LKZ = torch.arange(0, 6, device=device)
    
    # 找到模态m, n的索引
    L = np.where((config.LKY.cpu().numpy() == m) & (config.LKZ.cpu().numpy() == n))[0]
    if len(L) == 0:
        print(f"警告: 未找到模态 m={m}, n={n}")
        # 返回零数据 (注意MATLAB是IRMAX+1)
        IRMAX = config.nx0 + 1 + config.inside + config.outside
        return torch.zeros((IRMAX, f_n), dtype=torch.complex64, device=device)
    
    # MATLAB中的L是1基索引，Python是0基索引
    L_idx = L[0]  # 取第一个匹配的模态
    
    # 初始化输出数组 (严格按照MATLAB: obj.IRMAX+1)
    IRMAX = config.nx0 + config.inside + config.outside  # 对应MATLAB的IRMAX
    p2 = np.zeros((IRMAX + 1, f_n), dtype=np.complex128)  # MATLAB是IRMAX+1
    
    # 按照MATLAB的逻辑读取数据
    with open(binary_file, 'rb') as fid:
        # 读取RET
        RET = np.frombuffer(fid.read(4), dtype=np.int32)[0]
        
        # 对每个字段进行循环
        for a in range(f_n):
            # 跳过数据
            if a == 0:
                # 第一次：跳过(IRMAX+1)*(L-1)*2个double
                skip_elements = (IRMAX + 1) * L_idx * 2
                if skip_elements > 0:
                    fid.seek(skip_elements * 8, 1)  # 8字节每个double
            else:
                # 后续：跳过(IRMAX+1)*((LYM2)*(LZM2)-1)*2个double
                LYM2 = config.LYM2 if config.LYM2 else 400
                LZM2 = config.LZM2 if config.LZM2 else 4
                skip_elements = (IRMAX + 1) * (LYM2 * LZM2 - 1) * 2
                if skip_elements > 0:
                    fid.seek(skip_elements * 8, 1)
            
            # 读取(2, IRMAX+1)个double数据
            data1 = np.frombuffer(fid.read(2 * (IRMAX + 1) * 8), dtype=np.float64)
            data1 = data1.reshape(2, IRMAX + 1)
            
            # 组合为复数：data1(1,:) + i*data1(2,:)
            p2[:, a] = data1[0, :] + 1j * data1[1, :]
    
    # 🔴 MATLAB数据后处理逻辑 (第23-27行)
    IROUT = 128
    # MATLAB: pout=p2(1:fix(end/IROUT):end,:);  % 采样：每128点取1点
    end_idx = int(np.fix(p2.shape[0] / IROUT)) * IROUT
    p2_sampled = p2[0:end_idx:IROUT, :]  # 从0开始，每128点取1点
    
    # 返回处理后的数据
    p2_tensor = torch.from_numpy(p2_sampled).to(device)
    
    return p2_tensor


def fread_data_s(
    config: GENEConfig,
    binary_file: str,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    读取二进制密度场数据 (0000XXXX.dat)

    对标 MATLAB 的:
        p2 = fread_data_s(f_n, obj, file)

    MATLAB 逻辑简化为:
        data = fread(..., 'double')
        rows = obj.KYMt
        cols = total_elements / rows
        data = reshape(data, rows, cols)
        data2 = zeros(obj.LYM2 / (obj.KZMt + 1), obj.nx0, obj.KZMt + 1)
        for i = 1:(obj.KZMt+1)
            data2(:,:,i) = data(400*(i-1)+1:400*i, :)
        end
        p2 = data2
    """
    # 读取原始 1D double 数据
    data = np.fromfile(binary_file, dtype=np.float64)
    total_elements = data.size

    if config.nx0 is None or config.nx0 <= 0:
        raise ValueError("fread_data_s: config.nx0 必须在调用前设置好")

    # ==== 确保 KYMt / KZMt / LYM2 已经就绪 ====
    # 正常情况下，generate_timedata 已经在 config 上设置了 KYMt / KZMt
    # 如果没设置（例如直接从已有二进制启动），这里做一个兜底推断：
    if not getattr(config, "KYMt", None) or config.KYMt == 0 or \
       not getattr(config, "KZMt", None):

        # 按列数 = nx0 来推 rows
        cols = config.nx0
        if total_elements % cols != 0:
            raise ValueError(
                f"fread_data_s: 数据长度 {total_elements} 不能被 nx0={cols} 整除"
            )

        rows = total_elements // cols  # 等价于 MATLAB 的 obj.KYMt

        # 这个 JT-60SA GENE case 已知 poloidal 点数是 400
        # rows = 400 * (KZMt+1)
        ntheta = 400
        if rows % ntheta != 0:
            raise ValueError(
                f"fread_data_s: rows={rows} 不能拆成 400 * nphi，请检查数据格式"
            )
        nphi = rows // ntheta

        config.KYMt = rows
        config.KZMt = nphi - 1
        config.LYM2 = config.KYMt
        config.LZM2 = config.KZMt + 1
        # 如果 GENEConfig 里有衍生参数，这里统一更新一下
        if hasattr(config, "compute_derived_params"):
            config.compute_derived_params()

    rows = config.KYMt
    cols = total_elements // rows
    data_2d = data[:rows * cols].reshape(rows, cols)

    # ==== 按 MATLAB 方式重排成 3D ====
    dim1 = config.LYM2 // (config.KZMt + 1)  # 对应 400
    dim2 = config.nx0                          # 对应 128
    dim3 = config.KZMt + 1                     # 对应 29

    if dim1 * dim2 * dim3 != total_elements:
        print(
            f"[fread_data_s] 警告: dim1*dim2*dim3={dim1*dim2*dim3} "
            f"≠ total_elements={total_elements}，请检查 KYMt/KZMt/LYM2 设置"
        )

    data3d = np.zeros((dim1, dim2, dim3))
    poloidal = dim1  # 这里就是 400

    for i in range(dim3):  # i = 0..(KZMt)
        start_row = poloidal * i
        end_row = poloidal * (i + 1)
        data3d[:, :, i] = data_2d[start_row:end_row, :]

    # 转成 PyTorch tensor
    tensor = to_tensor(data3d, device=device)
    return tensor


def save_debug_data(config, output_mat_path: str):
    """
    保存 Python 侧第2阶段（平衡态 + 磁场）调试数据到一个 .mat 文件。

    对 run_pci.py 里的调用:
        debug_data = save_debug_data(gene_config, 'debug_stage_2_python.mat')
    完全兼容，返回一个 dict 方便你后面需要时再用。
    """
    import numpy as np
    import torch
    from scipy.io import savemat

    debug_data = {}

    # ===== 1. 磁场 3D 数据 (GBPR, GBPZ, GBTP, GBPP) =====
    for key in ['GBPR_3d', 'GBPZ_3d', 'GBTP_3d', 'GBPP_3d']:
        val = getattr(config, key, None)
        if val is not None:
            if torch.is_tensor(val):
                arr = val.detach().cpu().numpy()
            else:
                arr = np.array(val)
            debug_data[key] = arr
            print(f"DEBUG: Saved {key} with shape: {arr.shape}")

    # ===== 2. 坐标数据: GRC, GZC, GFC, GAC, GTC_f, GTC_c, PA =====
    for key in ['GRC', 'GZC', 'GFC', 'GAC', 'GTC_f', 'GTC_c', 'PA']:
        val = getattr(config, key, None)
        if val is not None:
            if torch.is_tensor(val):
                arr = val.detach().cpu().numpy()
            else:
                arr = np.array(val)
            debug_data[key] = arr
            print(f"DEBUG: Saved {key} with shape: {arr.shape}")

    # ===== 3. 网格参数 RG（6 个 double）=====
    # load_equdata_be 里你是这么存的: RG1, RG2, RG3，每个长度 2
    rg_vals = []
    for key in ['RG1', 'RG2', 'RG3']:
        val = getattr(config, key, None)
        if val is not None:
            rg_vals.extend(list(val))
    if rg_vals:
        rg_arr = np.array(rg_vals, dtype=np.float64)
        debug_data['RG'] = rg_arr
        print(f"DEBUG: Saved RG with shape: {rg_arr.shape}")

    # ===== 4. 网格步长 DR（3 个标量）=====  ★ 这里是之前报错的地方 ★
    dr_vals = []
    for key in ['DR1', 'DR2', 'DR3']:
        if hasattr(config, key):
            dr_vals.append(float(getattr(config, key)))
    if dr_vals:
        # 不再用 np.concatenate，而是直接构造成一维数组
        dr_arr = np.array(dr_vals, dtype=np.float64)
        debug_data['DR'] = dr_arr
        print(f"DEBUG: Saved DR with shape: {dr_arr.shape}")

    # ===== 5. 真正写 mat 文件 =====
    savemat(output_mat_path, debug_data)
    print(f"Python debug data saved to: {output_mat_path}")

    return debug_data

