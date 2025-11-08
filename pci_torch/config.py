"""
配置管理模块

包含GENE配置和光束配置类定义
"""

from typing import Optional, Tuple
import torch
import numpy as np


class GENEConfig:
    """GENE仿真配置参数"""
    
    def __init__(self):
        # ========== 基本仿真参数 ==========
        self.data_n: int = 1              # 数据编号
        self.time_series: bool = False    # 是否为时间序列
        self.var_type: str = 'local'      # 变量类型: 'local' 或 'global'
        
        # 网格参数
        self.nx0: int = 400              # theta方向网格数
        self.nky0: int = 20              # ky方向网格数
        self.nz0: int = 30               # z方向网格数
        
        # 物理参数
        self.q0: float = 1.4             # 安全因子
        self.shat: float = 0.4           # 剪切
        self.trpeps: float = 0.3         # 三角形度
        self.Lref: float = 2.97          # 参考长度
        
        # 粒子种类参数 (species)
        # 物种1 (通常是离子)
        self.name1: str = 'Ions'
        self.omn1: float = 2.5         # 密度梯度
        self.omt1: float = 7.0         # 温度梯度
        self.mass1: float = 1.0        # 质量
        self.temp1: float = 1.0        # 温度
        self.dens1: float = 1.0        # 密度
        self.charge1: float = 1.0      # 电荷
        
        # 物种2 (如果有电子)
        self.name2: Optional[str] = None
        self.omn2: Optional[float] = None
        self.omt2: Optional[float] = None
        self.mass2: Optional[float] = None
        self.temp2: Optional[float] = None
        self.dens2: Optional[float] = None
        self.charge2: Optional[float] = None
        
        # 单位归一化参数 (units)
        self.B_ref: float = 2.25       # 参考磁场 [T]
        self.T_ref: float = 4.0        # 参考温度 [keV]
        self.n_ref: float = 8.5        # 参考密度 [10^19 m^-3]
        self.L_ref: float = 2.97       # 参考长度 [m]
        self.m_ref: float = 1.0        # 参考质量 (质子质量)
        
        # 导出的归一化参数
        self.q_ref: Optional[float] = None      # 参考电荷
        self.c_ref: Optional[float] = None      # 参考速度
        self.omega_ref: Optional[float] = None  # 参考频率
        self.rho_ref: Optional[float] = None    # 参考回旋半径
        
        # ========== 计算得到的参数 ==========
        # 数据维度参数（从实际数据文件计算）
        self.KYMt: Optional[int] = None    # 总的theta点数 (通常400)
        self.KZMt: Optional[int] = None    # toroidal模式数-1
        self.LYM2: Optional[int] = None    # = KYMt (poloidal方向)
        self.LZM2: Optional[int] = None    # = KZMt + 1
        self.LMAX: Optional[int] = None    # 最大模式数
        
        # 扩展参数（用于插值边界处理）
        self.inside: int = 0      # 内侧扩展点数
        self.outside: int = 0     # 外侧扩展点数
        
        # ========== 平衡态数据（作为张量存储） ==========
        # 从equdata_BZ读取
        self.NSGMAX: Optional[int] = None       # 径向网格数
        self.NTGMAX: Optional[int] = None       # 角向网格数
        self.GRC: Optional[torch.Tensor] = None    # (NSGMAX, NTGMAX) R坐标
        self.GZC: Optional[torch.Tensor] = None    # (NSGMAX, NTGMAX) Z坐标
        self.GFC: Optional[torch.Tensor] = None    # 通量坐标
        self.PA: Optional[torch.Tensor] = None     # (2,) 等离子体轴心 [R, Z]
        self.GAC: Optional[torch.Tensor] = None    # (NSGMAX, NTGMAX) 小半径
        self.GTC_f: Optional[torch.Tensor] = None  # theta (flux coords)
        self.GTC_c: Optional[torch.Tensor] = None  # theta (cylindrical coords)
        
        self.Rmax: Optional[float] = None
        self.Rmin: Optional[float] = None
        self.Zmax: Optional[float] = None
        self.Zmin: Optional[float] = None
        
        # 从equdata_be读取 (磁场数据)
        self.NRGM: Optional[int] = None
        self.NZGM: Optional[int] = None
        self.NPHIGM: Optional[int] = None
        self.RG1: Optional[float] = None
        self.RG2: Optional[float] = None
        self.RG3: Optional[float] = None
        self.DR1: Optional[float] = None
        self.DR2: Optional[float] = None
        self.DR3: Optional[float] = None
        self.GBPR_2d: Optional[torch.Tensor] = None  # B_R 2D
        self.GBPZ_2d: Optional[torch.Tensor] = None  # B_Z 2D
        self.GBTP_2d: Optional[torch.Tensor] = None  # B_toroidal 2D
        self.GBPP_2d: Optional[torch.Tensor] = None  # Pressure 2D
        self.GBPR_3d: Optional[torch.Tensor] = None  # B_R 3d
        self.GBPZ_3d: Optional[torch.Tensor] = None  # B_Z 3d
        self.GBTP_3d: Optional[torch.Tensor] = None  # B_toroidal 3d
        self.GBPP_3d: Optional[torch.Tensor] = None  # Pressure 3d
    
    def compute_physics_params(self):
        """计算物理参数"""
        # 计算参考电荷
        self.q_ref = self.charge1 * 1.602176634e-19 if self.charge1 else None
        
        # 计算参考速度
        self.c_ref = np.sqrt(2 * (self.T_ref * 1000 * 1.602176634e-19) / 
                           (self.mass1 * 1.67262192369e-27)) if self.mass1 else None
        
        # 计算参考频率
        self.omega_ref = self.c_ref / self.L_ref if self.c_ref and self.L_ref else None
        
        # 计算参考回旋半径
        self.rho_ref = self.c_ref / (self.B_ref * self.q_ref / self.mass1) if all([
            self.c_ref, self.B_ref, self.q_ref, self.mass1
        ]) else None
    
    def update_from_dict(self, config_dict: dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def compute_derived_params(self):
        """计算衍生参数"""
        # 计算数据维度参数
        if self.KYMt is not None:
            self.LYM2 = self.KYMt
        if self.KZMt is not None:
            self.LZM2 = self.KZMt + 1
            self.LMAX = (self.LZM2 - 1) // 2
        
        # 重新计算物理参数
        self.compute_physics_params()
        
        # 添加FVER属性（GENE版本标识）
        self.FVER = 5  # GENE版本号


class BeamConfig:
    """光束配置，从LS_condition_JT60SA.txt读取"""
    
    def __init__(
        self,
        injection_point: Tuple[float, float, float] = (5000.0, 0.0, 0.1),
        detection_point: Tuple[float, float, float] = (5000.0, 0.0, 0.475),
        width_vertical: float = 0.15,
        width_toroidal: float = 0.15,
        div_vertical: int = 4,
        div_toroidal: int = 12,
        div_beam: int = 3000
    ):
        # 光束起点和终点 (R[mm], Z[mm], phi[0-1])
        # 从input/LS_condition_JT60SA.txt文件读取
        self.injection_point = injection_point
        self.detection_point = detection_point
        
        # 检测器区域尺寸 [m]
        self.width_vertical = width_vertical      # wid1: 垂直方向宽度
        self.width_toroidal = width_toroidal      # wid2: 环向方向宽度
        
        # 网格分辨率（检测器数量）
        # 从input/LS_condition_JT60SA.txt文件读取
        self.div_vertical = div_vertical       # div1: ±4 -> 2*4+1=9个点
        self.div_toroidal = div_toroidal      # div2: ±12 -> 2*12+1=25个点
        self.div_beam = div_beam        # divls: 光束方向采样点数
        
        # 计算导出参数
        self.n_detectors_v = 2 * self.div_vertical + 1      # 垂直方向
        self.n_detectors_t = 2 * self.div_toroidal + 1      # 环向方向
        self.n_beam_points = self.div_beam + 1              # 光束方向点数
    
    @property
    def inj_xyz(self) -> np.ndarray:
        """注入点笛卡尔坐标 [X, Y, Z]"""
        R, Z, phi = self.injection_point
        return np.array([
            R * np.cos(2 * np.pi * phi),
            R * np.sin(2 * np.pi * phi),
            Z
        ]) / 1000.0  # mm to m
    
    @property
    def det_xyz(self) -> np.ndarray:
        """检测点笛卡尔坐标 [X, Y, Z]"""
        R, Z, phi = self.detection_point
        return np.array([
            R * np.cos(2 * np.pi * phi),
            R * np.sin(2 * np.pi * phi),
            Z
        ]) / 1000.0  # mm to m
    
    @property
    def beam_direction(self) -> np.ndarray:
        """光束方向向量"""
        return self.det_xyz - self.inj_xyz
    
    @property
    def beam_length(self) -> float:
        """光束长度"""
        return np.linalg.norm(self.beam_direction)
    
    @property
    def beam_direction_unit(self) -> np.ndarray:
        """光束方向单位向量"""
        return self.beam_direction / self.beam_length
