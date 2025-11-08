"""
路径配置管理
所有路径通过配置类统一管理，避免硬编码
支持从JSON配置文件读取路径
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathConfig:
    """路径配置类"""
    
    # 输入路径
    input_dir: Path
    output_dir: Path
    
    # 自动推导的路径
    parameters_file: Optional[Path] = None
    equdata_bz_file: Optional[Path] = None
    equdata_be_file: Optional[Path] = None
    beam_config_file: Optional[Path] = None
    torus_data_file: Optional[Path] = None
    
    # 输出子目录
    mat_dir: Optional[Path] = None
    figures_dir: Optional[Path] = None
    fft_dir: Optional[Path] = None
    
    def __post_init__(self):
        """初始化后自动设置路径"""
        # 确保是Path对象
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        
        # 设置输入文件路径（如果未指定）
        if self.parameters_file is None:
            self.parameters_file = self.input_dir / 'parameters.dat'
        else:
            self.parameters_file = Path(self.parameters_file)
            
        if self.equdata_bz_file is None:
            self.equdata_bz_file = self.input_dir / 'equdata_BZ'
        else:
            self.equdata_bz_file = Path(self.equdata_bz_file)
            
        if self.equdata_be_file is None:
            self.equdata_be_file = self.input_dir / 'equdata_be'
        else:
            self.equdata_be_file = Path(self.equdata_be_file)
            
        if self.beam_config_file is None:
            self.beam_config_file = self.input_dir / 'LS_condition_JT60SA.txt'
        else:
            self.beam_config_file = Path(self.beam_config_file)
            
        if self.torus_data_file is None:
            self.torus_data_file = self.input_dir / 'TORUSIons_act.dat'
        else:
            self.torus_data_file = Path(self.torus_data_file)
        
        # 设置输出子目录
        if self.mat_dir is None:
            self.mat_dir = self.output_dir / 'mat'
        else:
            self.mat_dir = Path(self.mat_dir)
            
        if self.figures_dir is None:
            self.figures_dir = self.output_dir / 'figures'
        else:
            self.figures_dir = Path(self.figures_dir)
            
        if self.fft_dir is None:
            self.fft_dir = self.output_dir / 'fft'
        else:
            self.fft_dir = Path(self.fft_dir)
    
    @classmethod
    def from_config_file(cls, config_file: str = None) -> 'PathConfig':
        """从JSON配置文件创建配置
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认路径
            
        Returns:
            PathConfig对象
        """
        if config_file is None:
            # 使用默认配置文件路径
            config_file = Path(__file__).parent.parent / 'config' / 'paths.json'
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 读取JSON配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 提取路径配置
        input_config = config_data.get('input', {})
        output_config = config_data.get('output', {})
        preprocess_config = config_data.get('preprocess', {})
        
        # 构建完整路径
        root_input_dir = input_config.get('root_dir', '')
        root_output_dir = output_config.get('root_dir', '')
        
        # 构建子目录路径（相对于根输出目录）
        test_single_dir = root_output_dir + '/' + output_config.get('test_single_dir', 'test_single')
        figures_dir = root_output_dir + '/' + output_config.get('figures_dir', 'figures')
        mat_dir = root_output_dir + '/' + output_config.get('mat_dir', 'mat')
        fft_dir = root_output_dir + '/' + output_config.get('fft_dir', 'fft')
        
        # 文件路径设为None，让__post_init__处理为相对于input_dir的路径
        return cls(
            input_dir=root_input_dir,
            output_dir=test_single_dir,  # 主输出目录使用test_single_dir
            parameters_file=None,  # 让__post_init__构建完整路径
            equdata_bz_file=None,  # 让__post_init__构建完整路径
            equdata_be_file=None,  # 让__post_init__构建完整路径
            beam_config_file=None,  # 让__post_init__构建完整路径
            torus_data_file=None,  # 让__post_init__构建完整路径
            mat_dir=mat_dir,
            figures_dir=figures_dir,
            fft_dir=fft_dir,
        )
    
    def create_output_dirs(self):
        """创建所有输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mat_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.fft_dir.mkdir(parents=True, exist_ok=True)
    
    def get_integrated_signal_path(self, data_n: int) -> Path:
        """获取IntegratedSignal文件路径"""
        return self.mat_dir / f'IntegratedSignal_{data_n}_overall.mat'
    
    def get_local_cross_section_path(self, data_n: int) -> Path:
        """获取LocalCross-Section文件路径"""
        return self.mat_dir / f'LocalCross-Section_{data_n}_overall.mat'
    
    def get_time_data_file(self, time_step: int) -> Path:
        """获取时间数据文件路径"""
        return self.input_dir / f'TORUSIons_act_{time_step}.dat'
    
    def get_binary_data_file(self, time_step: int) -> Path:
        """获取二进制数据文件路径"""
        return self.input_dir / f'{time_step:08d}.dat'
    
    @classmethod
    def from_env(cls, prefix: str = 'PCI'):
        """从环境变量创建配置
        
        环境变量格式:
            PCI_INPUT_DIR: 输入目录
            PCI_OUTPUT_DIR: 输出目录
            PCI_PARAMETERS_FILE: 参数文件（可选）
            PCI_BEAM_CONFIG_FILE: 光束配置文件（可选）
        """
        input_dir = os.environ.get(f'{prefix}_INPUT_DIR')
        output_dir = os.environ.get(f'{prefix}_OUTPUT_DIR')
        
        if not input_dir or not output_dir:
            raise ValueError(f"必须设置环境变量 {prefix}_INPUT_DIR 和 {prefix}_OUTPUT_DIR")
        
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            parameters_file=os.environ.get(f'{prefix}_PARAMETERS_FILE'),
            beam_config_file=os.environ.get(f'{prefix}_BEAM_CONFIG_FILE'),
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'parameters_file': str(self.parameters_file),
            'equdata_bz_file': str(self.equdata_bz_file),
            'equdata_be_file': str(self.equdata_be_file),
            'beam_config_file': str(self.beam_config_file),
            'torus_data_file': str(self.torus_data_file),
            'mat_dir': str(self.mat_dir),
            'figures_dir': str(self.figures_dir),
            'fft_dir': str(self.fft_dir),
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"""PathConfig:
  Input:  {self.input_dir}
  Output: {self.output_dir}
  - Parameters: {self.parameters_file}
  - Beam Config: {self.beam_config_file}
  - MAT: {self.mat_dir}
  - Figures: {self.figures_dir}
  - FFT: {self.fft_dir}
"""

