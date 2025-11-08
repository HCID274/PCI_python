"""
PyPCI - GPU加速的PCI正向模拟系统

基于PyTorch，针对AMD ROCm优化
"""

__version__ = "0.1.0"

from .config import GENEConfig, BeamConfig
from .path_config import PathConfig
from .data_loader import (
    load_gene_config,
    load_beam_config,
    load_gene_data,
    load_equilibrium_data,
    load_gene_config_from_parameters,
    parse_parameters_dat,
    separate_torusdata,
    generate_timedata,
    fread_data_s,
)
from .forward_model import forward_projection
from .batch_processor import process_time_series, process_single_timepoint
from .fft_analysis import FFT3DAnalyzer
from .visualization import PCIVisualizer

__all__ = [
    "GENEConfig",
    "BeamConfig",
    "PathConfig",
    "load_gene_config",
    "load_gene_config_from_parameters",
    "load_beam_config",
    "load_gene_data",
    "load_equilibrium_data",
    "parse_parameters_dat",
    "separate_torusdata",
    "generate_timedata",
    "fread_data_s",
    "forward_projection",
    "process_time_series",
    "process_single_timepoint",
    "FFT3DAnalyzer",
    "PCIVisualizer",
]

