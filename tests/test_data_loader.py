"""
测试数据加载功能
"""

import pytest
import torch
from pathlib import Path
from pci_torch.config import GENEConfig, BeamConfig
from pci_torch.data_loader import (
    load_gene_config,
    load_beam_config,
    parse_fortran_namelist
)


def test_parse_fortran_namelist():
    """测试Fortran namelist解析"""
    sample_file = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input" / "parameters.dat"
    
    if not sample_file.exists():
        pytest.skip(f"测试文件不存在: {sample_file}")
    
    namelists = parse_fortran_namelist(str(sample_file))
    
    # 检查基本结构
    assert 'box' in namelists
    assert 'geometry' in namelists
    assert 'units' in namelists
    
    # 检查具体值
    assert namelists['box']['nx0'] == 128
    assert namelists['geometry']['q0'] == 2.0
    assert namelists['units']['Bref'] == 2.25


def test_load_beam_config():
    """测试光束配置加载"""
    sample_file = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input" / "LS_condition_JT60SA.txt"
    
    if not sample_file.exists():
        pytest.skip(f"测试文件不存在: {sample_file}")
    
    config = load_beam_config(str(sample_file))
    
    # 检查基本属性
    assert config.injection_point == (5000.0, 0.0, 0.1)
    assert config.detection_point == (5000.0, 0.0, 0.475)
    assert config.width_vertical == 0.15
    assert config.width_toroidal == 0.15
    assert config.div_vertical == 4
    assert config.div_toroidal == 12
    assert config.div_beam == 3000
    
    # 检查计算的属性
    assert config.n_detectors_v == 9
    assert config.n_detectors_t == 25
    assert config.n_beam_points == 3001


def test_gene_config_creation():
    """测试GENE配置对象创建"""
    config = GENEConfig()
    
    # 检查默认值
    assert config.nx0 == 128
    assert config.FVER == 5.0
    assert config.B_ref == 2.25
    
    # 检查导出参数计算
    assert config.rho_ref is not None
    assert config.omega_ref is not None


def test_load_gene_config():
    """测试完整GENE配置加载"""
    sample_file = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input" / "parameters.dat"
    
    if not sample_file.exists():
        pytest.skip(f"测试文件不存在: {sample_file}")
    
    config = load_gene_config(str(sample_file), device='cuda')
    
    # 检查加载的参数
    assert config.nx0 == 128
    assert config.nky0 == 64
    assert config.q0 == 2.0
    assert config.B_ref == 2.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



