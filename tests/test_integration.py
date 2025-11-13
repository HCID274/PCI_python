"""
端到端集成测试
"""

import pytest
import torch
from pathlib import Path
from pci_torch import (
    GENEConfig,
    BeamConfig,
    load_gene_config,
    load_beam_config,
    forward_projection
)


@pytest.fixture
def sample_config():
    """创建测试用配置"""
    config = GENEConfig()
    config.nx0 = 64  # 减小尺寸以加快测试
    config.nz0 = 32
    # 设置简化的等离子体参数
    config.PA = torch.tensor([config.major_R * config.L_ref, 0.0])
    config.GAC = torch.ones(config.nx0+1, 400+1) * config.trpeps * config.major_R * config.L_ref
    return config


@pytest.fixture
def sample_beam_config():
    """创建测试用光束配置"""
    return BeamConfig(
        injection_point=(5000.0, 0.0, 0.1),
        detection_point=(5000.0, 0.0, 0.475),
        width_vertical=0.15,
        width_toroidal=0.15,
        div_vertical=2,  # 减小以加快测试
        div_toroidal=4,
        div_beam=100
    )


def test_forward_projection_shape(sample_config, sample_beam_config):
    """测试正向投影输出形状"""
    device = 'cuda'
    
    # 创建测试密度场
    density_3d = torch.randn(400, sample_config.nx0, sample_config.nz0, device=device)
    
    # 执行正向投影
    pci_image = forward_projection(
        density_3d,
        sample_config,
        sample_beam_config,
        device=device
    )
    
    # 检查输出形状
    expected_shape = (
        sample_beam_config.n_detectors_v,
        sample_beam_config.n_detectors_t
    )
    assert pci_image.shape == expected_shape


def test_forward_projection_batch(sample_config, sample_beam_config):
    """测试批处理模式"""
    device = 'cuda'
    batch_size = 4
    
    # 创建批量密度场
    density_3d = torch.randn(
        batch_size, 400, sample_config.nx0, sample_config.nz0, 
        device=device
    )
    
    # 执行正向投影
    pci_images = forward_projection(
        density_3d,
        sample_config,
        sample_beam_config,
        device=device
    )
    
    # 检查输出形状
    expected_shape = (
        batch_size,
        sample_beam_config.n_detectors_v,
        sample_beam_config.n_detectors_t
    )
    assert pci_images.shape == expected_shape


def test_forward_projection_differentiable(sample_config, sample_beam_config):
    """测试可微分性"""
    device = 'cuda'
    
    # 创建需要梯度的密度场
    density_3d = torch.randn(
        400, sample_config.nx0, sample_config.nz0, 
        device=device,
        requires_grad=True
    )
    
    # 执行正向投影
    pci_image = forward_projection(
        density_3d,
        sample_config,
        sample_beam_config,
        device=device
    )
    
    # 计算损失并反向传播
    loss = pci_image.sum()
    loss.backward()
    
    # 检查梯度已计算
    assert density_3d.grad is not None
    assert density_3d.grad.shape == density_3d.shape


def test_forward_projection_consistency(sample_config, sample_beam_config):
    """测试输出一致性"""
    device = 'cuda'
    
    # 创建相同的密度场
    torch.manual_seed(42)
    density_3d = torch.randn(400, sample_config.nx0, sample_config.nz0, device=device)
    
    # 多次执行应该得到相同结果
    result1 = forward_projection(density_3d, sample_config, sample_beam_config, device=device)
    result2 = forward_projection(density_3d, sample_config, sample_beam_config, device=device)
    
    assert torch.allclose(result1, result2, atol=1e-6)


def test_forward_projection_linearity(sample_config, sample_beam_config):
    """测试线性性质"""
    device = 'cuda'
    
    # 创建两个密度场
    density1 = torch.randn(400, sample_config.nx0, sample_config.nz0, device=device)
    density2 = torch.randn(400, sample_config.nx0, sample_config.nz0, device=device)
    
    # 分别投影
    pci1 = forward_projection(density1, sample_config, sample_beam_config, device=device)
    pci2 = forward_projection(density2, sample_config, sample_beam_config, device=device)
    
    # 和的投影
    pci_sum = forward_projection(density1 + density2, sample_config, sample_beam_config, device=device)
    
    # 检查线性性质（允许数值误差）
    assert torch.allclose(pci_sum, pci1 + pci2, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要GPU")
def test_forward_projection_gpu(sample_config, sample_beam_config):
    """测试GPU执行"""
    device = 'cuda'
    
    # 将配置移到GPU
    sample_config.PA = sample_config.PA.to(device)
    sample_config.GAC = sample_config.GAC.to(device)
    
    # 创建GPU上的密度场
    density_3d = torch.randn(400, sample_config.nx0, sample_config.nz0, device=device)
    
    # 执行正向投影
    pci_image = forward_projection(
        density_3d,
        sample_config,
        sample_beam_config,
        device=device
    )
    
    # 检查结果在GPU上
    assert pci_image.device.type == 'cuda'
    
    # 检查形状
    expected_shape = (
        sample_beam_config.n_detectors_v,
        sample_beam_config.n_detectors_t
    )
    assert pci_image.shape == expected_shape


def test_full_pipeline_with_real_files():
    """使用实际文件的完整流程测试"""
    base_path = Path(__file__).parent.parent.parent / "TDS_class" / "sample" / "input"
    
    parameters_file = base_path / "parameters.dat"
    ls_condition_file = base_path / "LS_condition_JT60SA.txt"
    
    if not parameters_file.exists() or not ls_condition_file.exists():
        pytest.skip("测试文件不存在")
    
    device = 'cuda'
    
    # 加载配置
    config = load_gene_config(str(parameters_file), device=device)
    beam_config = load_beam_config(str(ls_condition_file))
    
    # 简化配置
    config.nx0 = 64
    config.nz0 = 32
    if config.PA is None:
        config.PA = torch.tensor([config.major_R * config.L_ref, 0.0], device=device)
        config.GAC = torch.ones(config.nx0+1, 400+1, device=device) * config.trpeps * config.major_R * config.L_ref
    
    # 简化光束配置
    beam_config.div_vertical = 2
    beam_config.div_toroidal = 4
    beam_config.div_beam = 100
    beam_config.__post_init__()
    
    # 创建测试数据
    density_3d = torch.randn(400, config.nx0, config.nz0, device=device)
    
    # 执行正向投影
    pci_image = forward_projection(
        density_3d,
        config,
        beam_config,
        device=device
    )
    
    # 基本检查
    assert pci_image.shape == (beam_config.n_detectors_v, beam_config.n_detectors_t)
    assert not torch.isnan(pci_image).any()
    assert not torch.isinf(pci_image).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])



