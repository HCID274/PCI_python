"""
测试坐标转换功能
"""

import pytest
import torch
import numpy as np
from pci_torch.coordinates import (
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    cartesian_to_flux,
    batch_coordinate_transform
)
from pci_torch.config import GENEConfig


def test_cartesian_to_cylindrical():
    """测试笛卡尔到柱坐标转换"""
    device = 'cpu'
    
    # 测试点: (1, 0, 0) -> (R=1, Z=0, phi=0)
    x = torch.tensor([1.0], device=device)
    y = torch.tensor([0.0], device=device)
    z = torch.tensor([0.0], device=device)
    
    R, Z, phi = cartesian_to_cylindrical(x, y, z)
    
    assert torch.allclose(R, torch.tensor([1.0], device=device), atol=1e-6)
    assert torch.allclose(Z, torch.tensor([0.0], device=device), atol=1e-6)
    assert torch.allclose(phi, torch.tensor([0.0], device=device), atol=1e-6)
    
    # 测试点: (0, 1, 0) -> (R=1, Z=0, phi=π/2)
    x = torch.tensor([0.0], device=device)
    y = torch.tensor([1.0], device=device)
    z = torch.tensor([0.0], device=device)
    
    R, Z, phi = cartesian_to_cylindrical(x, y, z)
    
    assert torch.allclose(R, torch.tensor([1.0], device=device), atol=1e-6)
    assert torch.allclose(phi, torch.tensor([np.pi/2], device=device), atol=1e-6)


def test_cylindrical_to_cartesian():
    """测试柱到笛卡尔坐标转换"""
    device = 'cpu'
    
    # 测试点: (R=1, Z=0, phi=0) -> (1, 0, 0)
    R = torch.tensor([1.0], device=device)
    Z = torch.tensor([0.0], device=device)
    phi = torch.tensor([0.0], device=device)
    
    x, y, z = cylindrical_to_cartesian(R, Z, phi)
    
    assert torch.allclose(x, torch.tensor([1.0], device=device), atol=1e-6)
    assert torch.allclose(y, torch.tensor([0.0], device=device), atol=1e-6)
    assert torch.allclose(z, torch.tensor([0.0], device=device), atol=1e-6)


def test_coordinate_roundtrip():
    """测试坐标转换往返"""
    device = 'cpu'
    
    # 原始笛卡尔坐标
    x_orig = torch.tensor([1.0, 0.0, -1.0, 0.5], device=device)
    y_orig = torch.tensor([0.0, 1.0, 0.0, 0.5], device=device)
    z_orig = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    
    # 笛卡尔 -> 柱 -> 笛卡尔
    R, Z, phi = cartesian_to_cylindrical(x_orig, y_orig, z_orig)
    x_back, y_back, z_back = cylindrical_to_cartesian(R, Z, phi)
    
    assert torch.allclose(x_back, x_orig, atol=1e-6)
    assert torch.allclose(y_back, y_orig, atol=1e-6)
    assert torch.allclose(z_back, z_orig, atol=1e-6)


def test_batch_coordinate_transform():
    """测试批量坐标转换"""
    device = 'cpu'
    
    # 批量点
    xyz = torch.rand(10, 3, device=device)
    
    config = GENEConfig()
    
    # 转换到柱坐标
    cyl = batch_coordinate_transform(xyz, config, to_system='cylindrical')
    
    assert cyl.shape == (10, 3)
    # R应该是正的
    assert torch.all(cyl[:, 0] >= 0)
    # phi应该在[0, 2π)
    assert torch.all(cyl[:, 2] >= 0)
    assert torch.all(cyl[:, 2] < 2*np.pi)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



