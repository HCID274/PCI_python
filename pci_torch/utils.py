"""
辅助工具函数
"""

import torch
import numpy as np
from typing import Union, Tuple


def to_tensor(
    data: Union[np.ndarray, torch.Tensor],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """将数据转换为PyTorch张量"""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    
    # 处理字节序问题
    if hasattr(data, 'dtype') and data.dtype.byteorder not in ('=', '|'):
        # 数据不是本机字节序，需要转换
        data = data.astype(data.dtype.newbyteorder('='))
    
    return torch.from_numpy(data).to(device=device, dtype=dtype)


def ensure_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    确保张量有batch维度
    
    返回:
        (张量, 是否添加了batch维度)
    """
    if x.ndim == 3:
        return x.unsqueeze(0), True
    return x, False


def remove_batch_dim(x: torch.Tensor, was_added: bool) -> torch.Tensor:
    """如果batch维度是添加的，则移除它"""
    if was_added:
        return x.squeeze(0)
    return x


def normalize_coordinates(
    coords: torch.Tensor,
    min_vals: torch.Tensor,
    max_vals: torch.Tensor
) -> torch.Tensor:
    """
    将坐标归一化到[-1, 1]范围（用于grid_sample）
    
    Args:
        coords: (..., 3) 坐标张量
        min_vals: (3,) 最小值
        max_vals: (3,) 最大值
    
    Returns:
        归一化后的坐标
    """
    return 2.0 * (coords - min_vals) / (max_vals - min_vals) - 1.0


def print_tensor_info(name: str, tensor: torch.Tensor):
    """打印张量的详细信息（用于调试）"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min/Max: {tensor.min().item():.6f} / {tensor.max().item():.6f}")
    print(f"  Mean/Std: {tensor.mean().item():.6f} / {tensor.std().item():.6f}")


def bisec(value: Union[torch.Tensor, float], array: torch.Tensor) -> Tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]:
    """
    二分查找 - 完全按照MATLAB的com/bisec.m实现
    
    MATLAB bisec.m:
    function yc=bisec(xx,data)
        [m1,n1]=size(data);
        m=max(m1,n1);
        if (data(1)<data(m))
            ya=1;
            yb=m;
        else
            ya=m;
            yb=1;
        end
        for k=1:40
            yt=round((ya+yb)/2.0);
            ymid=data(yt);
            if (ymid<=xx)
                ya=yt;
            else
                yb=yt;
            end
            if (abs(ya-yb)<=1)
                yc=[ya; yb];
                return 
            end
        end
        yc=[0;0];
        return
    end
    """
    # 获取数组大小 - 完全对应MATLAB第2-3行
    if array.ndim == 1:
        # 1D数组
        m1 = len(array)
        n1 = 1
    else:
        # 2D数组，展平处理以匹配MATLAB的max(m1,n1)逻辑
        m1, n1 = array.shape
    
    m = max(m1, n1)
    
    # 确定搜索方向 - 完全对应MATLAB第5-10行
    if array[0].item() < array[-1].item():
        # 升序: ya=1, yb=m
        ya = 1  # 保持MATLAB的1-based索引
        yb = m
    else:
        # 降序: ya=m, yb=1
        ya = m  # 保持MATLAB的1-based索引
        yb = 1
    
    # 获取查找值
    if isinstance(value, (int, float)):
        xx = value
    else:
        xx = value.item()
    
    # 二分查找循环 - 完全对应MATLAB第18-30行
    for k in range(1, 41):  # MATLAB是1:40
        # MATLAB第19行：yt=round((ya+yb)/2.0)
        yt = round((ya + yb) / 2.0)
        
        # MATLAB第20行：ymid=data(yt)，转换为0-based索引
        ymid = array[yt - 1].item()
        
        # MATLAB第21行：if (ymid<=xx)
        if ymid <= xx:
            ya = yt
        else:
            yb = yt
        
        # MATLAB第26行：if (abs(ya-yb)<=1)
        if abs(ya - yb) <= 1:
            # MATLAB第27行：yc=[ya; yb]
            # 转换为Python的0-based索引，保持MATLAB的原始顺序
            
            if ya == 0 and yb == 0:
                # MATLAB返回[0;0]的情况
                lower_idx = 0
                upper_idx = 0
            else:
                # 正常情况：直接转换索引，不改变相对顺序
                lower_idx = max(0, ya - 1)
                upper_idx = max(0, yb - 1)
                
                # 注意：MATLAB bisec.m的降序情况下，ya和yb的顺序与升序不同
                # 升序：ya <= yb，但降序：ya >= yb
                # Python实现应该保持这种差异，不要强制排序
            
            # 如果输入是标量，返回标量索引
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return lower_idx, upper_idx
            
            return lower_idx, upper_idx
    
    # MATLAB第31行：yc=[0;0]
    # 如果循环结束还没返回，返回错误标志
    lower_idx = 0
    upper_idx = 0
    
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return lower_idx, upper_idx
    
    return lower_idx, upper_idx


def bisec_batch(values: torch.Tensor, array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量二分查找 - 完全对应MATLAB的bisec.m
    
    通过逐个调用修正后的bisec()函数，确保与MATLAB bisec.m完全一致。
    虽然性能可能不如torch.searchsorted，但保证100%的MATLAB兼容性。
    
    Args:
        values: 要查找的值 (N,) 或任意shape
        array: 有序数组 (1D tensor)，可以是升序或降序
    
    Returns:
        (lower_idx, upper_idx): 形状与values相同的索引张量
        对于升序数组: array[lower_idx] <= values < array[upper_idx]
        对于降序数组: array[lower_idx] >= values >= array[upper_idx]
    """
    original_shape = values.shape
    values_flat = values.flatten()
    N = len(values_flat)
    
    # 确保array是1D张量
    if array.ndim > 1:
        raise ValueError(f"array must be 1D, got {array.ndim}D")
    
    if array.numel() == 0:
        # 如果array为空，返回默认索引
        lower_idx = torch.zeros(N, dtype=torch.long, device=values.device)
        upper_idx = torch.zeros(N, dtype=torch.long, device=values.device)
        return lower_idx.reshape(original_shape), upper_idx.reshape(original_shape)
    
    # 创建结果张量
    lower_idx = torch.zeros(N, dtype=torch.long, device=values.device)
    upper_idx = torch.zeros(N, dtype=torch.long, device=values.device)
    
    # 逐个处理每个查找值，确保与标量bisec()完全一致
    for i in range(N):
        value = values_flat[i]
        try:
            # 调用修正后的bisec函数
            l_idx, u_idx = bisec(value, array)
            lower_idx[i] = l_idx
            upper_idx[i] = u_idx
        except Exception as e:
            # 发生错误时，返回安全的默认值
            lower_idx[i] = 0
            upper_idx[i] = 0
    
    # Reshape回原始形状
    lower_idx = lower_idx.reshape(original_shape)
    upper_idx = upper_idx.reshape(original_shape)
    
    return lower_idx, upper_idx

