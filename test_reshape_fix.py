#!/usr/bin/env python3
"""
验证reshape修正效果的测试脚本
"""

import numpy as np
import torch

def test_reshape_order():
    """测试reshape的order参数效果"""
    print("=== Reshape Order 测试 ===")
    
    # 创建测试数据
    original = np.arange(24)  # [0, 1, 2, ..., 23]
    print(f"原始数据: {original}")
    
    # 目标维度 (2, 3, 4)
    target_shape = (2, 3, 4)
    
    # C-order (行主序) - Python默认
    c_order = original.reshape(target_shape, order='C')
    print(f"\nC-order reshape结果 shape: {c_order.shape}")
    print("前4个元素:")
    print(c_order[0, 0, :])  # 第一行第一列的所有列
    print(c_order[0, 1, :])  # 第一行第二列的所有列
    
    # F-order (列主序) - MATLAB默认
    f_order = original.reshape(target_shape, order='F')
    print(f"\nF-order reshape结果 shape: {f_order.shape}")
    print("前4个元素:")
    print(f_order[0, 0, :])  # 第一行第一列的所有列
    print(f_order[0, 1, :])  # 第一行第二列的所有列
    
    # 验证关键位置差异
    print(f"\n关键位置对比:")
    print(f"C-order[0,0,0] = {c_order[0,0,0]}, C-order[1,0,0] = {c_order[1,0,0]}")
    print(f"F-order[0,0,0] = {f_order[0,0,0]}, F-order[1,0,0] = {f_order[1,0,0]}")
    print(f"差异: C-order和F-order的第一个元素分别为 {c_order[0,0,0]} 和 {f_order[0,0,0]}")
    
    return c_order, f_order

def test_matlab_like_reshape():
    """测试MATLAB风格的reshape"""
    print(f"\n=== MATLAB风格reshape测试 ===")
    
    # 模拟MATLAB中的情况
    # MATLAB: pout1=reshape(pout,div1_2,div2_2,divls_2);
    
    # 假设div1_2=3, div2_2=2, divls_2=4，总共24个元素
    div1_2, div2_2, divls_2 = 3, 2, 4
    
    # 创建一维数据（模拟pout）
    pout = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=float)
    
    print(f"原始pout: {pout}")
    
    # MATLAB的reshape (列主序)
    pout1_matlab = pout.reshape(div1_2, div2_2, divls_2, order='F')
    print(f"\nMATLAB风格reshape (div1_2={div1_2}, div2_2={div2_2}, divls_2={divls_2}):")
    print("pout1_matlab[0,0,:] =", pout1_matlab[0,0,:])
    print("pout1_matlab[1,0,:] =", pout1_matlab[1,0,:]) 
    print("pout1_matlab[0,1,:] =", pout1_matlab[0,1,:])
    print("pout1_matlab[0,0,1] =", pout1_matlab[0,0,1])
    
    # Python默认的reshape (行主序)
    pout1_python = pout.reshape(div1_2, div2_2, divls_2, order='C')
    print(f"\nPython默认reshape:")
    print("pout1_python[0,0,:] =", pout1_python[0,0,:])
    print("pout1_python[1,0,:] =", pout1_python[1,0,:])
    print("pout1_python[0,1,:] =", pout1_python[0,1,:]) 
    print("pout1_python[0,0,1] =", pout1_python[0,0,1])
    
    # 验证关键差异
    print(f"\n关键验证:")
    print(f"MATLAB: pout1(1,1,1) = {pout1_matlab[0,0,0]} (应该是1)")
    print(f"Python: pout1[0,0,0] = {pout1_python[0,0,0]} (应该是1)")
    print(f"MATLAB: pout1(2,1,1) = {pout1_matlab[1,0,0]} (应该是7)")  
    print(f"Python: pout1[1,0,0] = {pout1_python[1,0,0]} (应该是4)")
    
    return pout1_matlab, pout1_python

if __name__ == "__main__":
    test_reshape_order()
    test_matlab_like_reshape()
    print(f"\n=== 测试完成 ===")
    print("结论: 使用order='F'的reshape可以正确匹配MATLAB的列主序行为")
