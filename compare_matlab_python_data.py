#!/usr/bin/env python
"""
MATLAB vs Python Figure 1 数据对比分析
用于识别导致图像差异的数值差异
"""

import numpy as np
import pandas as pd

def load_matlab_data():
    """加载MATLAB数据（从CSV文件）"""
    print("=== 加载MATLAB数据 ===")
    
    # 加载网格点数据 - 使用正确的分隔符
    matlab_grid = pd.read_csv('/tmp/matlab_grid_points.csv', header=None, sep='\s+')
    matlab_grid.columns = ['X', 'Y', 'Z']
    
    print(f"MATLAB grid points: {len(matlab_grid)}")
    print(f"MATLAB grid shape: {matlab_grid.shape}")
    
    # 从控制台输出手动提取的MATLAB关键数据
    # 根据之前运行MATLAB时的输出
    B2_start = np.array([4.904878, 2.725374, -0.200000])
    B2_end = np.array([-4.904878, -0.200000, 0.200000])
    p1_vector = np.array([-9.809757, -2.925374, 0.400000])
    xl1 = np.array([0.000000, 0.000000, 0.075000])  # 垂直向量1
    xl2 = np.array([0.000000, 0.000000, 0.075000])  # 垂直向量2
    
    return {
        'B2_start': B2_start,
        'B2_end': B2_end,
        'p1': p1_vector,
        'xl1': xl1,
        'xl2': xl2,
        'grid': matlab_grid.values.astype(float)  # 确保数值类型
    }

def load_python_data():
    """加载Python数据"""
    print("\n=== 加载Python数据 ===")
    
    # 加载numpy数据
    py_beam_start = np.load('/tmp/python_beam_start.npy')
    py_beam_end = np.load('/tmp/python_beam_end.npy')
    py_beam_vector = np.load('/tmp/python_beam_vector.npy')
    py_perp_vectors = np.load('/tmp/python_perp_vectors.npy')
    
    # 加载网格点数据 - 移除header
    python_grid = pd.read_csv('/tmp/python_grid_points.csv', header=0, sep=',')
    
    print(f"Python beam start: {py_beam_start}")
    print(f"Python beam end: {py_beam_end}")
    print(f"Python beam vector: {py_beam_vector}")
    print(f"Python perp vectors shape: {py_perp_vectors.shape}")
    print(f"Python grid points: {len(python_grid)}")
    
    return {
        'beam_start': py_beam_start,
        'beam_end': py_beam_end,
        'beam_vector': py_beam_vector,
        'perp_vectors': py_perp_vectors,
        'grid': python_grid.values.astype(float)  # 确保数值类型
    }

def compare_coordinates(matlab_data, python_data):
    """对比坐标数据"""
    print("\n=== 坐标数据对比 ===")
    
    matlab_start = matlab_data['B2_start']
    matlab_end = matlab_data['B2_end']
    
    python_start = python_data['beam_start']
    python_end = python_data['beam_end']
    
    print(f"MATLAB start: [{matlab_start[0]:.6f}, {matlab_start[1]:.6f}, {matlab_start[2]:.6f}]")
    print(f"Python start: [{python_start[0]:.6f}, {python_start[1]:.6f}, {python_start[2]:.6f}]")
    print(f"Start difference: [{matlab_start[0]-python_start[0]:.6f}, {matlab_start[1]-python_start[1]:.6f}, {matlab_start[2]-python_start[2]:.6f}]")
    
    print(f"\nMATLAB end: [{matlab_end[0]:.6f}, {matlab_end[1]:.6f}, {matlab_end[2]:.6f}]")
    print(f"Python end: [{python_end[0]:.6f}, {python_end[1]:.6f}, {python_end[2]:.6f}]")
    print(f"End difference: [{matlab_end[0]-python_end[0]:.6f}, {matlab_end[1]-python_end[1]:.6f}, {matlab_end[2]-python_end[2]:.6f}]")
    
    return {
        'start_diff': matlab_start - python_start,
        'end_diff': matlab_end - python_end
    }

def compare_vectors(matlab_data, python_data):
    """对比向量数据"""
    print("\n=== 向量数据对比 ===")
    
    # MATLAB向量（从控制台输出提取）
    matlab_p1 = matlab_data['p1']
    python_p1 = python_data['beam_vector']
    
    # MATLAB垂直向量（从控制台输出提取）
    matlab_xl1 = matlab_data['xl1']
    matlab_xl2 = matlab_data['xl2']
    python_xl1 = python_data['perp_vectors'][0]
    python_xl2 = python_data['perp_vectors'][1]
    
    print(f"MATLAB p1: [{matlab_p1[0]:.6f}, {matlab_p1[1]:.6f}, {matlab_p1[2]:.6f}]")
    print(f"Python p1: [{python_p1[0]:.6f}, {python_p1[1]:.6f}, {python_p1[2]:.6f}]")
    print(f"p1 difference: [{matlab_p1[0]-python_p1[0]:.6f}, {matlab_p1[1]-python_p1[1]:.6f}, {matlab_p1[2]-python_p1[2]:.6f}]")
    
    print(f"\nMATLAB xl1: [{matlab_xl1[0]:.6f}, {matlab_xl1[1]:.6f}, {matlab_xl1[2]:.6f}]")
    print(f"Python xl1: [{python_xl1[0]:.6f}, {python_xl1[1]:.6f}, {python_xl1[2]:.6f}]")
    print(f"xl1 difference: [{matlab_xl1[0]-python_xl1[0]:.6f}, {matlab_xl1[1]-python_xl1[1]:.6f}, {matlab_xl1[2]-python_xl1[2]:.6f}]")
    
    print(f"\nMATLAB xl2: [{matlab_xl2[0]:.6f}, {matlab_xl2[1]:.6f}, {matlab_xl2[2]:.6f}]")
    print(f"Python xl2: [{python_xl2[0]:.6f}, {python_xl2[1]:.6f}, {python_xl2[2]:.6f}]")
    print(f"xl2 difference: [{matlab_xl2[0]-python_xl2[0]:.6f}, {matlab_xl2[1]-python_xl2[1]:.6f}, {matlab_xl2[2]-python_xl2[2]:.6f}]")
    
    return {
        'p1_diff': matlab_p1 - python_p1,
        'xl1_diff': matlab_xl1 - python_xl1,
        'xl2_diff': matlab_xl2 - python_xl2
    }

def compare_grid_points(matlab_data, python_data):
    """对比网格点数据"""
    print("\n=== 网格点数据对比 ===")
    
    matlab_grid = matlab_data['grid']
    python_grid = python_data['grid']
    
    print(f"MATLAB grid shape: {matlab_grid.shape}")
    print(f"Python grid shape: {python_grid.shape}")
    
    # 对比前5个点
    print(f"\n前5个网格点对比:")
    for i in range(min(5, len(matlab_grid), len(python_grid))):
        mat_point = matlab_grid[i]
        py_point = python_grid[i]
        diff = mat_point - py_point
        print(f"点{i+1}:")
        print(f"  MATLAB: [{mat_point[0]:.6f}, {mat_point[1]:.6f}, {mat_point[2]:.6f}]")
        print(f"  Python: [{py_point[0]:.6f}, {py_point[1]:.6f}, {py_point[2]:.6f}]")
        print(f"  Diff:   [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}]")
    
    # 对比后5个点
    print(f"\n后5个网格点对比:")
    for i in range(max(0, len(matlab_grid)-5), len(matlab_grid)):
        if i < len(python_grid):
            mat_point = matlab_grid[i]
            py_point = python_grid[i]
            diff = mat_point - py_point
            print(f"点{i+1}:")
            print(f"  MATLAB: [{mat_point[0]:.6f}, {mat_point[1]:.6f}, {mat_point[2]:.6f}]")
            print(f"  Python: [{py_point[0]:.6f}, {py_point[1]:.6f}, {py_point[2]:.6f}]")
            print(f"  Diff:   [{diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}]")
    
    # 统计差异
    if len(matlab_grid) == len(python_grid):
        diff = matlab_grid - python_grid
        max_diff = np.max(np.abs(diff), axis=0)
        mean_diff = np.mean(np.abs(diff), axis=0)
        print(f"\n网格点差异统计:")
        print(f"最大差异: X={max_diff[0]:.6f}, Y={max_diff[1]:.6f}, Z={max_diff[2]:.6f}")
        print(f"平均差异: X={mean_diff[0]:.6f}, Y={mean_diff[1]:.6f}, Z={mean_diff[2]:.6f}")
        
        return {
            'grid_shape_match': True,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        }
    else:
        print(f"\n警告: MATLAB和Python网格点数量不同!")
        return {
            'grid_shape_match': False,
            'max_diff': None,
            'mean_diff': None
        }

def analyze_coordinate_system():
    """分析坐标系统问题"""
    print("\n=== 坐标系统分析 ===")
    
    # 从控制台输出提取的MATLAB关键数据
    print("根据MATLAB控制台输出:")
    print("B2_start: [4.904878, 2.725374, -0.200000]")
    print("B2_end:   [-4.904878, -0.200000, 0.200000]")
    print("p1:       [-9.809757, -2.925374, 0.400000]")
    
    # Python数据
    print("\nPython数据:")
    print("B2_start: [4.045085, 2.938926, -0.200000]")
    print("B2_end:   [-4.938442, 0.782172, 0.200000]")
    print("p1:       [-8.983527, -2.156754, 0.400000]")
    
    print("\n=== 问题诊断 ===")
    
    # 分析起点差异
    start_diff = np.array([4.904878, 2.725374, -0.200000]) - np.array([4.045085, 2.938926, -0.200000])
    print(f"起点差异: {start_diff}")
    
    # 分析终点差异
    end_diff = np.array([-4.904878, -0.200000, 0.200000]) - np.array([-4.938442, 0.782172, 0.200000])
    print(f"终点差异: {end_diff}")
    
    print("\n关键发现:")
    print("1. MATLAB中的B2_start和B2_end坐标与Python不同")
    print("2. 这表明坐标转换(R,Z,phi)->(X,Y,Z)存在问题")
    print("3. 需要检查单位转换和角度计算")
    print("4. 最主要问题：网格初始化起点不同！")
    print("   - MATLAB从B2(2,:)开始（检测点）")
    print("   - Python从B2_start开始（注入点）")

def main():
    """主函数"""
    print("MATLAB vs Python Figure 1 数据对比分析")
    print("=" * 50)
    
    try:
        # 加载数据
        matlab_data = load_matlab_data()
        python_data = load_python_data()
        
        # 对比分析
        coord_diff = compare_coordinates(matlab_data, python_data)
        vector_diff = compare_vectors(matlab_data, python_data)
        grid_diff = compare_grid_points(matlab_data, python_data)
        
        # 坐标系统分析
        analyze_coordinate_system()
        
        print("\n" + "=" * 50)
        print("对比分析完成")
        
        # 总结关键问题
        print("\n=== 关键问题总结 ===")
        if np.linalg.norm(coord_diff['start_diff']) > 1e-6:
            print("❌ 起点坐标存在显著差异")
        if np.linalg.norm(coord_diff['end_diff']) > 1e-6:
            print("❌ 终点坐标存在显著差异")
        if np.linalg.norm(vector_diff['p1_diff']) > 1e-6:
            print("❌ 光束方向向量存在显著差异")
        if not grid_diff['grid_shape_match']:
            print("❌ 网格点数量不匹配")
            
        # 最重要的问题
        print("\n=== 根本原因 ===")
        print("🔍 关键问题：网格初始化起点不同！")
        print("   MATLAB: xls=ones(div1_2,div2_2,divls_2)*B2(2,1)  # 从检测点开始")
        print("   Python: xls=...*B2_start  # 从注入点开始")
        print("   这导致整个网格的起点完全不同！")
            
    except Exception as e:
        print(f"对比分析出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
