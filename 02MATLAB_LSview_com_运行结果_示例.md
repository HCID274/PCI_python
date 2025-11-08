# MATLAB LSview_com 运行结果详细记录

**文档目标**: 作为Python实现的蓝图，确保所有MATLAB生成的图表都在Python中正确复现
**基于源码**: `/work/DTMP/lhqing/PCI/code/TDS_class/plot/LSview_com.m`
**生成时间**: 2025年11月8日

---

## 1. 主要核心图表 (Core Figures)

### 1.1 Figure 1: 3D光束几何图
**文件**: `LSview_com.m` (第136-153行)
**功能**: 显示光束的3D几何形状和托卡马克边界

**包含元素**:
- **光束起点和终点**: 红色圆点标记
  ```matlab
  plot3(B2(:,1),B2(:,2),B2(:,3),'o');
  ```
- **光束传播路径**: 蓝色点线显示采样轨迹
  ```matlab
  plot3(xls1,yls1,zls1,'.');
  ```
- **托卡马克边界**: 黑色线条显示托卡马克轮廓
  ```matlab
  xls_b=dataC.GRC(end,:).'*cos([0:30]*2*pi/30);
  yls_b=dataC.GRC(end,:).'*sin([0:30]*2*pi/30);
  zls_b=repmat(dataC.GZC(end,:).',1,31);
  plot3(xls_b(1:2:end,:).',yls_b(1:2:end,:).',zls_b(1:2:end,:).','k-');
  ```
- **坐标轴标签**: "X", "Y", "Z"

**注释代码部分** (第161-170行):
- 光束中心线3D图
- 方向向量箭头图 (`quiver3`)

### 1.2 Figure 2: 多用途图表
**出现的文件**: 
- `LSview_com.m` 中未直接出现
- `probe_multi2.m` (第140-141行): PCI信号强度图
- `probe_multi3.m` (第100-101行): PCI信号强度图
- 其他模块中用于能量、轨迹等

**在probe_multi系列中**:
```matlab
figure(2)
plot(abs(pout));  % 显示PCI信号的绝对值
```
**功能**: 显示PCI信号沿光束路径的强度变化

### 1.3 Figure 3: 检测器信号等高线图 ⭐
**文件**: `LSview_com.m` (第215-225行)
**功能**: 显示检测器阵列上的PCI信号分布

**核心代码**:
```matlab
[xx1,yy1]=meshgrid(wid1/2*[-div1:div1]/div1,-wid2/2*[-div2:div2]/div2);
xx1 = fliplr(xx1);  % 水平翻转

figure(3)
contourf(yy1.',xx1.',pout2,100,'LineStyle','none');
shading flat;
axis equal;
colorbar
xlabel('x (m)');
ylabel('y (m)');
```
**网格设置**:
- `xx1`: 水平轴网格 (环向检测器位置)
- `yy1`: 垂直轴网格 (垂直检测器位置)
- 100层等高线填充图

**关键数据**:
- `pout2`: 积分后的检测器信号 (2D数组)
- `wid1`: 垂直宽度参数
- `wid2`: 环向宽度参数
- `div1, div2`: 网格细分参数

### 1.4 Figure 4: 2D波数空间图 ⭐
**文件**: `LSview_com.m` (第228-229行)
**调用函数**: `plotWaveNumberSpace(yy1.',xx1.',pout2,dataC)`

**函数文件**: `/work/DTMP/lhqing/PCI/code/TDS_class/plot/plotWaveNumberSpace.m`

**核心算法**:
```matlab
% 1. 数据预处理
Mean = mean(mean(realSpaceData(:)));
realSpaceData = realSpaceData - Mean;

% 2. 2D FFT
P = fft2(realSpaceData);
P_shifted = fftshift(P);
Amp = abs(P_shifted);

% 3. 频率坐标计算
dx = (xx(1,2) - xx(1,1));
dy = (yy(2,1) - yy(1,1));
kx = 2*pi*(-Nx/2:Nx/2-1)/((Nx-1)*dx);
ky = 2*pi*(-Ny/2:Ny/2-1)/((Ny-1)*dy);

% 4. 归一化
if obj.FVER == 5 %GENE
    kx = kx*obj.rho_ref;
    ky = ky*obj.rho_ref;
else
    kx = kx*0.003;
    ky = ky*0.003;
end

% 5. 绘制等高线
contourf(KX, KY, log(Amp), 100, 'LineStyle', 'none');
xlabel('k_xρ_i')
ylabel('k_yρ_i')
colorbar;
```
**输出**: kx-ky空间的频谱功率分布

### 1.5 Figure 5: 光束位置截面图
**文件**: `/work/DTMP/lhqing/PCI/code/TDS_class/plot/LS_location.m` (第10-19行)
**功能**: 在R-Z截面显示光束路径

**核心代码**:
```matlab
figure(5)
plot(obj.GRC(end,:),zls_b,'k-');  % 托卡马克边界
axis equal;
hold on

for i = 1:div3
    plot(R((div1+1)/2,:,i),Z((div1+1)/2,:,i),'r.')  % 光束中心线
    title('beam path')
end
hold off
```
**显示内容**:
- 托卡马克边界轮廓
- 光束在R-Z平面的投影轨迹

---

## 2. 磁场分析图表 (Magnetic Field Analysis)

### 2.1 Figures 201-203: LSmag磁场分量图
**文件**: `/work/DTMP/lhqing/PCI/code/TDS_class/sim_data/task_eq/@task_eqClass/LSmag.m` (第31-38行)
**功能**: 显示沿光束路径的磁场分量

**图201 - Y分量**:
```matlab
figure(201)
plot(yy,'.')  % By分量
```

**图202 - X分量**:
```matlab
figure(202)
plot(xx,'.')  % Bx分量
```

**图203 - 角度**:
```matlab
figure(203)
plot(ang,'.')  % 磁场角度
```
**磁场计算**:
```matlab
% 光束方向向量分解
Lx = [-p1(2)/p1(1); 1; 0]/((-p1(2)/p1(1))^2+1)^(1/2)
Ly = [1; p1(2)/p1(1); 0]/((p1(2)/p1(1))^2+1)^(1/2)
Lz = p1/(p1(1)^2+p1(2)^2+p1(3)^2)^(1/2);

for a=1:num1
    BB(1)=BR*cos(PHI0)-BT*sin(PHI0);
    BB(2)=BR*sin(PHI0)+BT*cos(PHI0);
    BB(3)=BZ;
    
    xx(end-a+1) = dot(BB, Lx);  % X分量投影
    yy(end-a+1) = dot(BB, Ly);  % Y分量投影
    ang(end-a+1) = atan2(yy(end-a+1),xx(end-a+1))/pi*180;  % 角度
end
```

---

## 3. 局部性分析图表 (Localization Analysis)

### 3.1 Figure 21: ρ值比较图 (probe_multi2)
**文件**: `/work/DTMP/lhqing/PCI/code/TDS_class/sim_data/GENE/plot/probe_multi2.m` (第143-153行)
**功能**: 比较有/无涨落情况下的径向位置

**代码**:
```matlab
figure(21)
plot(loc,".b")      % 有涨落情况
hold on
plot(nonloc,".r")   % 无涨落情况
hold off
ylim([0,1])
xlabel('beam path')
ylabel('ρ')
minmax = sprintf('ρ_{min} = %.2f , ρ_{max} = %.2f',...
    min(loc, [], 'omitnan'), max(loc, [], 'omitnan'));
title(minmax);
legend("With fluctuation","Without fluctuation",'Location', 'Best')
```

### 3.2 Figure 22: ρ值比较图 (probe_multi3)
**文件**: `/work/DTMP/lhqing/PCI/code/TDS_class/sim_data/GENE/plot/probe_multi3.m` (第103-113行)
**功能**: 与Figure 21相同，但用于多模态分析

**代码结构与Figure 21相同**

---

## 4. 磁场分量详细分析 (Detailed Field Components)

### 4.1 Figures 301-304: 磁场分量对比图
**文件**: `probe_multi2.m` (第155-187行)

**Figure 301 - By分量对比**:
```matlab
figure(301)
plot(yy,".b")           % 有涨落
hold on
plot(non_yy,".r")       % 无涨落
hold off 
xlabel('beam path')
ylabel('By')
legend("With fluctuation","Without fluctuation",'Location', 'Best')
```

**Figure 302 - Bx分量对比**:
```matlab
figure(302)
plot(xx,".b")
hold on
plot(non_xx,".r")
hold off 
xlabel('beam path')
ylabel('Bx')
legend("With fluctuation","Without fluctuation",'Location', 'Best')
```

**Figure 303 - θ角度对比**:
```matlab
figure(303)
plot(ang,".b")
hold on
plot(non_ang,".r")
hold off 
xlabel('beam path')
ylabel('theta')
legend("With fluctuation","Without fluctuation",'Location', 'Best')
```

**Figure 304 - θ2角度对比**:
```matlab
figure(304)
plot(ang2,".b")
hold on
plot(non_ang2,".r")
hold off 
xlabel('beam path')
ylabel('theta2')
legend("With fluctuation","Without fluctuation",'Location', 'Best')
```

**磁场分量数据来源**:
```matlab
[xx, yy, ang, ang2] = LSmag2(dataC,x1(a,1),x1(a,2),x1(a,3),Lx,Ly,Lz);
```
其中 `LSmag2` 函数计算：
- `xx`: Bx分量投影
- `yy`: By分量投影  
- `ang`: 第一角度分量
- `ang2`: 第二角度分量

---

## 5. 图表生成调用链 (Generation Call Chain)

### 5.1 主要调用流程
```
LSview_com.m
├── Figure 1: 3D光束几何
├── Figure 3: 检测器信号等高线
├── Figure 4: 2D波数空间 (调用plotWaveNumberSpace.m)
└── LSmag() → Figures 201-203: 磁场分量
    └── probe_multi2/multi3() → Figures 21, 301-304: 局部性分析
```

### 5.2 条件性图表生成
- **Figure 2**: 在`probe_multi`系列函数中生成，显示信号强度
- **Figures 21-22, 301-304**: 仅在多模态分析中生成
- **Figure 5**: 在`LS_location`函数中独立生成

---

## 6. 数据流和依赖关系 (Data Flow & Dependencies)

### 6.1 核心数据结构
```matlab
% 光束几何参数
wid1, wid2          % 宽度参数
div1, div2, divls   % 网格细分参数
B2                  % 起点终点坐标 [2x3]

% 信号数据
pout1              % 3D光束积分信号 [div1_2 x div2_2 x divls_2]
pout2              % 2D检测器信号 [div1_2 x div2_2]

% 网格坐标
xx1, yy1           % 检测器位置网格
```

### 6.2 函数调用依赖
- `LSview_com` → `LSmag` → `probeEQ_mag`
- `LSview_com` → `probe_multi2` → `probeEQ_local_s`
- `LSview_com` → `plotWaveNumberSpace`

---

## 7. Python实现对照表 (Python Implementation Mapping)

| MATLAB Figure | Python模块/函数 | 对应文件 | 状态 |
|---------------|----------------|----------|------|
| Figure 1 | `PCIVisualizer.plot_beam_geometry_3d()` | `pci_torch/visualization.py` | ✅ 已实现 |
| Figure 2 | `forward_projection()` 信号强度 | `pci_torch/forward_model.py` | ⚠️ 需要添加图表 |
| Figure 3 | `PCIVisualizer.plot_detector_contour()` | `pci_torch/visualization.py` | ✅ 已实现 |
| Figure 4 | `plot_wavenumber_space_2d()` | `pci_torch/visualization.py` | ✅ 已实现 |
| Figure 5 | `PCIVisualizer.plot_beam_path_2d()` | 需要添加 | ❌ 未实现 |
| Figures 201-203 | `PCIVisualizer.plot_magnetic_field()` | 需要添加 | ❌ 未实现 |
| Figure 21/22 | `PCIVisualizer.plot_localization_comparison()` | 需要添加 | ❌ 未实现 |
| Figures 301-304 | `PCIVisualizer.plot_field_components()` | 需要添加 | ❌ 未实现 |

---

## 8. 关键修正记录 (Critical Fixes)

### 8.1 已验证的修正
- **光束方向向量**: `p1 = B2_start - B2_end` (从终点指向起点) ✅
- **phi角度转换**: `phi * 2 * π` ✅
- **GAC数据处理**: 使用线性查找替代bisec ✅
- **边界检查**: 使用实际最小最大值 ✅

### 8.2 需要注意的问题
- **索引转换**: MATLAB 1-based → Python 0-based
- **数组维度**: MATLAB列优先 → Python行优先
- **复数处理**: 确保实部/虚部正确对应

---

## 9. 下一步Python实现计划 (Next Steps for Python)

### 9.1 立即需要添加的图表
1. **Figure 2**: 信号强度沿光束路径的线图
2. **Figure 5**: 光束R-Z截面位置图
3. **Figures 201-203**: 磁场分量分析图
4. **Figure 21/22**: 局部性分析对比图
5. **Figures 301-304**: 详细磁场分量对比图

### 9.2 实现优先级
1. **高优先级**: Figure 2, Figure 5 (核心可视化)
2. **中优先级**: Figures 201-203 (磁场分析)
3. **低优先级**: 局部性分析图表 (高级功能)

---

**文档完成日期**: 2025年11月8日  
**最后更新**: 根据最新MATLAB源码分析  
**状态**: 等待Python实现补充
