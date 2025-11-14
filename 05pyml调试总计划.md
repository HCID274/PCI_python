## 一、当前进度 & 核心思路

### 已经确认无误的部分

* **阶段 1：参数解析 & 路径配置**
  入口参数、路径、字节序等已经跑通，用不上再折腾。

* **阶段 2：GENE 配置 & Equilibrium 数据加载** ✅

  * `parameters.dat` → 所有标量（`nx0, q0, shat, trpeps, B_ref, rho_ref, inside, outside ...`）
  * `equdata_BZ` → `GRC/GZC/GFC/GAC/GTC_f/GTC_c` 形状完全一致，差值在 1e-15 级别
  * `equdata_be` → 所有磁场数组、网格参数完全一致
    👉 这一块可以视为“黄金标准已复刻”。

### 之后所有问题，必然在：

1. **密度数据链路**：文本→二进制→3D reshape→padding→重排
2. **光束几何 + 插值 + 线积分**：beam_grid / 坐标转换 / 三线性插值

所以新的蓝图要在这两条线上**交叉推进，但顺序上先卡死“密度原始 3D 是否一致”**，再看几何和插值。

---

## 二、重新分段的调试蓝图（从已完成往下）

我用你原来的 6 个阶段做外层框架，在里面插入更细的子阶段，方便你一段段打钩。

---

### ✅ 阶段 1：参数解析 & 路径配置（已完成）

* 不再重点动它，之后如果有问题，一般也不是这儿导致的。

---

### ✅ 阶段 2：GENE 配置 & Equilibrium 数据加载（已完成）

* 你现在的 `debug_gene_param_eq_py.npz` vs MATLAB 的 `debug_gene_param_eq_ml_301_t9807.mat` 已经说明：

  * 物理参数、`inside/outside`、EQ 网格、磁场都对上 → **这层当基石用就行了**。

---

### ✅ 阶段 2.5：密度 3D 读入一致性（已完成）

> 这是之前蓝图里缺的一环，现在要单独拎出来放在 beam geometry 之前。

**目标：**

* 确认：**同一个 `00009807.dat`**，Python 的 `fread_data_s` 输出的 3D 张量，和 MATLAB 的对应函数输出的 3D 数组，在：

  * `shape` 上完全相同（比如 `(400, 128, 29)`）
  * 数值上 `max abs diff ~ 1e-12` 或更小

**要对比的对象：**

* MATLAB：刚从二进制 reshape 成 `(ntheta, nx, nz)` 的那一坨（还没做 padding / 周期边界）
* Python：你当前 `fread_data_s` 返回的 `density_3d`

**文件命名建议：**

* Python：`debug_gene_density_py_t9807.npz`（变量名 `density3d`）
* MATLAB：`debug_gene_density_ml_00009807.mat`（变量名例如 `data3d`）

**通过标准：**

* ✅ 第一层：`shape` 一致 → 确认维度顺序没搞反（ntheta / nx / nz）
* ✅ 第二层：`max abs diff` 在 1e-12 左右 → reshape / 坐标次序都一致

**验证结果：**

* ✅ `shape` 完全一致：`(400, 128, 29)` = `(ntheta, nx, nz)`
* ✅ 数值完全一致：`max abs diff = 0.0`，`rel diff = 0.0`
* ✅ **结论**：GENE 原始输出 → 0000XXXX.dat → (ntheta, nx, nz) 这条链路在 Python 和 MATLAB 之间已经完全等价，可以放心当成"验证通过"的模块。

➡️ **阶段 2.5 已通过，可以进入 Stage 3 / 4。**

---

### 🔜 阶段 3：光束配置和几何计算（Beam Geometry）

> 在 2.5 OK 后，这一层只依赖：
>
> * GENE/EQ 配置（已经验证好）
> * `LS_condition_JT60SA.txt` + 一些几何公式

**目标：**

* Python 的 `compute_beam_grid` 生成的 `beam_grid`
  与 MATLAB 在 `LSview_com.m` 中生成的 `(xls, yls, zls)` 组成的 3D 网格，在：

  * 形状上相同 → `(2*div1+1, 2*div2+1, divls+1, 3)`
  * 数值上 `allclose`（允许 1e-12 左右误差）

**要对比的内容：**

* 三维网格：

  * MATLAB：可以把 `xls, yls, zls` reshape 成 `(div1_2, div2_2, divls_2, 3)` 统一存为 `beam_grid_3d`
  * Python：`beam_grid` 张量（本来就是 `[v_idx, t_idx, s_idx, 3]` 这种结构）
* 起止点：`B2_start / B2_end` 以及 `p1`（光束方向向量）

**确认点：**

* φ 角的处理是否完全一致：`phi * 2 * pi`
* mm → m 的单位换算是否在两个代码里都只做了一次
* `div1/div2/divls` 的遍历次序：MATLAB 是外层 j=1:div1_2 再 j=1:div2_2 再 j=1:divls_2；Python 必须映射成同样顺序

**文件建议：**

* MATLAB：`debug_stage_3_matlab.mat` → `beam_grid_3d`
* Python：`debug_stage_3_python.npz` → `beam_grid_3d`

---

### 🔜 阶段 4：密度场预处理 – “从 3D 到 processed_density_field”

> 这一层对应你原来的“第4阶段：数据预处理和时间序列处理”，但我们只看**一个时间点**。

内部可以再细拆两小步，方便排错：

#### 4.1 在 MATLAB 里拆出“3D → 扩展 + 重排”的入口/出口

**目标：**

* 在 MATLAB 中找到：`data3d`（2.5 里的原始 3D）是在哪一步被：

  * 扩展径向（inside/outside padding）
  * 添加 toroidal/poloidal 周期边界
  * 重排 poloidal/phi 的索引
* 找一个“最终进入 `probe_multi2` 插值环节前”的密度场版本，命名为 `processed_density_field_ml` 保存起来。

#### 4.2 Python 侧对应位置导出同样的 processed 版本

* 你 Python 这边相应的位置多半在：

  * `fread_data_s` 后，或者
  * `forward_projection` 内部在送入插值前的一层缓存（视你代码结构而定）

**目标：**

* Python 侧拿到与 MATLAB **语义上等价**的 `processed_density_field_py`：

  * `shape` 一致（注意：这一步的 `nx` 可能已经不是 `nx0` 了，而是 `nx0 + inside + outside + 边界点`）
  * 周期边界最后一列/行是不是第一列/行的拷贝
  * padding 区域的数值匹配（是否复制边界，还是填 0）

**对比重点：**

* **reshape 顺序**：是否总是 `(ntheta, nx, nz)` 一致
* **padding 策略**：两边是不是都“复制边界值”而不是填 0（或别的）
* **poloidal 方向翻转 / 重排**：有些代码会在 poloidal 上做 `flipud / circshift`，要看是否同步做了

---

### 🔜 阶段 5：PCI 正向投影（核心插值）的单点调试

> 在 2.5、3、4 都通过后再干这一步，否则你会看见一堆乱差却不知道是谁的锅。

**核心策略：**

* 不先对比完整的 `pout1/pout2`，而是**挑一条光线上的单个采样点**当“探针”，逐层对比中间量。

**建议流程：**

1. 选一个固定的 `(iv, it, is)`，例如 beam_grid 的中心或某个非边界点：

   * MATLAB：在 `probe_multi2` / `probeEQ_local_s` 里打印/存这个点对应的输入 `(R,Z,phi)` 和中间变量；
   * Python：在 `forward_projection` / 插值函数里对同一个索引做同样的 dump。

2. 对比以下中间量（两边一一对应）：

   * 坐标变换：

     * `(R,Z,phi)` → `(r, theta)`（相对磁轴）
   * `bisec` 查找结果：

     * radial index / theta index / phi index，注意 MATLAB 1-based vs Python 0-based
   * flux surface 内部/外部判断的布尔值
   * 参与三线性插值的 8 个网格点值（密度）
   * 8 个权重 & 插值结果一个标量

3. 单点一致后，再放开为**一条光线**的所有采样点，对比这条线上的插值结果向量。

---

### 🔜 阶段 6：整条光束 / 全平面信号 & 可视化

当上面都 OK 时：

* 对比完整的：

  * `pout1` ：沿光束的 3D 信号
  * `pout2` ：探测器面上的 2D 积分信号
* 再对比 FFT 后的波数谱（kx-ky平面），确认：

  * FFT 正反、归一化方式
  * `fftshift` 是否一致
  * k 轴的单位是否同样乘了 `rho_ref`

这一步如果出了差，通常就是**整体归一化系数 / 滤波 / 平移**这样的末端问题，已经比较容易了。

---

## 三、执行顺序总结（给你一个简版 checklist）

从现在开始（你已经过了 Stage 2）：

1. **✅ Stage 2.5：原始 3D 密度 `density3d` 对比**

   * ✅ shape 一致：`(400, 128, 29)`
   * ✅ 值完全一致：`max abs diff = 0.0`，`rel diff = 0.0`

2. **Stage 3：光束几何 `beam_grid_3d` 对比**

3. **Stage 4：processed density（padding + 周期 + 重排）对比**

4. **Stage 5：单点插值中间量对比 → 单光束对比**

5. **Stage 6：整面 `pout2` / 波数谱 / 可视化对比**

你可以把这套蓝图打印出来，当 checklist 用：每通过一小步，就在那一行打个 ✓，并在 Git 里做一次 commit，方便回滚。

如果你接下来确定了 2.5 的对比结果（shape / max diff），直接贴出来，我们就按这个蓝图继续往下细化对应阶段的具体调试代码。
