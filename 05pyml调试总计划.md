该计划的核心是**“分段输出，精确对比”**，确保在进入下一阶段前，当前阶段的所有中间数据都与MATLAB的“黄金标准”完全一致。

---

### **调试总策略**

1.  **环境准备**:
    *   在MATLAB代码的每个阶段末尾，使用 `save('debug_stage_X.txt', 'var1', 'var2', ...)` 命令，将该阶段的关键变量保存为 `.txt` 文件。
    *   在Python中，确保已安装 `scipy` 和 `numpy` 库。`scipy.io.loadmat` 可以读取MATLAB保存的文件，`numpy.allclose` 是对比浮点数数组的利器。
    *   建议使用版本控制工具（如 Git），每通过一个阶段的调试，就进行一次提交（commit），方便追踪问题和回退。

2.  **对比原则**:
    *   **形状 (Shape) 优先**: 首先检查数组的维度和大小 (`.shape`) 是否完全一致。
    *   **数据类型 (Dtype) 其次**: 确保数据类型匹配（例如 `float64` vs `float32`）。
    *   **数值精度 (Value) 最后**: 使用 `numpy.allclose(matlab_var, python_var, rtol=1e-5, atol=1e-8)` 进行浮点数比较，不要用 `==` 直接比较。`rtol` (相对容差) 和 `atol` (绝对容差) 可以根据需要调整。
    *   **注意索引**: 始终牢记 **MATLAB是1-based索引，Python是0-based索引**。这是最常见的错误来源。

---

### **分阶段调试计划 (Step-by-Step Debugging Plan)**

#### **第1阶段：参数解析和路径配置 (Parameter & Path Verification)**

*   **🎯 目标**: 验证所有从命令行和配置文件中读取的参数完全一致。
*   **MATLAB 操作**:
    1.  在 `LSview_com` 函数的路径配置完成后，保存所有解析出的参数。
    2.  添加代码:
        ```matlab
        % Stage 1 Debug Output
        debug_data.sim_type = 'GENE';
        debug_data.data_no = 301;
        debug_data.time_point = 98.07;
        debug_data.var_type = 4;
        debug_data.input_dir = '/work/DTMP/lhqing/PCI/Data/matlab_input/301/';
        debug_data.output_dir = '/work/DTMP/lhqing/PCI/Data/matlab_output/';
        debug_data.byte_order = 'ieee-le';
        save('debug_stage_1_matlab.txt', '-struct', 'debug_data');
        ```
*   **Python 操作**:
    1.  在`run_pci.py`中，解析完 `argparse` 和 `paths.json` 后，将对应的配置变量保存。
    2.  使用 `numpy` 或 `scipy.io.savemat` 保存：
        ```python
        # Stage 1 Debug Output
        import scipy.io

        debug_data = {
            'sim_type': task_config.sim_type,
            'data_no': task_config.data_no,
            'time_point': task_config.time,
            'var_type': task_config.var,
            'input_dir': path_config.input_dir,
            'output_dir': path_config.output_dir,
            # 确认Python代码中如何处理字节序
            'byte_order': 'ieee-le'
        }
        scipy.io.savemat('debug_stage_1_python.txt', debug_data)
        ```
*   **对比与检查**:
    *   加载两个 `.txt` 文件，逐一对比字符串和数值是否相等。
    *   **检查点**: 路径的末尾是否有斜杠 `/`？字符串大小写是否一致？

#### **第2阶段：GENE配置和Equilibrium数据加载 (Config & EQ Data Loading)**

*   **🎯 目标**: 验证 `parameters.dat` 和 Equilibrium 二进制文件 (`equdata_BZ`, `equdata_be`) 的读取结果完全一致。这是**极其关键且易错**的一步。
*   **MATLAB 操作**:
    1.  在 `fread_param2.m` 和 `fread_EQ1.m` 执行完毕后，保存所有加载和计算出的变量。
    2.  添加代码:
        ```matlab
        % Stage 2 Debug Output
        save('debug_stage_2_matlab.txt', ...
             'nx0', 'nky0', 'nz0', 'q0', 'shat', 'trpeps', ... % from parameters.dat
             'GRC', 'GZC', 'GFC', 'PA', 'GAC', ...             % from equdata_BZ
             'GBPR', 'GBPZ', 'GBTP', 'GBPP', 'RG', 'DR', ...   % from equdata_be
             'B_ref', 'rho_ref');                             % derived params
        ```*   **Python 操作**:
    1.  在 `load_gene_config_from_parameters` 和 `load_equilibrium_data` 执行完毕后，保存对应的 NumPy 数组。
    2.  添加代码:
        ```python
        # Stage 2 Debug Output
        # ... after loading all data ...
        scipy.io.savemat('debug_stage_2_python.txt', {
            'nx0': gene_config.nx0, 'nky0': gene_config.nky0, 'nz0': gene_config.nz0,
            'q0': gene_config.q0, 'shat': gene_config.shat, 'trpeps': gene_config.trpeps,
            'GRC': eq_data.GRC, 'GZC': eq_data.GZC, 'GFC': eq_data.GFC,
            'PA': eq_data.PA, 'GAC': eq_data.GAC,
            'GBPR': eq_data.GBPR, 'GBPZ': eq_data.GBPZ, 'GBTP': eq_data.GBTP, 'GBPP': eq_data.GBPP,
            'RG': eq_data.RG, 'DR': eq_data.DR,
            'B_ref': gene_config.B_ref, 'rho_ref': gene_config.rho_ref
        })
        ```
*   **对比与检查**:
    *   **字节序 (Endianness)**: 你的流程树提到 `equdata_be` 是 Big-Endian，`equdata_BZ` 可能是 Little-Endian。请务必在Python的 `numpy.fromfile` 或等效函数中正确设置 `dtype`，例如 `np.dtype('>f8')` 表示 Big-Endian 8字节浮点数。这是最常见的二进制文件读取错误。
    *   **FORTRAN Namelist**: 确认Python的解析器能正确处理FORTRAN的namelist格式，特别是数组的赋值方式。
    *   **数组形状**: MATLAB读取的数组可能是Fortran顺序（列优先），而NumPy默认是C顺序（行优先）。在 `reshape` 时可能需要指定 `order='F'`。使用 `.shape` 仔细检查。
    *   **数值对比**: 使用 `numpy.allclose` 逐个对比所有数组。

#### **第3阶段：光束配置和几何计算 (Beam Geometry)**

*   **🎯 目标**: 验证生成的三维光束采样网格 (`beam_grid`) 完全一致。
*   **MATLAB 操作**:
    1.  在生成3D光束采样网格后，保存该网格。
    2.  添加代码 (假设网格变量名为 `beam_grid_3d`):
        ```matlab
        % Stage 3 Debug Output
        save('debug_stage_3_matlab.txt', 'beam_grid_3d'); % Shape: (div1*2+1, div2*2+1, divls+1, 3)
        ```
*   **Python 操作**:
    1.  在 `compute_beam_grid` 函数执行后，保存返回的 `beam_grid` 张量/数组。
    2.  添加代码:
        ```python
        # Stage 3 Debug Output
        # beam_grid is a torch.Tensor, convert to numpy
        scipy.io.savemat('debug_stage_3_python.txt', {'beam_grid_3d': beam_grid.cpu().numpy()})
        ```
*   **对比与检查**:
    *   **数组维度顺序**: 确认 MATLAB `(i, j, k, l)` 的维度顺序与 Python `(i, j, k, l)` 完全对应。
    *   **浮点数精度**: 向量计算 (`B2_start - B2_end`) 和三角函数 (`phi * 2 * pi`) 的浮点数误差可能会累积。使用 `numpy.allclose` 进行对比。

#### **第4阶段：数据预处理 (Density Field Preprocessing)**

*   **🎯 目标**: 验证从二进制文件读取并经过Reshape、Padding和重排后的最终密度场数据完全一致。
*   **MATLAB 操作**:
    1.  在 `probe_multi2.m` 内部，在进入核心插值计算循环之前，保存最终处理好的3D密度场数组 (假设变量名为 `processed_density_field`)。
    2.  添加代码:
        ```matlab
        % Stage 4 Debug Output
        save('debug_stage_4_matlab.txt', 'processed_density_field', '-v7.3'); % Use -v7.3 for large arrays
        ```
*   **Python 操作**:
    1.  在 `fread_data_s` (或等效的加载和预处理函数) 执行完毕后，保存最终的密度场数组。
    2.  添加代码:
        ```python
        # Stage 4 Debug Output
        # density_field_tensor is the final processed tensor
        scipy.io.savemat('debug_stage_4_python.txt', {'processed_density_field': density_field_tensor.cpu().numpy()})
        ```
*   **对比与检查**:
    *   **Reshape顺序**: 同样，注意 `order='F'` vs `order='C'` 的问题。
    *   **Padding**: 检查径向内外扩展的逻辑和数值是否与MATLAB完全一致。
    *   **Poloidal数据重排**: 这是一个复杂的索引操作，需要逐行对比MATLAB的实现逻辑。
    *   **周期边界**: 确认添加的周期边界是否正确（例如，最后一列/行是否是第一列/行的拷贝）。

#### **第5阶段：PCI正向投影计算 (Forward Projection - The Core)**

*   **🎯 目标**: 逐一拆解和验证核心插值算法 `probeEQ_local_s.m` / `probe_local_trilinear`。这是整个调试过程的**核心难点**。
*   **策略**: 不要直接对比最终的 `pout1` 和 `pout2`。必须深入函数内部，对比中间结果。选择光束路径上的**同一个采样点** (例如，`beam_grid` 的 `[0,0,0,:]` 点) 作为“探针”，对比其所有中间计算结果。
*   **子步骤与操作**:
    1.  **5a. 坐标转换**:
        *   **MATLAB**: 在 `probeEQ_local_s.m` 中，保存计算出的相对磁轴的 `(r, theta)`。
        *   **Python**: 在 `probe_local_trilinear` 中，保存对应的 `r` 和 `theta`。
        *   **对比**: `numpy.allclose`。检查 `mod` 函数的行为差异（MATLAB的 `mod` 和 Python的 `%` 对负数的处理不同，`a - m * floor(a/m)` 是正确的跨语言实现）。
    2.  **5b. `bisec` 查找**:
        *   **MATLAB**: 保存 `bisec` 函数返回的索引值 (如 `theta_p1`, `r_p1` 等)。
        *   **Python**: 保存 `bisec_batch` 或 `bisec` 返回的索引值。
        *   **对比**: **直接对比数值**。MATLAB返回的是1-based，Python返回的是0-based。对比前，将MATLAB的索引减1 (`matlab_index - 1`)。
    3.  **5c. 边界检查**:
        *   **MATLAB**: 保存边界检查的逻辑结果 (一个布尔值，表示点是否在plasma内部)。
        *   **Python**: 保存对应的布尔值。
        *   **对比**: 必须相等。注意 `(r < GAC[theta_p1]) AND (r < GAC[theta_p2])` 逻辑中的索引是否正确。
    4.  **5d. 权重计算**:
        *   **MATLAB**: 保存计算出的8个三线性插值权重。
        *   **Python**: 保存对应的8个权重。
        *   **对比**: `numpy.allclose`。
    5.  **5e. 插值结果**:
        *   **MATLAB**: 保存对单个点插值后的密度场采样值。
        *   **Python**: 保存对应的值。
        *   **对比**: `numpy.allclose`。
    6.  **5f. 最终信号 (`pout1`, `pout2`)**:
        *   只有当以上所有子步骤对单个点都验证通过后，才去对比最终的积分信号 `pout1` 和检测器信号 `pout2`。
        *   **MATLAB**: 保存 `pout1` 和 `pout2`。
        *   **Python**: 保存 `pci_signal_1d` 和 `detector_signal_2d`。
        *   **对比**: `numpy.allclose`。

#### **第6阶段：可视化和结果输出 (Visualization)**

*   **🎯 目标**: 验证绘图数据的一致性。
*   **策略**: 如果第5阶段的输出 `pout2` 已经完全一致，那么这一阶段的问题通常出在绘图函数本身。
*   **操作**:
    1.  **Figure 3 (检测器信号)**: 对比传入 `contourf` 的 `pout2` 数据。既然数据已经验证过，问题可能在于 `meshgrid` 的 `indexing='xy'` (默认) vs `indexing='ij'` 参数，或者 `contourf(X, Y, Z)` 的参数顺序。
    2.  **Figure 4 (波数空间)**:
        *   **MATLAB**: 保存 `fft2` 的输入和输出。
        *   **Python**: 保存 `fft2` 的输入和输出。
        *   **对比**: `numpy.allclose` 对比复数数组。确认FFT的归一化因子 (`norm="ortho"`) 和 `fftshift` 的使用是否一致。
        *   **坐标轴**: 确认 `kx`, `ky` 的计算和归一化 (`*rho_ref`) 逻辑完全相同。

### **总结与建议**

1.  **耐心是关键**: 这个过程会很繁琐，但非常有效。严格按照阶段顺序，不要跳步。
2.  **从简到繁**: 从单个标量开始，然后是一维数组，最后是高维数组。对于核心算法（第5阶段），从单个点开始调试，成功后再扩展到整个数组。
3.  **日志记录**: 在Python代码中多使用 `print` 或日志库，输出中间变量的形状和数值，方便快速定位问题。
4.  **优先怀疑点**:
    *   **字节序 (Endianness)** in Stage 2.
    *   **数组维度顺序 (Fortran vs. C)** in Stages 2, 4.
    *   **索引 (1-based vs. 0-based)** in Stage 5.
    *   **浮点数精度问题** throughout.

遵循以上计划，您将能够系统性地定位并修复Python代码中的问题。祝您调试顺利！