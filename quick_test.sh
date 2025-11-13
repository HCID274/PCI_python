#!/bin/bash
# 快速测试脚本 - 在提交PBS作业前验证环境

echo "==================== PyPCI 快速测试 ===================="
echo "测试时间: $(date)"
echo ""

# 切换到pyPCI目录
cd /work/DTMP/lhqing/PCI/code/pyPCI

# 检查虚拟环境
echo "1. 检查虚拟环境..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "   ✓ 虚拟环境已激活"
else
    echo "   ✗ 错误: 未找到虚拟环境"
    exit 1
fi

# 检查Python和PyTorch
echo ""
echo "2. 检查Python环境..."
python --version
echo "   PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "   GPU可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 检查输入文件
echo ""
echo "3. 检查输入文件..."
INPUT_DIR="/work/DTMP/lhqing/PCI/Data/sample/input"
for file in parameters.dat equdata_BZ equdata_be LS_condition_JT60SA.txt TORUSIons_act.dat; do
    if [ -f "$INPUT_DIR/$file" ]; then
        size=$(du -h "$INPUT_DIR/$file" | cut -f1)
        echo "   ✓ $file ($size)"
    else
        echo "   ✗ 缺失: $file"
    fi
done

# 测试导入
echo ""
echo "4. 测试模块导入..."
python -c "
try:
    from pci_torch import (
        load_gene_config_from_parameters,
        load_beam_config,
        process_time_series,
        FFT3DAnalyzer
    )
    print('   ✓ 所有模块导入成功')
except Exception as e:
    print(f'   ✗ 导入失败: {e}')
    exit(1)
"

# 测试配置加载
echo ""
echo "5. 测试配置加载..."
python -c "
import sys
sys.path.insert(0, '/work/DTMP/lhqing/PCI/code/pyPCI')
try:
    from pci_torch.data_loader import load_gene_config_from_parameters, load_beam_config
    
    config = load_gene_config_from_parameters(
        '/work/DTMP/lhqing/PCI/Data/sample/input/parameters.dat',
        '/work/DTMP/lhqing/PCI/Data/sample/input',
        device='hip'
    )
    print(f'   ✓ GENE配置加载成功')
    print(f'     - nx0={config.nx0}, nky0={config.nky0}, nz0={config.nz0}')
    print(f'     - q0={config.q0}, shat={config.shat}')
    
    beam_config = load_beam_config('/work/DTMP/lhqing/PCI/Data/sample/input/LS_condition_JT60SA.txt')
    print(f'   ✓ 光束配置加载成功')
    print(f'     - 检测器: {beam_config.n_detectors_v} x {beam_config.n_detectors_t}')
    print(f'     - 光束采样点: {beam_config.n_beam_points}')
    
except Exception as e:
    print(f'   ✗ 配置加载失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "==================== 测试结果 ===================="
    echo "✓ 所有测试通过！"
    echo ""
    echo "可以提交PBS作业:"
    echo "  cd /work/DTMP/lhqing/PCI/code/pyPCI"
    echo "  qsub run_pci_gpu.pbs"
    echo ""
    echo "或运行完整pipeline:"
    echo "  python examples/complete_pipeline.py \\"
    echo "    --input_dir /work/DTMP/lhqing/PCI/Data/sample/input \\"
    echo "    --output_dir /work/DTMP/lhqing/PCI/Data/sample/output \\"
    echo "    --data_n 246 \\"
    echo "    --device cuda \\"
    echo "    --save_all_figures"
    echo "================================================="
else
    echo ""
    echo "==================== 测试失败 ===================="
    echo "请检查上面的错误信息"
    echo "================================================="
    exit 1
fi

