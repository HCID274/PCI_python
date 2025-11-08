#!/bin/bash
# 单时间点测试脚本（本地运行，不需要PBS）

cd /work/DTMP/lhqing/PCI/code/pyPCI

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "错误: 未找到虚拟环境"
    exit 1
fi

echo "=========================================="
echo "单时间点测试"
echo "=========================================="

python test_single_time.py \
    --input_dir /work/DTMP/lhqing/PCI/Data/sample/input \
    --output_dir /work/DTMP/lhqing/PCI/Data/sample/output/test_single \
    --time_t 0.83 \
    --device cpu

echo ""
echo "测试完成！查看结果："
echo "ls -lh /work/DTMP/lhqing/PCI/Data/sample/output/test_single/"

