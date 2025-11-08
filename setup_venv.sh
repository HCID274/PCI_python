#!/bin/bash
# 创建和配置PyPCI虚拟环境

set -e  # 遇到错误立即退出

VENV_DIR="/work/DTMP/lhqing/PCI/code/pyPCI/venv"
PCI_DIR="/work/DTMP/lhqing/PCI/code/pyPCI"

echo "========================================"
echo "创建PyPCI虚拟环境"
echo "========================================"
echo "虚拟环境路径: $VENV_DIR"
echo "PyPCI路径: $PCI_DIR"
echo ""

# 检查是否已存在虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo "警告: 虚拟环境已存在，将重新创建"
    rm -rf "$VENV_DIR"
fi

# 创建虚拟环境
echo "步骤1: 创建Python虚拟环境..."
python3 -m venv "$VENV_DIR"

# 激活虚拟环境
echo "步骤2: 激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级pip和安装基础工具
echo "步骤3: 升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch for ROCm
echo "步骤4: 安装PyTorch for ROCm..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# 安装其他依赖
echo "步骤5: 安装其他依赖..."
cd "$PCI_DIR"
pip install -r requirements.txt

# 安装PyPCI（开发模式）
echo "步骤6: 安装PyPCI..."
pip install -e .

# 验证安装
echo ""
echo "========================================"
echo "验证安装"
echo "========================================"
python -c "
import torch
import sys
print('Python版本:', sys.version)
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
print('CUDA设备数:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA设备名:', torch.cuda.get_device_name(0))
    print('CUDA架构:', torch.cuda.get_arch_list())
"

echo ""
echo "尝试导入pci_torch..."
python -c "import pci_torch; print('pci_torch导入成功，版本:', pci_torch.__version__)"

echo ""
echo "========================================"
echo "虚拟环境创建完成！"
echo "========================================"
echo "激活命令: source $VENV_DIR/bin/activate"
echo ""



