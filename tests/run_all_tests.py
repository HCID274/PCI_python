"""
运行所有测试

使用方法:
    python run_all_tests.py
"""

import pytest
import sys
from pathlib import Path


def main():
    """运行所有测试"""
    test_dir = Path(__file__).parent
    
    print("=" * 70)
    print("运行 PyPCI 测试套件")
    print("=" * 70)
    
    # 运行pytest
    args = [
        str(test_dir),
        "-v",           # verbose
        "-s",           # 不捕获输出
        "--tb=short",   # 简短的traceback
        "--color=yes",  # 彩色输出
    ]
    
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 70)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()



