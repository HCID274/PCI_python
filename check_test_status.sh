#!/bin/bash
# 检查PyPCI测试作业状态

echo "========================================"
echo "PyPCI测试作业状态检查"
echo "========================================"
echo ""

# 检查队列中的作业
echo "1. 当前作业状态:"
echo "----------------------------------------"
qstat -u lhqing | grep test_pypci || echo "  没有正在运行的test_pypci作业"
echo ""

# 列出最新的日志文件
echo "2. 最新的测试日志:"
echo "----------------------------------------"
LOGDIR="/work/DTMP/lhqing/PCI/code/pyPCI/test_results"
if [ -d "$LOGDIR" ]; then
    LATEST_LOG=$(ls -t ${LOGDIR}/test_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  日志文件: $LATEST_LOG"
        echo "  文件大小: $(du -h "$LATEST_LOG" | cut -f1)"
        echo "  修改时间: $(stat -c %y "$LATEST_LOG" | cut -d. -f1)"
    else
        echo "  没有找到日志文件"
    fi
else
    echo "  结果目录不存在"
fi
echo ""

# 如果有日志文件，显示最后50行
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "3. 日志末尾内容:"
    echo "----------------------------------------"
    tail -50 "$LATEST_LOG"
    echo "----------------------------------------"
fi

echo ""
echo "========================================"
echo "完整日志查看命令:"
echo "  cat $LATEST_LOG"
echo "========================================"



