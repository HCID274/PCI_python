#!/bin/bash
# 检查PBS作业状态

JOB_ID=117917

echo "==================== 作业状态 ===================="
qstat -f $JOB_ID.pbs | grep -E "(Job_Name|job_state|queue|exec_host|resources_used)" | head -10

echo ""
echo "==================== 输出日志 ===================="
if [ -f "pyPCI_complete.o${JOB_ID}" ]; then
    echo "最新输出 (最后50行):"
    tail -50 pyPCI_complete.o${JOB_ID}
else
    echo "还没有输出文件"
fi

echo ""
echo "==================== 错误日志 ===================="
if [ -f "pyPCI_complete.e${JOB_ID}" ]; then
    echo "错误输出:"
    tail -30 pyPCI_complete.e${JOB_ID}
else
    echo "还没有错误文件"
fi

echo ""
echo "==================== 输出目录 ===================="
if [ -d "/work/DTMP/lhqing/PCI/Data/sample/output" ]; then
    echo "输出文件:"
    ls -lh /work/DTMP/lhqing/PCI/Data/sample/output/ | head -20
else
    echo "输出目录不存在"
fi

