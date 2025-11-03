#!/bin/bash

source /tools/Xilinx/Vivado/2024.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib:/tools/Xilinx/Vitis/2024.1/aietools/lib/lnx64.o
export AIE_COMPILER_THREADS=128

v++ -c \
  --mode aie \
  --include $XILINX_VITIS/aietools/include \
  --include "./aie" \
  --aie.xlopt=1 \
  --platform $XILINX_VITIS/base_platforms/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm \
  --work_dir ./Work \
  --target hw \
  --aie.heapsize=6000 \
  --aie.stacksize=26000 \
  aie/graph.cpp
  # --aie.xlopt=2 \
  # --aie.Xxloptstr="-annotate-pragma" \

aiesimulator \
  --profile \
  --dump-vcd=aiesim \
  --pkg-dir=./Work
  # --hang-detect-time=5000000 \
  # --evaluate-fifo-depth \