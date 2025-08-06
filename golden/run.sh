#!/bin/bash

v++ -c \
  --mode aie \
  --include $XILINX_VITIS/aietools/include \
  --include "./aie" \
  --aie.xlopt=0 \
  --platform $XILINX_VITIS/base_platforms/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm \
  --work_dir ./Work \
  --target hw \
  --aie.heapsize=16420 \
  aie/graph.cpp

aiesimulator \
  --profile \
  --dump-vcd=aiesim \
  --pkg-dir=./Work