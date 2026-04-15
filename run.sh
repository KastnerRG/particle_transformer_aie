#!/bin/bash

source /tools/Xilinx/Vivado/2024.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib:/tools/Xilinx/Vitis/2024.1/aietools/lib/lnx64.o
export AIE_COMPILER_THREADS=128


# Optional controls for simulator behavior.
# Default is profile-off because some tool versions may abort during
# profiling report teardown even after kernels finish correctly.
AIESIM_PROFILE=${AIESIM_PROFILE:-0}
AIESIM_DUMP_VCD=${AIESIM_DUMP_VCD:-1}
AIESIM_VCD_NAME=${AIESIM_VCD_NAME:-aiesim}

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

aiesim_args=(--pkg-dir=./Work)

if [ "$AIESIM_PROFILE" = "1" ]; then
  aiesim_args+=(--profile)
  echo "[run.sh] AIESimulator profiling enabled"
else
  echo "[run.sh] AIESimulator profiling disabled (default)"
fi

if [ "$AIESIM_DUMP_VCD" = "1" ]; then
  aiesim_args+=("--dump-vcd=${AIESIM_VCD_NAME}")
  echo "[run.sh] AIESimulator VCD dump enabled: ${AIESIM_VCD_NAME}.vcd"
else
  echo "[run.sh] AIESimulator VCD dump disabled"
fi

aiesimulator "${aiesim_args[@]}"
  # --hang-detect-time=5000000 \
  # --evaluate-fifo-depth \