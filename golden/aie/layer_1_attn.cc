
#include "kernels.h"

void attn1(output_stream_int8 * __restrict o_head, input_stream_int8 * __restrict q_head, input_stream_int8 * __restrict k_head){ attention<2, 8, 8, 80, 8, 20, 64, 160, 8>(q_head, k_head, o_head);}
