
#include "kernels.h"

void attn1(input_stream_int8 * __restrict q_head, input_stream_int8 * __restrict k_head, output_stream_int8 * __restrict o_head){ attention<4, 8, 8, 40, 8, 8, 64, 160, 8>(q_head, k_head, o_head);}
