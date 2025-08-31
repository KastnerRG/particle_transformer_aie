
#include "kernels.h"

void attn1(input_window_int8 * __restrict q_head, input_window_int8 * __restrict k_head, output_window_int8 * __restrict o_head){ attention<2, 8, 8, 16, 150, 8, 11>(q_head, k_head, o_head);}
