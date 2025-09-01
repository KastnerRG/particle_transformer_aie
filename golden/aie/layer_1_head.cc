
#include "kernels.h"

void head1(input_window_int8 * __restrict x, input_window_int8 * __restrict v, output_window_int8 * __restrict a){ dense_in<2, 8, 8, 75, 18, 2, 10, false>(x, a, v);}
