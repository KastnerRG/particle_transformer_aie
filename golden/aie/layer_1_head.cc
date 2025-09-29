
#include "kernels.h"

void head1(input_stream_int8 * __restrict x, input_stream_int8 * __restrict v, output_stream_int8 * __restrict a){ head<4, 8, 8, 40, 20, 8, 10>(x, a, v);}
