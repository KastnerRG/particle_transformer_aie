
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
  input_window_int8 * __restrict matA, 
  output_window_int8 * __restrict matC,
  const int8 matB []
  ){
  using MMUL = aie::mmul<m, k, n, int8, int8>;

  const int8* __restrict pA=(int8*)matA->ptr;
  const int8* __restrict pB=(int8*)matB;
  int8* __restrict pC = (int8*) matC->ptr;

  //For profiling only 
  unsigned long long cycle_num[2];
  aie::tile tile=aie::tile::current();
  cycle_num[0]=tile.cycles();

  for (unsigned im = 0; im < Tm; ++im) 
  chess_unroll_loop(Tm)
  {
    for (unsigned in = 0; in < Tn; ++in) 
    chess_unroll_loop(Tn)
    {
      const int8 * __restrict pA1 = pA + ( im * Tk + 0) * MMUL::size_A;
      const int8 * __restrict pB1 = pB + ( 0 * Tn + in) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      aie::vector<int8, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

      MMUL C; 
      C.mul(A, B);

      for (unsigned ik = 0; ik < Tk-1; ++ik) 
      chess_flatten_loop
      {
        A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;
        C.mac(A, B);
      }
      auto C_vec = C.template to_vector<int8>(SHIFT);
      auto C_out = is_relu ? aie::max(C_vec, (int8)0) : C_vec;
      aie::store_v(pC, C_out); pC += MMUL::size_C;
    }
  }
  //For profiling only 
  cycle_num[1]=tile.cycles();
  printf("start=%lld,end=%lld,total=%lld\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);
}

#endif