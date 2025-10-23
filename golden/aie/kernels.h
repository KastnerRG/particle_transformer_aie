
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"


template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
  input_stream_int8 * __restrict sA,
  output_stream_int8 * __restrict sC,
  const int8 matB []
  ) {
  using MMUL = aie::mmul<m, k, n, int8, int8>; // m = 4, k = 8, n = 8
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;
  int count = 0;
  
  const int8* __restrict Bbase = (const int8*)matB;
  const unsigned strideB_perK  = MMUL::size_B * Tn; // row length

  for (unsigned im = 0; im < Tm; ++im) {
   // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik){ // read in all tiles in a row
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }
    for (unsigned in = 0; in < Tn; ++in) { //column of B
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B; // current tile in row

      for (unsigned ik = 0; ik < Tk; ++ik) {//row of B
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK); // ptr + current tile in row + curr row * row length
        if (ik == 0) C.mul(Abuf[0], b); // row 0
        else         C.mac(Abuf[ik], b); // row Tk
      }

      VC v = C.template to_vector<int8>(SHIFT);
      if (is_relu) v = aie::max(v, (int8)0);
      writeincr(sC, v);
      count++;
    }
  }
}

// (Q @ K^T):  (T, d_model) @ (T, d_model)^T -> (T, T)
// m=4, k=8, n=8, T=160, d_model=64, Tm(rows)=160/m=40, Tn(columns)=64/n=8
template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_S>
void scores(
  input_stream_int8 * __restrict sQ, // adf::input_buffer<int8, adf::extents<T*d_model>> & sQ,
  input_stream_int8 * __restrict sK, // adf::input_buffer<int8, adf::extents<T*d_model>> & sK,
  output_stream_int8 * __restrict sS
) {
  using MMUL = aie::mmul<m, n, m, int8, int8>; // 4x8x4
  using VA   = aie::vector<int8, MMUL::size_A>; // 4x8
  using VB   = aie::vector<int8, MMUL::size_A>; // 8x4
  using VC   = aie::vector<int8, MMUL::size_C>; // 4x4

  VB matB[Tm*Tn]; //store all of matB in mem

  for (unsigned i = 0; i < Tm; ++i) { // rows
    for (unsigned j = 0; j < Tn; ++j) { // columns
      matB[i*Tn+j] = aie::transpose(readincr_v<MMUL::size_A>(sK), m, n);
    }
  }
  
  // row by row multiplication
  for (unsigned im = 0; im < Tm; ++im) {   // rows of Q
    VA Abuf[Tn]; // row of tiles
    for (unsigned in = 0; in < Tn; ++in) { // columns of Q
      Abuf[in] = readincr_v<MMUL::size_A>(sQ);
    }
    for (unsigned jm = 0; jm < Tm; ++jm) { // rows of K
      MMUL C;
      for (unsigned in = 0; in < Tn; ++in) { // columns of K
        if (in == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
        else         C.mac(Abuf[in], matB[jm*Tn+in]);
      }
      VC V = C.template to_vector<int8>(SHIFT_S);
      writeincr(sS, V);
    }
  }
}

// (scores @ V)  (T,T) @ (T,d_model) -> (T,d_model) mxk
// Tm = 160/4 = 40, Tk = 160/4 = 20, Tn = 64/8 = 8
// 160 x 160 x 64 tiled with 4 x 4 x 8
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT>
void context(
  input_stream_int8 * __restrict sS,
  input_stream_int8 * __restrict sV,
  output_stream_int8 * __restrict sC
) {
  using MMUL = aie::mmul<m, m, n, int8, int16>; // 4x4x8 -> 4x8
  using VA   = aie::vector<int8,  MMUL::size_A>; // mxm (int8)
  using VB   = aie::vector<int16, MMUL::size_B>; // mxn (int16)
  using VC   = aie::vector<int16, MMUL::size_C>; // mxn (int16)

  using VBin = aie::vector<int8, MMUL::size_B>; // 4x8 (int8)
  using VCout = aie::vector<int8, MMUL::size_C>;

  VB matB[Tm*Tn];

  for (unsigned im = 0; im < Tm; ++im) { // rows
    for (unsigned in = 0; in < Tn; ++in) { // columns
      VBin B = readincr_v<32>(sV); // 4x8
      VB B16 = B.unpack();
      matB[im*Tn+in] = B16; //convert to int16 for 4x4x8
    }
  }

  for (unsigned im = 0; im < Tm; ++im) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tm];
    for (unsigned jm = 0; jm < Tm; ++jm) {
      Abuf[jm] = readincr_v<MMUL::size_A>(sS); // one tile
    }
    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      for (unsigned jm = 0; jm < Tm; ++jm) {//row of B
        if (jm == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
        else         C.mac(Abuf[jm], matB[jm*Tn+in]);
      }

      VC v = C.template to_vector<int16>(SHIFT);
      VCout vout = v.pack();
      writeincr(sC, vout);
    }
  }
}


// (context @ Wo)  (T,d_model) @ (d_model,d_model) -> (T,d_model)
template <int m, int k, int n, int d_model, int T, int SHIFT_O>
void output(
  input_stream_int8* __restrict context_in,
  output_stream_int8* __restrict out_stream,
  const int8 Wo[]
) {
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  const int Tm = T / m;
  const int Tk = d_model / k;
  const int Tn = d_model / n;

  const int8* __restrict Bbase = (const int8*)Wo;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im)
  chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik)
      Abuf[ik] = readincr_v<MMUL::size_A>(context_in);

    for (unsigned in = 0; in < Tn; ++in)
    chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT_O);
      writeincr(out_stream, v);
    }
  }
}

#endif