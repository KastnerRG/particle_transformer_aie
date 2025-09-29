
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

// template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
// void dense(
//   input_stream_int8 * __restrict sA, 
//   output_stream_int8 * __restrict sC,
//   const int8 matB []
//   ){
//   using MM = aie::mmul<m, k, n, int8, int8>;
//   using VA = aie::vector<int8, MM::size_A>;
//   using VB = aie::vector<int8, MM::size_B>;
//   using VC = aie::vector<int8, MM::size_C>;

//   // Tune at build time with -DNB=2/4/8 as needed
//   const unsigned NB = 4;

//   const int8* __restrict Bbase = (const int8*)matB;
//   const unsigned strideB_perK  = MM::size_B * Tn;   // bytes to jump between successive K-slices
//   const unsigned blocksN       = (Tn + NB - 1) / NB;

//   // Iterate M-tiles
//   for (unsigned im = 0; im < Tm; ++im)
//   chess_prepare_for_pipelining chess_loop_range(1,)
//   {
//     // Buffer A for this M row once: Tk * size_A
//     VA Abuf[Tk];
//     for (unsigned ik = 0; ik < Tk; ++ik)
//     chess_prepare_for_pipelining chess_loop_range(1,)
//     {
//       Abuf[ik] = readincr_v<MM::size_A>(sA);
//     }

//     // Walk N in NB-sized blocks; emit each block as soon as its K loop is done
//     for (unsigned blk = 0; blk < blocksN; ++blk)
//     chess_prepare_for_pipelining chess_loop_range(1,)
//     {
//       const unsigned in0    = blk * NB;
//       const unsigned n_this = (in0 + NB <= Tn) ? NB : (Tn - in0);

//       // Up to NB accumulators (specialized with if-guards)
//       MM C0, C1, C2, C3;

//       // ---- K = 0: initialize accumulators with MUL
//       {
//         const VA A0 = Abuf[0];
//         const int8* __restrict pB0 = Bbase + (0 * strideB_perK) + in0 * MM::size_B;

//         if (n_this >= 1) { VB b0 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C0.mul(A0, b0); }
//         if (n_this >= 2) { VB b1 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C1.mul(A0, b1); }
//         if (n_this >= 3) { VB b2 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C2.mul(A0, b2); }
//         if (n_this >= 4) { VB b3 = aie::load_v<MM::size_B>(pB0);                    C3.mul(A0, b3); }
//       }

//       // ---- K = 1..Tk-1: MAC
//       for (unsigned ik = 1; ik < Tk; ++ik)
//       chess_prepare_for_pipelining chess_loop_range(1,)
//       {
//         const VA A  = Abuf[ik];
//         const int8* __restrict pBk = Bbase + (ik * strideB_perK) + in0 * MM::size_B;

//         if (n_this >= 1) { VB b0 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C0.mac(A, b0); }
//         if (n_this >= 2) { VB b1 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C1.mac(A, b1); }
//         if (n_this >= 3) { VB b2 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C2.mac(A, b2); }
//         if (n_this >= 4) { VB b3 = aie::load_v<MM::size_B>(pBk);                    C3.mac(A, b3); }
//       }

//       // ---- Quantize (+ReLU) and stream out immediately for this block
//       if (n_this >= 1) { VC v = C0.template to_vector<int8>(SHIFT); if (is_relu) v = aie::max(v,(int8)0); writeincr(sC, v); }
//       if (n_this >= 2) { VC v = C1.template to_vector<int8>(SHIFT); if (is_relu) v = aie::max(v,(int8)0); writeincr(sC, v); }
//       if (n_this >= 3) { VC v = C2.template to_vector<int8>(SHIFT); if (is_relu) v = aie::max(v,(int8)0); writeincr(sC, v); }
//       if (n_this >= 4) { VC v = C3.template to_vector<int8>(SHIFT); if (is_relu) v = aie::max(v,(int8)0); writeincr(sC, v); }
//     }
//   }
// }

template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
  input_stream_int8 * __restrict sA, 
  output_stream_int8 * __restrict sC,
  const int8 matB [] // Tk x Tn, k x n
  ) {
  using MMUL = aie::mmul<m, k, n, int8, int8>; // m = 4, k = 8, n = 8
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;
  
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
    }
  }
}

// (Q @ K^T)  (T, d_model) @ (T, d_model)^T -> (T, T)
// m = 4, k = 8, n = 8, Tm (rows) = 160 / m = 40, Tn (columns) = 64 / n = 8
template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_S>
void attention(
  input_stream_int8 * __restrict sQ, // adf::input_buffer<int8, adf::extents<T*d_model>> & sQ,
  input_stream_int8 * __restrict sK, // adf::input_buffer<int8, adf::extents<T*d_model>> & sK,
  output_stream_int8 * __restrict sS
) {
  using MMUL = aie::mmul<m, n, m, int8, int8>; // 4x8x4
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;
  VB matB[Tm*Tn];

  for (unsigned im = 0; im < Tm; ++im) { // rows
    for (unsigned in = 0; in < Tn; ++in) { // columns
      matB[in*Tm+in] = aie::transpose(readincr_v<MMUL::size_B>(sK), m, n); //curr row + curr column
    }
  }
  
  //int8* pQ = sQ.data(); // (160,64)
  //int8* pK = sK.data(); // (160,64)
  for (unsigned im = 0; im < Tm; ++im) {   // rows of Q and K
    VA Abuf[Tn]; // row of tiles
    MMUL C;
    for (unsigned in = 0; in < Tn; ++in) { // columns of Q
      Abuf[in] = readincr_v<MMUL::size_A>(sQ);
    }
    for (unsigned in = 0; in < Tn; ++in) { // columns of K
      if (in == 0) C.mul(Abuf[0], matB[im*Tn+in]);
      else         C.mac(Abuf[in], matB[im*Tn+in]);
    }
    VC V = C.template to_vector<int8>(SHIFT_S);
    writeincr(sS, V);
  }
}

// (scores @ V)  (T,T) @ (T,d_model) -> (T,d_model)
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT>
void head(
  input_stream_int8 * __restrict sS,
  output_stream_int8 * __restrict sC,
  input_stream_int8 * __restrict sV
) {
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  // Note that matB = buffer for V
  alignas(aie::vector_decl_align) VB matB[Tk*Tn];
  constexpr unsigned strideB_perK = MMUL::size_B * Tn;
  const int8* __restrict Bbase = (const int8*)matB;

  for (unsigned ik = 0; ik < Tk; ++ik) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      matB[ik*Tn+in] = readincr_v<MMUL::size_B>(sV);
    }
  }

  for (unsigned im = 0; im < Tm; ++im) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik)
      Abuf[ik] = readincr_v<MMUL::size_A>(sS); // one tile

    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {//row of B
        VB b = matB[ik*Tn];
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT);
      writeincr(sC, v);
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

// // dense layer with weights as input
// template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT>
// void head(
//   input_stream_int8 * __restrict matA,
//   output_stream_int8 * __restrict matC,
//   input_window_int8 * __restrict matB
//   ){
//   using MMUL = aie::mmul<m, k, n, int8, int8>;
//   // const int8* __restrict pA = (int8*) matA->ptr;
//   const int8* __restrict pB = (int8*) matB->ptr;
//   //int8* __restrict pC      = (int8*) matC->ptr;

//   //For profiling only 
//   // unsigned long long cycle_num[2];
//   // aie::tile tile=aie::tile::current();
//   // cycle_num[0]=tile.cycles();

//   for (unsigned im = 0; im < Tm; ++im) 
//   //chess_unroll_loop(Tm)
//   {
//     for (unsigned in = 0; in < Tn; ++in) 
//     //chess_unroll_loop(Tn)
//     {
//       //const int8 * __restrict pA1 = pA + ( im * Tk + 0) * MMUL::size_A;
//       const int8 * __restrict pB1 = pB + ( 0 * Tn + in) * MMUL::size_B;

//       aie::vector<int8, MMUL::size_A> A = readincr_v<MMUL::size_A>(matA);
//       //aie::vector<int8, MMUL::size_A> A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
//       aie::vector<int8, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

//       MMUL C; 
//       C.mul(A, B);

//       for (unsigned ik = 0; ik < Tk-1; ++ik) 
//       chess_flatten_loop
//       {
//         A = readincr_v<MMUL::size_A>(matA);
//         //A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
//         B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;
//         C.mac(A, B);
//       }
//       // auto C_vec = C.template to_vector<int8>(SHIFT);
//       // aie::store_v(pC, C_vec); pC += MMUL::size_C;
//       using VC = aie::vector<int8, MMUL::size_C>;
//       VC v = C.template to_vector<int8>(SHIFT);
//       writeincr(matC, v);
//     }
//   }
//   //For profiling only 
//   // cycle_num[1]=tile.cycles();
//   // printf("start=%lld,end=%lld,total=%lld\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);
// }

// // Output projection kernel - interleaves context vectors and applies output projection
// template <int m, int k, int n, int head_dim, int d_model, int head_idx, int num_heads, int T, int SHIFT_O>
// void output(
//   input_stream_int8* __restrict context_vector,
//   output_window_int8* __restrict output,
//   const int8 Wo[]
// ) {
//   // Get pointers
//   //const int8* __restrict pContext = (int8*)context_vector->ptr;
//   int8* __restrict pOutput = (int8*)output->ptr;
  
//   // Static buffer for combined heads
//   // static int8 combined_heads[160*64];  // Using max size, adjust as needed
  
//   // // Interleave this head's context vector into the combined heads buffer
//   // for (int t = 0; t < T; t++) {
//   //   for (int d = 0; d < head_dim; d++) {
//   //     // Place this head's output in the right position in the combined buffer
//   //     combined_heads[t*d_model + head_idx*head_dim + d] = pContext[t*head_dim + d];
//   //   }
//   // }
//   //printf("head_idx = %d\nnum_heads = %d\n", head_idx, num_heads);
  
//   // If this is the last head, perform output projection
//   if (head_idx == num_heads - 1) {
//     // // Create temporary windows for the dense kernel
//     // input_window_int8 temp_in;
//     // output_window_int8 temp_out;
//     // temp_in.ptr = combined_heads;
//     // temp_out.ptr = pOutput;
//     // dense<m, k, n, T/m, d_model/k, d_model/n, SHIFT_O, false>(&temp_in, &temp_out, Wo);

//     // tiled matrix multiplication like dense layer (combined_heads @ Wo)
//     const int Tm = T / m;
//     const int Tk = d_model / k;
//     const int Tn = d_model / n;

//     using MMUL = aie::mmul<m, k, n, int8, int8>;

//     //const int8* __restrict pA = pContext; // Matrix A is the combined heads buffer
//     const int8* __restrict pB = Wo;             // Matrix B is the output weight matrix
//     int8* __restrict pC = pOutput;              // Matrix C is the final output

//     //printf("Tm = %d\nTn = %d\n", Tm, Tn);
//     for (unsigned im = 0; im < Tm; ++im) 
//     {
//       for (unsigned in = 0; in < Tn; ++in) 
//       {
//         aie::vector<int8, MMUL::size_A> A = readincr_v<MMUL::size_A>(context_vector);
//         //const int8 * __restrict pA1 = pA + (im * Tk + 0) * MMUL::size_A;
//         const int8 * __restrict pB1 = pB + ( 0 * Tn + in) * MMUL::size_B;
        
//         //aie::vector<int8, MMUL::size_A> A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
//         aie::vector<int8, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

//         MMUL C; 
//         C.mul(A, B);

//         for (unsigned ik = 0; ik < Tk-1; ++ik) 
//         chess_flatten_loop
//         {
//           A = readincr_v<MMUL::size_A>(context_vector);
//           //A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
//           B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;
//           C.mac(A, B);
//         }
        
//         // Convert accumulator to vector, apply output shift, and store.
//         auto C_vec = C.template to_vector<int8>(SHIFT_O);
//         aie::store_v(pC, C_vec); 
//         pC += MMUL::size_C;
//       }
//     }
//     // for (int i = 0; i < 8; ++i) { // limit to 8 elements
//     //   printf("pA[%d] = %d\npB[%d] = %d\n", i, pA[i], i, pB[i]);
//     // }

//   }
// }

#endif