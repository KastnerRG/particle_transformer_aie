
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"


template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, int SCALE, bool is_relu, bool has_bias>
void dense(
  input_stream_int8 * __restrict sA,
  output_stream_int8 * __restrict sC,
  const int8 matB [],
  const int8 bias []
  ) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);

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

      aie::vector<int32, MMUL::size_C> raw_v = C.template to_vector<int32>(0);
      if constexpr (has_bias) {
        for (unsigned rr = 0; rr < m; ++rr) {
          for (unsigned cc = 0; cc < n; ++cc) {
            const int idx = rr * n + cc;
            raw_v[idx] = raw_v[idx] + (int32)bias[in * n + cc];
          }
        }
      }
      auto scaled_v = aie::mul(raw_v, SCALE);
      VC v = scaled_v.template to_vector<int8>(SHIFT);
      if (is_relu) v = aie::max(v, (int8)0);
      writeincr(sC, v);
    }
  }
}

// (Q @ K^T):  (T, d_model) @ (T, d_model)^T -> (T, T)
// m=4, k=8, n=8, T=160, d_model=64, Tm(rows)=160/m=40, Tn(columns)=64/n=8
template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_IN, int SCALE_IN, int SHIFT_S, int SCALE, bool enable_softmax>
void scores(
  input_stream_int8 * __restrict sQ, // adf::input_buffer<int8, adf::extents<T*d_model>> & sQ,
  input_stream_int8 * __restrict sK, // adf::input_buffer<int8, adf::extents<T*d_model>> & sK,
  output_stream_int8 * __restrict sS
) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);

  using MMUL = aie::mmul<m, n, m, int8, int8>; // 4x8x4
  using VA   = aie::vector<int8, MMUL::size_A>; // 4x8
  using VB   = aie::vector<int8, MMUL::size_A>; // 8x4
  using VC32 = aie::vector<int32, MMUL::size_C>; // 4x4 int32
  using VC8  = aie::vector<int8, MMUL::size_C>;  // 4x4 int8

  VB matB[Tm*Tn]; //store all of matB in mem

  // IntSoftmax polynomial parameters (aligned with NumPy golden path).
  // softmax scaling factor: s = SCALE_IN / 2^SHIFT_IN
  constexpr int N_EXP = 30;
  constexpr int SOFTMAX_BIT = 7;
  const int shift_in_eff = (SHIFT_IN < 0) ? 0 : ((SHIFT_IN > 30) ? 30 : SHIFT_IN);
  const int64 scale_in_eff = (SCALE_IN <= 0) ? 1 : (int64)SCALE_IN;
  const int64 pow2_shift = (int64)1 << shift_in_eff;

  // x0_int = floor(-0.6931 / s) = floor(-0.6931 * 2^SHIFT_IN / SCALE_IN)
  const long double x0_ratio = (6931.0L * (long double)pow2_shift) / (10000.0L * (long double)scale_in_eff);
  const int X0_INT = -(int)(x0_ratio + 0.999999999L);  // -ceil(x0_ratio)

  // B_INT = floor((0.96963238 / 0.35815147) / s)
  const long double b_ratio = (96963238.0L * (long double)pow2_shift) / (35815147.0L * (long double)scale_in_eff);
  const int B_INT = (int)b_ratio;

  // C_INT = floor((1.0 / 0.35815147) / s^2)
  const long double c_ratio = (100000000.0L * (long double)pow2_shift * (long double)pow2_shift)
                            / (35815147.0L * (long double)scale_in_eff * (long double)scale_in_eff);
  const int C_INT = (int)c_ratio;

  auto floor_div = [](int a, int b) -> int {
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r > 0) != (b > 0))) {
      --q;
    }
    return q;
  };

  for (unsigned i = 0; i < Tm; ++i) { // rows
    for (unsigned j = 0; j < Tn; ++j) { // columns
      matB[i*Tn+j] = aie::transpose(readincr_v<MMUL::size_A>(sK), m, n);
    }
  }
  
  // Row-wise fused scores with optional integer softmax.
  for (unsigned im = 0; im < Tm; ++im) {   // rows of Q
    VA Abuf[Tn]; // row of tiles

    for (unsigned in = 0; in < Tn; ++in) { // columns of Q
      Abuf[in] = readincr_v<MMUL::size_A>(sQ);
    }

    if constexpr (enable_softmax) {
      int32 row_scores[m][T];
      int8 row_probs[m][T];

      // Build full int32 score rows for this 4-row stripe.
      for (unsigned jm = 0; jm < Tm; ++jm) { // rows of K
        MMUL C;
        for (unsigned in = 0; in < Tn; ++in) { // columns of K
          if (in == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
          else         C.mac(Abuf[in], matB[jm*Tn+in]);
        }
        VC32 V = C.template to_vector<int32>(0);

        const unsigned base_col = jm * m;
        for (unsigned r = 0; r < m; ++r) {
          for (unsigned c = 0; c < m; ++c) {
            row_scores[r][base_col + c] = V[r * m + c];
          }
        }
      }

      // Row-wise integer softmax on int32 score rows.
      for (unsigned r = 0; r < m; ++r) {
        int32 row_max = row_scores[r][0];
        for (unsigned col = 1; col < T; ++col) {
          if (row_scores[r][col] > row_max) row_max = row_scores[r][col];
        }

        int64 exp_vals[T];
        int64 exp_sum = 0;

        for (unsigned col = 0; col < T; ++col) {
          int x = row_scores[r][col] - row_max;
          const int min_x = N_EXP * X0_INT;
          if (x < min_x) x = min_x;

          const int q = floor_div(x, X0_INT);
          const int rem = x - X0_INT * q;

          int64 z = (int64)rem * (int64)(rem + B_INT) + (int64)C_INT;
          int shift = N_EXP - q;
          if (shift < 0) shift = 0;
          if (shift > 62) shift = 62;

          int64 exp_int = z << shift;
          if (exp_int < 0) exp_int = 0;

          exp_vals[col] = exp_int;
          exp_sum += exp_int;
        }

        if (exp_sum <= 0) exp_sum = 1;
        const uint64 factor = ((uint64)1 << 32) / (uint64)exp_sum;

        for (unsigned col = 0; col < T; ++col) {
          uint64 p = ((uint64)exp_vals[col] * factor) >> (32 - SOFTMAX_BIT);
          if (p > 127) p = 127;
          row_probs[r][col] = (int8)p;
        }
      }

      // Emit in original 4x4 tile order expected by context.
      for (unsigned jm = 0; jm < Tm; ++jm) {
        VC8 out;
        const unsigned base_col = jm * m;
        for (unsigned r = 0; r < m; ++r) {
          for (unsigned c = 0; c < m; ++c) {
            out[r * m + c] = row_probs[r][base_col + c];
          }
        }
        writeincr(sS, out);
      }
    } else {
      // Legacy path: emit quantized raw scores directly without softmax.
      for (unsigned jm = 0; jm < Tm; ++jm) { // rows of K
        MMUL C;
        for (unsigned in = 0; in < Tn; ++in) { // columns of K
          if (in == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
          else         C.mac(Abuf[in], matB[jm*Tn+in]);
        }
        aie::vector<int32, MMUL::size_C> raw_v = C.template to_vector<int32>(0);
        auto scaled_v = aie::mul(raw_v, SCALE);
        VC8 v = scaled_v.template to_vector<int8>(SHIFT_S);
        writeincr(sS, v);
      }
    }
  }
}

// (scores @ V)  (T,T) @ (T,d_model) -> (T,d_model) mxk
// Tm = 160/4 = 40, Tk = 160/4 = 40, Tn = 64/8 = 8
// 160 x 160 x 64 tiled with 4 x 4 x 8
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, int SCALE>
void context(
  input_stream_int8 * __restrict sS,
  input_stream_int8 * __restrict sV,
  output_stream_int8 * __restrict sC
) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);

  using MMUL = aie::mmul<m, m, n, int8, int16>; // 4x4x8 -> 4x8
  using VA   = aie::vector<int8,  MMUL::size_A>; // 4x4 (int8)
  using VB   = aie::vector<int16, MMUL::size_B>; // 4x8 (int16)
  using VC   = aie::vector<int16, MMUL::size_C>; // 4x8 (int16)

  using VBin = aie::vector<int8, MMUL::size_B>; // 4x8 (int8)
  using VCout = aie::vector<int8, MMUL::size_C>; // 4x8 (int8)

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

      aie::vector<int32, MMUL::size_C> raw_v = C.template to_vector<int32>(0);
      auto scaled_v = aie::mul(raw_v, SCALE);
      VCout v = scaled_v.template to_vector<int8>(SHIFT);
      writeincr(sC, v);
    }
  }
}

// Concatenate two (m*Tm) x (n*Tn) int8 matrices.
template <int m, int n, int Tm, int Tn>
void concat(
  input_stream_int8 * __restrict sA,
  input_stream_int8 * __restrict sB,
  output_stream_int8 * __restrict sC
) {
  using V = aie::vector<int8, m*n>;
 

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      writeincr(sC, readincr_v<m*n>(sA));
    }
    for (int in = 0; in < Tn; ++in) {
      writeincr(sC, readincr_v<m*n>(sB));
    }
  }
}

// Add two (m*Tm) x (n*Tn) int8 matrices element-wise with saturation.
template <int m, int n, int Tm, int Tn>
void resadd(
  input_stream_int8 * __restrict sA,
  input_stream_int8 * __restrict sB,
  output_stream_int8 * __restrict sC
) {
  using V = aie::vector<int8, m*n>;

  // Explicitly set saturation mode
  aie::set_saturation(aie::saturation_mode::saturate);

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      V vA = readincr_v<m*n>(sA);
      V vB = readincr_v<m*n>(sB);
      V vC = aie::saturating_add(vA, vB);  // saturating addition
      writeincr(sC, vC);
    }
  }
}

// (context @ Wo)  (T,d_model) @ (d_model,d_model) -> (T,d_model)
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT_O, int SCALE, bool has_bias>
void output(
  input_stream_int8* __restrict sA,
  input_stream_int8* __restrict sB,
  output_stream_int8* __restrict sO,
  const int8 Wo[],
  const int8 bias[]
) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);

  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  const int8* __restrict Bbase = (const int8*)Wo;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk/2; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }
    for (unsigned ik = Tk/2; ik < Tk; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sB);
    }

    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      aie::vector<int32, MMUL::size_C> raw_v = C.template to_vector<int32>(0);
      if constexpr (has_bias) {
        for (unsigned rr = 0; rr < m; ++rr) {
          for (unsigned cc = 0; cc < n; ++cc) {
            const int idx = rr * n + cc;
            raw_v[idx] = raw_v[idx] + (int32)bias[in * n + cc];
          }
        }
      }
      auto scaled_v = aie::mul(raw_v, SCALE);
      VC v = scaled_v.template to_vector<int8>(SHIFT_O);
      v = aie::max(v, (int8)0);
      writeincr(sO, v);
    }
  }
}

// Dense followed by row-wise integer softmax.
// (x @ W) -> int8 dense tiles, then softmax across each full output row.
// The softmax bitwidth is fixed to 7 to match the golden NumPy path.
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT_IN, int SCALE_IN, bool has_bias>
void dense_softmax(
  input_stream_int8 * __restrict sA,
  output_stream_int8 * __restrict sC,
  const int8 matB [],
  const int8 bias []
) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);

  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC32 = aie::vector<int32, MMUL::size_C>;
  using VC8  = aie::vector<int8, MMUL::size_C>;

  constexpr int SOFTMAX_BIT = 7;
  constexpr int N_EXP = 30;

  // softmax scaling factor: s = SCALE_IN / 2^SHIFT_IN
  const int shift_in_eff = (SHIFT_IN < 0) ? 0 : ((SHIFT_IN > 30) ? 30 : SHIFT_IN);
  const int64 scale_in_eff = (SCALE_IN <= 0) ? 1 : (int64)SCALE_IN;
  const int64 pow2_shift = (int64)1 << shift_in_eff;

  // x0_int = floor(-0.6931 / s) = floor(-0.6931 * 2^SHIFT_IN / SCALE_IN)
  const long double x0_ratio = (6931.0L * (long double)pow2_shift) / (10000.0L * (long double)scale_in_eff);
  const int X0_INT = -(int)(x0_ratio + 0.999999999L);  // -ceil(x0_ratio)

  // B_INT = floor((0.96963238 / 0.35815147) / s)
  const long double b_ratio = (96963238.0L * (long double)pow2_shift) / (35815147.0L * (long double)scale_in_eff);
  const int B_INT = (int)b_ratio;

  // C_INT = floor((1.0 / 0.35815147) / s^2)
  const long double c_ratio = (100000000.0L * (long double)pow2_shift * (long double)pow2_shift)
                            / (35815147.0L * (long double)scale_in_eff * (long double)scale_in_eff);
  const int C_INT = (int)c_ratio;

  auto floor_div = [](int a, int b) -> int {
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r > 0) != (b > 0))) {
      --q;
    }
    return q;
  };

  const int8* __restrict Bbase = (const int8*)matB;
  const unsigned strideB_perK  = MMUL::size_B * Tn;
  constexpr int ROW_COLS = Tn * n;

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }

    int32 row_scores[m][ROW_COLS];
    int8 row_probs[m][ROW_COLS];

    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC32 V = C.template to_vector<int32>(0);
      const unsigned base_col = in * n;
      for (unsigned r = 0; r < m; ++r) {
        for (unsigned c = 0; c < n; ++c) {
          int32 v = V[r * n + c];
          if constexpr (has_bias) {
            v += (int32)bias[base_col + c];
          }
          row_scores[r][base_col + c] = v;
        }
      }
    }

    for (unsigned r = 0; r < m; ++r) {
      int32 row_max = row_scores[r][0];
      for (unsigned col = 1; col < ROW_COLS; ++col) {
        if (row_scores[r][col] > row_max) row_max = row_scores[r][col];
      }

      int64 exp_vals[ROW_COLS];
      int64 exp_sum = 0;

      for (unsigned col = 0; col < ROW_COLS; ++col) {
        int x = row_scores[r][col] - row_max;
        const int min_x = N_EXP * X0_INT;
        if (x < min_x) x = min_x;

        const int q = floor_div(x, X0_INT);
        const int rem = x - X0_INT * q;

        int64 z = (int64)rem * (int64)(rem + B_INT) + (int64)C_INT;
        int shift = N_EXP - q;
        if (shift < 0) shift = 0;
        if (shift > 62) shift = 62;

        int64 exp_int = z << shift;
        if (exp_int < 0) exp_int = 0;

        exp_vals[col] = exp_int;
        exp_sum += exp_int;
      }

      if (exp_sum <= 0) exp_sum = 1;
      const uint64 factor = ((uint64)1 << 32) / (uint64)exp_sum;

      for (unsigned col = 0; col < ROW_COLS; ++col) {
        uint64 p = ((uint64)exp_vals[col] * factor) >> (32 - SOFTMAX_BIT);
        if (p > 127) p = 127;
        row_probs[r][col] = (int8)p;
      }
    }

    for (unsigned in = 0; in < Tn; ++in) {
      VC8 out;
      const unsigned base_col = in * n;
      for (unsigned r = 0; r < m; ++r) {
        for (unsigned c = 0; c < n; ++c) {
          out[r * n + c] = row_probs[r][base_col + c];
        }
      }
      writeincr(sC, out);
    }
  }
}

#endif