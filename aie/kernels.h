
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
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  const int8* __restrict Bbase = (const int8*)matB;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }
    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT);
      if (is_relu) v = aie::max(v, (int8)0);
      writeincr(sC, v);
    }
  }
}


template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense_bias(
  input_stream_int8 * __restrict sA,
  output_stream_int8 * __restrict sC,
  const int8 matB[],
  const int8 bias[]
) {
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  aie::set_saturation(aie::saturation_mode::saturate);

  const int8* __restrict Bbase = (const int8*)matB;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }

    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT);

      alignas(32) int8 bias_tile[m * n];
      for (unsigned rr = 0; rr < m; ++rr) {
        for (unsigned cc = 0; cc < n; ++cc) {
          bias_tile[rr * n + cc] = bias[in * n + cc];
        }
      }
      VC vb = aie::load_v<m * n>(bias_tile);
      v = aie::saturating_add(v, vb);

      if (is_relu) v = aie::max(v, (int8)0);
      writeincr(sC, v);
    }
  }
}


template <int m, int n, int Tm, int Tn, int OUT_SCALE = 127>
void softmax_rows(
  input_stream_int8 * __restrict sIn,
  output_stream_int8 * __restrict sOut,
  const int16 lut[256]
) {
  constexpr int TILE_ELEMS = m * n;
  alignas(32) int8 tile_row[Tn][TILE_ELEMS];
  alignas(32) int numer[Tn * n];

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      for (int e = 0; e < TILE_ELEMS; ++e) {
        tile_row[in][e] = readincr(sIn);
      }
    }

    for (int rr = 0; rr < m; ++rr) {
      int row_max = -128;
      for (int in = 0; in < Tn; ++in) {
        for (int cc = 0; cc < n; ++cc) {
          int v = (int)tile_row[in][rr * n + cc];
          if (v > row_max) row_max = v;
        }
      }

      int denom = 0;
      for (int in = 0; in < Tn; ++in) {
        for (int cc = 0; cc < n; ++cc) {
          int delta = (int)tile_row[in][rr * n + cc] - row_max;
          if (delta < -128) delta = -128;
          if (delta > 127)  delta = 127;
          int lut_idx = delta + 128;
          int val = (int)lut[lut_idx];
          numer[in * n + cc] = val;
          denom += val;
        }
      }
      if (denom <= 0) denom = 1;

      for (int in = 0; in < Tn; ++in) {
        for (int cc = 0; cc < n; ++cc) {
          int p = (numer[in * n + cc] * OUT_SCALE + (denom >> 1)) / denom;
          if (p < 0) p = 0;
          if (p > 127) p = 127;
          tile_row[in][rr * n + cc] = (int8)p;
        }
      }
    }

    for (int in = 0; in < Tn; ++in) {
      for (int e = 0; e < TILE_ELEMS; ++e) {
        writeincr(sOut, tile_row[in][e]);
      }
    }
  }
}


template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_S>
void scores(
  input_stream_int8 * __restrict sQ,
  input_stream_int8 * __restrict sK,
  output_stream_int8 * __restrict sS
) {
  using MMUL = aie::mmul<m, n, m, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_A>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  VB matB[Tm * Tn];

  for (unsigned i = 0; i < Tm; ++i) {
    for (unsigned j = 0; j < Tn; ++j) {
      matB[i * Tn + j] = aie::transpose(readincr_v<MMUL::size_A>(sK), m, n);
    }
  }

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tn];
    for (unsigned in = 0; in < Tn; ++in) {
      Abuf[in] = readincr_v<MMUL::size_A>(sQ);
    }
    for (unsigned jm = 0; jm < Tm; ++jm) {
      MMUL C;
      for (unsigned in = 0; in < Tn; ++in) {
        if (in == 0) C.mul(Abuf[0], matB[jm * Tn + in]);
        else         C.mac(Abuf[in], matB[jm * Tn + in]);
      }
      VC V = C.template to_vector<int8>(SHIFT_S);
      writeincr(sS, V);
    }
  }
}


template <int m, int k, int n, int Tqm, int Tkm, int Tn, int d_model, int SHIFT_S>
void scores_cross(
  input_stream_int8 * __restrict sQ,
  input_stream_int8 * __restrict sK,
  output_stream_int8 * __restrict sS
) {
  using MMUL = aie::mmul<m, n, m, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_A>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  VB matB[Tkm * Tn];

  for (unsigned i = 0; i < Tkm; ++i) {
    for (unsigned j = 0; j < Tn; ++j) {
      matB[i * Tn + j] = aie::transpose(readincr_v<MMUL::size_A>(sK), m, n);
    }
  }

  for (unsigned iq = 0; iq < Tqm; ++iq) {
    VA Abuf[Tn];
    for (unsigned in = 0; in < Tn; ++in) {
      Abuf[in] = readincr_v<MMUL::size_A>(sQ);
    }
    for (unsigned jk = 0; jk < Tkm; ++jk) {
      MMUL C;
      for (unsigned in = 0; in < Tn; ++in) {
        if (in == 0) C.mul(Abuf[0], matB[jk * Tn + in]);
        else         C.mac(Abuf[in], matB[jk * Tn + in]);
      }
      VC V = C.template to_vector<int8>(SHIFT_S);
      writeincr(sS, V);
    }
  }
}


template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT>
void context(
  input_stream_int8 * __restrict sS,
  input_stream_int8 * __restrict sV,
  output_stream_int8 * __restrict sC
) {
  using MMUL = aie::mmul<m, m, n, int8, int16>;
  using VA   = aie::vector<int8,  MMUL::size_A>;
  using VB   = aie::vector<int16, MMUL::size_B>;
  using VC   = aie::vector<int16, MMUL::size_C>;
  using VBin = aie::vector<int8, MMUL::size_B>;
  using VCout = aie::vector<int8, MMUL::size_C>;

  VB matB[Tm * Tn];

  for (unsigned im = 0; im < Tm; ++im) {
    for (unsigned in = 0; in < Tn; ++in) {
      VBin B = readincr_v<32>(sV);
      VB B16 = B.unpack();
      matB[im * Tn + in] = B16;
    }
  }

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tm];
    for (unsigned jm = 0; jm < Tm; ++jm) {
      Abuf[jm] = readincr_v<MMUL::size_A>(sS);
    }
    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      for (unsigned jm = 0; jm < Tm; ++jm) {
        if (jm == 0) C.mul(Abuf[0], matB[jm * Tn + in]);
        else         C.mac(Abuf[jm], matB[jm * Tn + in]);
      }

      VC v = C.template to_vector<int16>(SHIFT);
      VCout vout = v.pack();
      writeincr(sC, vout);
    }
  }
}


template <int m, int n, int Tqm, int Tkm, int Tn, int SHIFT>
void context_cross(
  input_stream_int8 * __restrict sS,
  input_stream_int8 * __restrict sV,
  output_stream_int8 * __restrict sC
) {
  using MMUL = aie::mmul<m, m, n, int8, int16>;
  using VA   = aie::vector<int8,  MMUL::size_A>;
  using VB   = aie::vector<int16, MMUL::size_B>;
  using VC   = aie::vector<int16, MMUL::size_C>;
  using VBin = aie::vector<int8, MMUL::size_B>;
  using VCout = aie::vector<int8, MMUL::size_C>;

  VB matB[Tkm * Tn];

  for (unsigned ik = 0; ik < Tkm; ++ik) {
    for (unsigned in = 0; in < Tn; ++in) {
      VBin B = readincr_v<32>(sV);
      VB B16 = B.unpack();
      matB[ik * Tn + in] = B16;
    }
  }

  for (unsigned iq = 0; iq < Tqm; ++iq) {
    VA Abuf[Tkm];
    for (unsigned jk = 0; jk < Tkm; ++jk) {
      Abuf[jk] = readincr_v<MMUL::size_A>(sS);
    }
    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      for (unsigned jk = 0; jk < Tkm; ++jk) {
        if (jk == 0) C.mul(Abuf[0], matB[jk * Tn + in]);
        else         C.mac(Abuf[jk], matB[jk * Tn + in]);
      }
      VC v = C.template to_vector<int16>(SHIFT);
      VCout vout = v.pack();
      writeincr(sC, vout);
    }
  }
}


template <int m, int n, int Tm, int Tn>
void concat(
  input_stream_int8 * __restrict sA,
  input_stream_int8 * __restrict sB,
  output_stream_int8 * __restrict sC
) {
  using V = aie::vector<int8, m * n>;

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      writeincr(sC, readincr_v<m * n>(sA));
    }
    for (int in = 0; in < Tn; ++in) {
      writeincr(sC, readincr_v<m * n>(sB));
    }
  }
}


template <int m, int n, int Tm, int Tn>
void resadd(
  input_stream_int8 * __restrict sA,
  input_stream_int8 * __restrict sB,
  output_stream_int8 * __restrict sC
) {
  using V = aie::vector<int8, m * n>;
  aie::set_saturation(aie::saturation_mode::saturate);

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      V vA = readincr_v<m * n>(sA);
      V vB = readincr_v<m * n>(sB);
      V vC = aie::saturating_add(vA, vB);
      writeincr(sC, vC);
    }
  }
}


template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT_O, bool is_relu, bool use_bias, bool use_residual>
void output(
  input_stream_int8* __restrict sA,
  input_stream_int8* __restrict sB,
  output_stream_int8* __restrict sO,
  const int8 Wo[],
  const int8 bias[],
  const int8 residual[]
) {
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  aie::set_saturation(aie::saturation_mode::saturate);

  const int8* __restrict Bbase = (const int8*)Wo;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk / 2; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sA);
    }
    for (unsigned ik = Tk / 2; ik < Tk; ++ik) {
      Abuf[ik] = readincr_v<MMUL::size_A>(sB);
    }

    for (unsigned in = 0; in < Tn; ++in) {
      MMUL C;
      const int8* __restrict pB = Bbase + in * MMUL::size_B;

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT_O);

      if constexpr (use_bias) {
        alignas(32) int8 bias_tile[m * n];
        for (unsigned rr = 0; rr < m; ++rr) {
          for (unsigned cc = 0; cc < n; ++cc) {
            bias_tile[rr * n + cc] = bias[in * n + cc];
          }
        }
        VC vb = aie::load_v<m * n>(bias_tile);
        v = aie::saturating_add(v, vb);
      }

      if constexpr (use_residual) {
        alignas(32) int8 residual_tile[m * n];
        for (unsigned rr = 0; rr < m; ++rr) {
          for (unsigned cc = 0; cc < n; ++cc) {
            residual_tile[rr * n + cc] = residual[in * n + cc];
          }
        }
        VC vr = aie::load_v<m * n>(residual_tile);
        v = aie::saturating_add(v, vr);
      }

      if constexpr (is_relu) {
        v = aie::max(v, (int8)0);
      }
      writeincr(sO, v);
    }
  }
}


template <int m, int n, int Tm, int Tn>
void emit_const(
  output_stream_int8 * __restrict sOut,
  const int8 data_tiled[]
) {
  constexpr int TILE_ELEMS = m * n;
  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      int base = (im * Tn + in) * TILE_ELEMS;
      for (int e = 0; e < TILE_ELEMS; ++e) {
        writeincr(sOut, data_tiled[base + e]);
      }
    }
  }
}


#endif
