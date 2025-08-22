
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
  adf::input_window_int8 * __restrict matA, 
  adf::output_window_int8 * __restrict matC,
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

// Projection kernel - projects input to Q, K, V for a specific head
template <int m, int k, int n, int head_dim, int SHIFT_Q, int SHIFT_K, int SHIFT_V>
void projection_kernel(
  input_window_int8* __restrict input,
  output_window_int8* __restrict q_head,
  output_window_int8* __restrict k_head,
  output_window_int8* __restrict v_head,
  const int8 Wq[],
  const int8 Wk[],
  const int8 Wv[],
  const int head_idx,
  const int T,
  const int d_model
) {
  // Get pointers
  const int8* __restrict pInput = (int8*)input->ptr;
  int8* __restrict pQ = (int8*)q_head->ptr;
  int8* __restrict pK = (int8*)k_head->ptr;
  int8* __restrict pV = (int8*)v_head->ptr;
  
  // Extract head-specific weights
  const int8* Wq_head = Wq + head_idx * head_dim * d_model;
  const int8* Wk_head = Wk + head_idx * head_dim * d_model;
  const int8* Wv_head = Wv + head_idx * head_dim * d_model;
  
  // Project input to Q, K, V for this specific head using dense
  input_window_int8 temp_in;
  output_window_int8 temp_q_out, temp_k_out, temp_v_out;
  
  // Set up windows
  temp_in.ptr = (int8*)pInput;
  temp_q_out.ptr = pQ;
  temp_k_out.ptr = pK;
  temp_v_out.ptr = pV;
  
  // Call dense for Q, K, V projections with template parameters
  dense<m, k, n, T/m, d_model/k, head_dim/n, SHIFT_Q, false>(&temp_in, &temp_q_out, Wq_head);
  dense<m, k, n, T/m, d_model/k, head_dim/n, SHIFT_K, false>(&temp_in, &temp_k_out, Wk_head);
  dense<m, k, n, T/m, d_model/k, head_dim/n, SHIFT_V, false>(&temp_in, &temp_v_out, Wv_head);
}

// Attention kernel - calculates attention scores and applies them to values
template <int m, int k, int n, int head_dim, int SHIFT_S, int SHIFT_C>
void attention_kernel(
  input_window_int8* __restrict q_head,
  input_window_int8* __restrict k_head,
  input_window_int8* __restrict v_head,
  output_window_int8* __restrict context_vector,
  const int T
) {
  // Get pointers
  const int8* __restrict pQ = (int8*)q_head->ptr;
  const int8* __restrict pK = (int8*)k_head->ptr;
  const int8* __restrict pV = (int8*)v_head->ptr;
  int8* __restrict pContext = (int8*)context_vector->ptr;
  
  // Temporary buffer for attention scores (T×T)
  int8 scores[150*150];  // Using max size, adjust as needed
  
  // Calculate attention scores (Q @ K^T) - manual implementation needed for transpose
  for (int q_pos = 0; q_pos < T; q_pos++) {
    for (int k_pos = 0; k_pos < T; k_pos++) {
      // Calculate dot product
      int32_t score = 0;
      for (int d = 0; d < head_dim; d++) {
        score += (int32_t)pQ[q_pos*head_dim + d] * (int32_t)pK[k_pos*head_dim + d];
      }
      
      // Apply shift for quantization
      scores[q_pos*T + k_pos] = (score >> SHIFT_S);
    }
  }
  
  // Apply attention scores to values using dense function
  input_window_int8 scores_window;
  output_window_int8 context_window;
  
  // Set up window pointers
  scores_window.ptr = (int8*)scores;
  context_window.ptr = (int8*)pContext;
  
  // Use dense for attention @ V multiplication with template parameters
  dense<m, k, n, T/m, T/k, head_dim/n, SHIFT_C, false>(&scores_window, &context_window, (int8*)pV);
}

// Output projection kernel - interleaves context vectors and applies output projection
template <int m, int k, int n, int head_dim, int d_model, int SHIFT_O>
void output_kernel(
  input_window_int8* __restrict context_vector,
  output_window_int8* __restrict output,
  const int8 Wo[],
  const int head_idx,
  const int num_heads,
  const int T
) {
  // Get pointers
  const int8* __restrict pContext = (int8*)context_vector->ptr;
  int8* __restrict pOutput = (int8*)output->ptr;
  
  // Static buffer for combined heads
  static int8 combined_heads[150*64];  // Using max size, adjust as needed
  
  // Interleave this head's context vector into the combined heads buffer
  for (int t = 0; t < T; t++) {
    for (int d = 0; d < head_dim; d++) {
      // Place this head's output in the right position in the combined buffer
      combined_heads[t*d_model + head_idx*head_dim + d] = pContext[t*head_dim + d];
    }
  }
  
  // If this is the last head, perform output projection
  if (head_idx == num_heads - 1) {
    // Create temporary windows for the dense kernel
    input_window_int8 temp_in;
    output_window_int8 temp_out;
    
    temp_in.ptr = combined_heads;
    temp_out.ptr = pOutput;
    
    // Call dense for output projection with template parameters
    dense<m, k, n, T/m, d_model/k, d_model/n, SHIFT_O, false>(&temp_in, &temp_out, Wo);
  }
}

// Residual addition kernel - adds two int8 tensors with saturation
template <int m, int n, int Tm, int Tn>
void residual_add_kernel(
  input_window_int8* __restrict input1,
  input_window_int8* __restrict input2,
  output_window_int8* __restrict output
) {
  // Get pointers
  const int8* __restrict pInput1 = (int8*)input1->ptr;
  const int8* __restrict pInput2 = (int8*)input2->ptr;
  int8* __restrict pOutput = (int8*)output->ptr;
  
  // For profiling only
  unsigned long long cycle_num[2];
  aie::tile tile = aie::tile::current();
  cycle_num[0] = tile.cycles();
  
  // Process all elements
  const int total_elements = Tm * m * Tn * n;
  
  for (int i = 0; i < total_elements; i++) {
    // Add with int16 precision to avoid overflow
    int16_t sum = (int16_t)pInput1[i] + (int16_t)pInput2[i];
    
    // Saturate to int8 range
    if (sum > 127) sum = 127;
    if (sum < -128) sum = -128;
    
    // Store result
    pOutput[i] = (int8_t)sum;
  }
  
  // For profiling only
  cycle_num[1] = tile.cycles();
  printf("residual_add: start=%lld,end=%lld,total=%lld\n", 
         cycle_num[0], cycle_num[1], cycle_num[1] - cycle_num[0]);
}
