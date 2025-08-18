#ifndef MHA_GRAPH_H
#define MHA_GRAPH_H

#include <adf.h>
#include "kernels.h"

using namespace adf;

// MHAGraph as an AIE subgraph
template <int m, int k, int n, int num_heads, int d_model, 
          int SHIFT_Q, int SHIFT_K, int SHIFT_V, int SHIFT_O, int SHIFT_S, int SHIFT_C>
class MHAGraph : public graph {
private:
    // Calculate head dimension
    static constexpr int head_dim = d_model / num_heads;
    
    // Kernel instances
    kernel projection_kernel_instance;
    kernel attention_kernel_instance;
    kernel output_kernel_instance;
    
public:
    // Graph ports
    port<input> in;
    port<output> out;
    
    MHAGraph(const int8_t* Wq, const int8_t* Wk, const int8_t* Wv, const int8_t* Wo) {
        // Create kernels
        projection_kernel_instance = kernel::create_object<
            projection_kernel<m, k, n, head_dim, SHIFT_Q, SHIFT_K, SHIFT_V>
        >();
        
        attention_kernel_instance = kernel::create_object<
            attention_kernel<m, k, n, head_dim, SHIFT_S, SHIFT_C>
        >();
        
        output_kernel_instance = kernel::create_object<
            output_kernel<m, k, n, head_dim, d_model, SHIFT_O>
        >();
        
        // Set kernel sources
        source(projection_kernel_instance) = "kernels.h";
        source(attention_kernel_instance) = "kernels.h";
        source(output_kernel_instance) = "kernels.h";
        
        // Set runtime ratios
        runtime<ratio>(projection_kernel_instance) = 0.1;
        runtime<ratio>(attention_kernel_instance) = 0.1;
        runtime<ratio>(output_kernel_instance) = 0.1;
        
        // Set kernel arguments
        projection_kernel_instance.set_arg(0, Wq);
        projection_kernel_instance.set_arg(1, Wk);
        projection_kernel_instance.set_arg(2, Wv);
        
        output_kernel_instance.set_arg(0, Wo);
        
        // Calculate window sizes based on template parameters
        constexpr int T = m * (d_model/k/m);  // Sequence length based on tiling parameters
        
        // Connect graph input to projection kernel
        connect<window<d_model * T * sizeof(int8)>>(in, projection_kernel_instance.in[0]);
        
        // Connect projection outputs to attention inputs
        connect<window<head_dim * T * sizeof(int8)>>(projection_kernel_instance.out[0], attention_kernel_instance.in[0]); // Q
        connect<window<head_dim * T * sizeof(int8)>>(projection_kernel_instance.out[1], attention_kernel_instance.in[1]); // K
        connect<window<head_dim * T * sizeof(int8)>>(projection_kernel_instance.out[2], attention_kernel_instance.in[2]); // V
        
        // Connect attention output to output kernel
        connect<window<head_dim * T * sizeof(int8)>>(attention_kernel_instance.out[0], output_kernel_instance.in[0]);
        
        // Connect output kernel to graph output
        connect<window<d_model * T * sizeof(int8)>>(output_kernel_instance.out[0], out);
    }
};

#endif // MHA_GRAPH_H