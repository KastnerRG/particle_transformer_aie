#ifndef RESIDUAL_GRAPH_H
#define RESIDUAL_GRAPH_H

#include <adf.h>
#include "kernels.h"

using namespace adf;

// Simple wrapper graph for residual addition kernel
template <int m, int n, int Tm, int Tn>
class ResidualGraph : public graph {
public:
    port<input> in1;   // Main input
    port<input> in2;   // Residual/skip connection input
    port<output> out;  // Output after residual addition
    kernel residual_kernel_instance;
    
    ResidualGraph() {
        // Create residual kernel
        residual_kernel_instance = kernel::create_object<residual_add_kernel<m, n, Tm, Tn>>();
        
        // Set kernel source and runtime
        source(residual_kernel_instance) = "kernels.h";
        runtime<ratio>(residual_kernel_instance) = 1.0;
        
        // Connect ports - account for tiled dimensions
        connect<window<Tm*m*Tn*n*sizeof(int8_t)>>(in1, residual_kernel_instance.in[0]);
        connect<window<Tm*m*Tn*n*sizeof(int8_t)>>(in2, residual_kernel_instance.in[1]);
        connect<window<Tm*m*Tn*n*sizeof(int8_t)>>(residual_kernel_instance.out[0], out);
    }
};

#endif // RESIDUAL_GRAPH_H
