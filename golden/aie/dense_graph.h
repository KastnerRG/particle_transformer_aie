#ifndef DENSE_GRAPH_H
#define DENSE_GRAPH_H

#include <adf.h>
#include "kernels.h"

using namespace adf;

// Simple wrapper graph for dense kernel
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu = false>
class DenseGraph : public graph {
public:
    port<input> in;
    port<output> out;
    kernel dense_kernel_instance;
    
    DenseGraph(const int8_t* W) {
        // Create dense kernel with weights
        dense_kernel_instance = kernel::create_object<dense<m, k, n, Tm, Tk, Tn, SHIFT, is_relu>>();
        
        // Set kernel source and runtime
        source(dense_kernel_instance) = "kernels.h";
        runtime<ratio>(dense_kernel_instance) = 1.0;
        
        // Set weight matrix
        dense_kernel_instance.set_arg(2, W);
        
        // Connect ports - account for tiled dimensions
        connect<window<Tm*m*Tk*k*sizeof(int8_t)>>(in, dense_kernel_instance.in[0]);
        connect<window<Tm*m*Tn*n*sizeof(int8_t)>>(dense_kernel_instance.out[0], out);
    }
};

#endif // DENSE_GRAPH_H
