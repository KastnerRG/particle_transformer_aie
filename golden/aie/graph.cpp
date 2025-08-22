#include <adf.h>
#include "include.h"
#include <vector>
#include "model.h"
#include "dense_graph.h"
#include "mha_graph.h"
#include "residual_graph.h"
#include "weights.h"

using namespace adf;

class mainGraph : public adf::graph {
private:
  // Dense graph for layer 0
  DenseGraph<2, 8, 8, 75, 4, 8, 3, true> dense_graph_0 { k0 };
  // MHA graph for layer 1
  MHAGraph<2, 8, 8, 4, 64, 11, 10, 10, 9, 6, 12> mha_graph_1 { Wq1, Wk1, Wv1, Wo1 };
  // Residual graph for layer 2 with residual from layer 0
  ResidualGraph<2, 8, 75, 8> residual_graph_2;
  // Dense graph for layer 3
  DenseGraph<2, 8, 8, 75, 8, 8, 3, true> dense_graph_3 { k3 };
  // Dense graph for layer 4
  DenseGraph<2, 8, 8, 75, 8, 8, 3, true> dense_graph_4 { k4 };
  // Residual graph for layer 5 with residual from layer 2
  ResidualGraph<2, 8, 75, 8> residual_graph_5;
  // MHA graph for layer 6
  MHAGraph<2, 8, 8, 4, 64, 11, 11, 11, 9, 7, 11> mha_graph_6 { Wq6, Wk6, Wv6, Wo6 };
  // Residual graph for layer 7 with residual from layer 5
  ResidualGraph<2, 8, 75, 8> residual_graph_7;
  // Dense graph for layer 8
  DenseGraph<2, 8, 8, 75, 8, 8, 3, true> dense_graph_8 { k8 };
  // Dense graph for layer 9
  DenseGraph<2, 8, 8, 75, 8, 8, 3, true> dense_graph_9 { k9 };
  // Residual graph for layer 10 with residual from layer 7
  ResidualGraph<2, 8, 75, 8> residual_graph_10;
  // Dense graph for layer 11
  DenseGraph<2, 8, 8, 75, 8, 8, 3, true> dense_graph_11 { k11 };
  // Dense graph for layer 12
  DenseGraph<2, 8, 8, 75, 8, 1, 3, false> dense_graph_12 { k12 };
  graph* layers [N_LAYERS];

public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  mainGraph(){

    AIE_IN = input_plio::create(plio_128_bits, "data/x0.txt");
    AIE_OUT = output_plio::create(plio_128_bits, "data/out_sim.txt");

    // Assign layer 0 pointer
    layers[0] = &dense_graph_0;
    // Connect input to first layer
    connect<window< 4800>>(AIE_IN.out[0], dense_graph_0.in);

    // Assign layer 1 pointer
    layers[1] = &mha_graph_1;
    // Connect layer 0 to layer 1
    connect<window< 9600>>(layers[0]->out[0], mha_graph_1.in);

    // Assign layer 2 pointer
    layers[2] = &residual_graph_2;
    // Connect layer 1 to layer 2
    connect<window< 9600>>(layers[1]->out[0], residual_graph_2.in1);
    // Connect residual from layer 0 to layer 2
    connect<window< 9600>>(layers[0]->out[0], residual_graph_2.in2);

    // Assign layer 3 pointer
    layers[3] = &dense_graph_3;
    // Connect layer 2 to layer 3
    connect<window< 9600>>(layers[2]->out[0], dense_graph_3.in);

    // Assign layer 4 pointer
    layers[4] = &dense_graph_4;
    // Connect layer 3 to layer 4
    connect<window< 9600>>(layers[3]->out[0], dense_graph_4.in);

    // Assign layer 5 pointer
    layers[5] = &residual_graph_5;
    // Connect layer 4 to layer 5
    connect<window< 9600>>(layers[4]->out[0], residual_graph_5.in1);
    // Connect residual from layer 2 to layer 5
    connect<window< 9600>>(layers[2]->out[0], residual_graph_5.in2);

    // Assign layer 6 pointer
    layers[6] = &mha_graph_6;
    // Connect layer 5 to layer 6
    connect<window< 9600>>(layers[5]->out[0], mha_graph_6.in);

    // Assign layer 7 pointer
    layers[7] = &residual_graph_7;
    // Connect layer 6 to layer 7
    connect<window< 9600>>(layers[6]->out[0], residual_graph_7.in1);
    // Connect residual from layer 5 to layer 7
    connect<window< 9600>>(layers[5]->out[0], residual_graph_7.in2);

    // Assign layer 8 pointer
    layers[8] = &dense_graph_8;
    // Connect layer 7 to layer 8
    connect<window< 9600>>(layers[7]->out[0], dense_graph_8.in);

    // Assign layer 9 pointer
    layers[9] = &dense_graph_9;
    // Connect layer 8 to layer 9
    connect<window< 9600>>(layers[8]->out[0], dense_graph_9.in);

    // Assign layer 10 pointer
    layers[10] = &residual_graph_10;
    // Connect layer 9 to layer 10
    connect<window< 9600>>(layers[9]->out[0], residual_graph_10.in1);
    // Connect residual from layer 7 to layer 10
    connect<window< 9600>>(layers[7]->out[0], residual_graph_10.in2);

    // Assign layer 11 pointer
    layers[11] = &dense_graph_11;
    // Connect layer 10 to layer 11
    connect<window< 9600>>(layers[10]->out[0], dense_graph_11.in);

    // Assign layer 12 pointer
    layers[12] = &dense_graph_12;
    // Connect layer 11 to layer 12
    connect<window< 9600>>(layers[11]->out[0], dense_graph_12.in);

    // Connect last layer to output
    connect<window< 1200>>(layers[12]->out[0], AIE_OUT.in[0]);
  }
};

mainGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(ITERATIONS);
  mygraph.end();
  return 0;
}
