
#include <adf.h>
#include "include.h"
#include <vector>
#include "model.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel layers [N_LAYERS];

public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  simpleGraph(){

    AIE_IN = input_plio::create(plio_128_bits, "data/x0.txt");
    AIE_OUT = output_plio::create(plio_128_bits, "data/out_sim.txt");

    #include "layer_graph.h"

    for (int i = 0; i < N_LAYERS; i++) {
      source(layers[i]) = "model.cc";
      runtime<ratio>(layers[i]) = 1.0;
    }
  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(ITERATIONS);
  mygraph.end();
  return 0;
}
