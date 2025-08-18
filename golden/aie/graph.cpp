
#include <adf.h>
#include "include.h"
#include <vector>
#include "model.h"

using namespace adf;

class mainGraph : public adf::graph {
private:
  graph* layers [N_LAYERS];

public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  mainGraph(){

    AIE_IN = input_plio::create(plio_128_bits, "data/x0.txt");
    AIE_OUT = output_plio::create(plio_128_bits, "data/out_sim.txt");

    #include "layer_graph.h"
  }
};

mainGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(ITERATIONS);
  mygraph.end();
  return 0;
}
