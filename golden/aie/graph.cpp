
#include <adf.h>
#include "include.h"
#include <vector>

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

  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();

  // First input -> first output (graph latency)
  adf::event::handle h_latency =
  adf::event::start_profiling(mygraph.AIE_IN, mygraph.AIE_OUT,
                                adf::event::io_stream_start_difference_cycles);

  mygraph.run(ITERATIONS);
  mygraph.end();

  long long latency_cycles   = adf::event::read_profiling(h_latency);
  adf::event::stop_profiling(h_latency);

  const int AIE_clock_Hz = 1.2e9;
  printf("\n\n\n--------GRAPH LATENCY    (First in  -> First out) : %lld cycles, %.1f ns\n\n\n"  , latency_cycles, (1e9 * latency_cycles) / AIE_clock_Hz);

  return 0;
}