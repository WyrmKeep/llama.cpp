# MMQ tile sweep (micro-benchmark, not for merge)

This folder holds a micro-benchmark for the tile configurations of the MMQ kernel (`ggml/src/ggml-cuda/mmq.cuh`) on one GPU.
It is a tool of the decision-model-kernels project. It is not part of llama.cpp and has no CMake target.

It has three parts:

| File | Use |
|---|---|
| `ggml/src/ggml-cuda/mmq-config-sweep.cuh` | A table with one entry. The macros `SWEEP_TYPE`, `SWEEP_NT`, `SWEEP_OCC`, `SWEEP_I`, `SWEEP_J`, `SWEEP_STREAMK` and `SWEEP_FALLBACK` set the entry. `mmq.cuh` includes it when `SWEEP_J` is defined. |
| `ggml/src/ggml-cuda/mmq.cuh` | Four changes: the extern variables `g_sweep_*`, the include above, the diagnostics, and the runtime choice of the stream-k grid (`g_sweep_policy`). With `g_sweep_policy` 0 the launch is the same as in the base commit. |
| `tools/dmk-mmq-sweep/` | `sweep_variant.cu` is one tile configuration. `gen_variants.py` lists the configurations. `build_sweep.sh` compiles each one in a builder container and links them with the harness `mmq_sweep.cu`. |

The harness runs `launch_mul_mat_q` of each configuration for the weight shapes (type, rows, K) and the numbers of tokens that you give. It uses random weights and activations. It compares each output bit by bit with the output of the base configuration and writes one TSV row for each measurement.

```
tools/dmk-mmq-sweep/build_sweep.sh <this tree> r1 <out dir>      # builds <out dir>/llama
<out dir>/llama sweep --shapes q4_0:9216:2560,q8_0:4096:2560 --ns 64,129,1173 --policies 0,1,4
```

The grid rules of `--policies` are: 0 the rule of the base commit, 1 one block for each tile (no K split), 2 persistent blocks, `nsm` times the blocks for each SM, 3 persistent blocks, `nsm` blocks, 4 the rule of the base commit with a wave of `nsm` times the blocks for each SM.

Time the program only on a GPU that no other job uses.
