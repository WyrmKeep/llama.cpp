// Sweep-only config: one entry, selected by -DSWEEP_TYPE -DSWEEP_J -DSWEEP_NT -DSWEEP_OCC -DSWEEP_I -DSWEEP_STREAMK -DSWEEP_FALLBACK.
// Every other (type, J, fallback) key returns GGML_TYPE_COUNT, so its kernel instantiation compiles to nothing.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_ampere(ggml_type type, int J, bool fallback) {
    CASE(SWEEP_TYPE, SWEEP_NT, SWEEP_OCC, SWEEP_I, SWEEP_J, SWEEP_LAYOUT, MMQ_ITER_K, SWEEP_STREAMK, SWEEP_FALLBACK);
    return ggml_cuda_mmq_config(GGML_TYPE_COUNT, 256, 1, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}
