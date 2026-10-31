// sweep_variant.cu: one MMQ tile configuration, compiled once for each variant.
//
// The variant is chosen with these macros (see gen_variants.py and build_sweep.sh):
//   SWEEP_TYPE      ggml_type of the weights (GGML_TYPE_Q4_0 or GGML_TYPE_Q8_0)
//   SWEEP_NT        threads for each CUDA block
//   SWEEP_OCC       occupancy of __launch_bounds__ (minimum blocks for each SM)
//   SWEEP_I, SWEEP_J  tile size in the weight rows and in the tokens
//   SWEEP_STREAMK   0 or 1
//   SWEEP_FALLBACK  0 or 1 (1 is for a number of weight rows that is not a multiple of 128)
//   SWEEP_NAME      suffix of the two functions that this file defines
//   SWEEP_LAYOUT    shared memory layout of the weights (GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0)
//   SWEEP_ABL       timing ablation of the loads of the k loop (0 none, 1 no weight load, 2 no activation load, 3 neither); default 0
//
// The file includes the MMQ code of the patched tree (mmq-config-sweep.cuh gives the one config entry),
// so the kernel is the kernel of the build, with another tile configuration.
#include "common.cuh"
#include "mmq.cuh"

#define SWEEP_CAT_(a, b) a##b
#define SWEEP_CAT(a, b) SWEEP_CAT_(a, b)

void SWEEP_CAT(sweep_run_, SWEEP_NAME)(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    launch_mul_mat_q<SWEEP_TYPE, SWEEP_J, SWEEP_FALLBACK, GGML_PREC_Q8, false>(ctx, args, stream);
}

// out[0] registers, out[1] local memory bytes (spills and stack), out[2] static shared bytes,
// out[3] dynamic shared bytes of the launch, out[4] blocks for each SM, out[5] error code of the queries
void SWEEP_CAT(sweep_info_, SWEEP_NAME)(int * out) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(SWEEP_TYPE, SWEEP_J, SWEEP_FALLBACK, cc, GGML_PREC_Q8);
    const int nbytes_shared = (int) mmq_get_nbytes_shared(config, cc);
    cudaFuncAttributes attr = {};
    int err = (int) cudaFuncGetAttributes(&attr, mul_mat_q<SWEEP_TYPE, SWEEP_J, SWEEP_FALLBACK, GGML_PREC_Q8, false>);
    out[0] = attr.numRegs;
    out[1] = (int) attr.localSizeBytes;
    out[2] = (int) attr.sharedSizeBytes;
    out[3] = nbytes_shared;
    out[4] = 0;
    if (nbytes_shared <= (int) ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        cudaFuncSetAttribute(mul_mat_q<SWEEP_TYPE, SWEEP_J, SWEEP_FALLBACK, GGML_PREC_Q8, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
        int bps = 0;
        err |= (int) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps,
            mul_mat_q<SWEEP_TYPE, SWEEP_J, SWEEP_FALLBACK, GGML_PREC_Q8, false>, config.nthreads, nbytes_shared);
        out[4] = bps;
    }
    out[5] = err;
}
