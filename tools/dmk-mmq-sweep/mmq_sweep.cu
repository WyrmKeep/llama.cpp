// mmq_sweep.cu: micro-benchmark of MMQ tile configurations for the weight shapes of one model.
//
// It runs the MMQ launch code of llama.cpp (launch_mul_mat_q) for each variant that variants.inc lists,
// for each weight shape (type, rows M, K) and each number of tokens N. The weights and the activations are random.
// It writes one TSV row for each measurement to stdout.
//
// Usage: mmq_sweep [--shapes q4_0:9216:2560,...] [--ns 13,26,...] [--variants substring,...] [--policies 0,1,2]
//                  [--reps-ms 15] [--trials 3] [--check 1] [--list] [--j-baseline-only 0]
//
// The harness defines the few functions of ggml-cuda.cu that the MMQ code calls, so it does not need the library.
#include "common.cuh"
#include "mmq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------
// The knobs and diagnostics of the patched launch code.
int g_sweep_policy      = 0;
int g_sweep_last_grid   = 0;
int g_sweep_last_fixup  = 0;
int g_sweep_last_ntiles = 0;
int g_sweep_last_bps    = 0;

// ---------------------------------------------------------------------------------------------
// Stubs for ggml-cuda.cu.
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() { return g_info; }
void ggml_cuda_set_device(int device) { cudaSetDevice(device); }
int  ggml_cuda_get_device() { int d = 0; cudaGetDevice(&d); return d; }

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  in function %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    fflush(stderr);
    abort();
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    abort();
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

namespace {
struct sweep_pool : public ggml_cuda_pool {
    void * buf = nullptr;
    size_t cap = 0;
    void * alloc(size_t size, size_t * actual_size) override {
        if (size > cap) {
            if (buf) { cudaFree(buf); }
            if (cudaMalloc(&buf, size) != cudaSuccess) { fprintf(stderr, "pool: cudaMalloc %zu failed\n", size); abort(); }
            cap = size;
        }
        *actual_size = cap;
        return buf;
    }
    void free(void *, size_t) override {}
};
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context &, int, int) {
    return std::make_unique<sweep_pool>();
}

// ---------------------------------------------------------------------------------------------
// The variants.
typedef void (*run_fn)(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
typedef void (*info_fn)(int *);

struct Variant {
    const char * name;
    ggml_type    type;
    int nt, occ, I, J, streamk, fallback;
    run_fn  run;
    info_fn info;
    int  res[6];
    bool launchable;
};

#define VARIANT(name, type, nt, occ, I, J, sk, fb) \
    void sweep_run_##name(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t); \
    void sweep_info_##name(int *);
#include "variants.inc"
#undef VARIANT

static std::vector<Variant> g_variants = {
#define VARIANT(name, type, nt, occ, I, J, sk, fb) \
    { #name, type, nt, occ, I, J, sk, fb, sweep_run_##name, sweep_info_##name, {0, 0, 0, 0, 0, 0}, false },
#include "variants.inc"
#undef VARIANT
};

// ---------------------------------------------------------------------------------------------
// Data.
__device__ static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

__global__ static void fill_random(uint8_t * p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] = (uint8_t) hash32((uint32_t) i * 2654435761U + seed);
    }
}

// set the scale of each weight block (a half at the start of the block)
__global__ static void fix_x_scales(uint8_t * p, size_t nblocks, int ts, uint32_t seed) {
    for (size_t b = blockIdx.x * (size_t) blockDim.x + threadIdx.x; b < nblocks; b += (size_t) gridDim.x * blockDim.x) {
        const float d = 0.004f + 0.0001f * (float) (hash32((uint32_t) b + seed) & 63);
        *(half *) (p + b * ts) = __float2half(d);
    }
}

// set the scales in the first 16 bytes of each block_q8_1_mmq
__global__ static void fix_y_scales(uint8_t * p, size_t nblocks, int ds_layout, uint32_t seed) {
    for (size_t b = blockIdx.x * (size_t) blockDim.x + threadIdx.x; b < nblocks; b += (size_t) gridDim.x * blockDim.x) {
        uint8_t * q = p + b * sizeof(block_q8_1_mmq);
        for (int k = 0; k < 4; ++k) {
            const float d = 0.004f + 0.0001f * (float) (hash32((uint32_t) (b * 4 + k) + seed) & 63);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                half2 v = __floats2half2_rn(d, 0.5f * d * (float) ((int) (hash32((uint32_t) (b * 4 + k) ^ seed) & 255) - 128));
                ((half2 *) q)[k] = v;
            } else {
                ((float *) q)[k] = d;
            }
        }
    }
}

__global__ static void cmp_kernel(const float * a, const float * b, size_t n, unsigned long long * ndiff, int * maxabs_bits) {
    unsigned long long nd = 0;
    float mx = 0.0f;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float x = a[i], y = b[i];
        if (__float_as_uint(x) != __float_as_uint(y)) {
            ++nd;
            const float d = fabsf(x - y);
            mx = (d != d) ? 3.0e38f : fmaxf(mx, d);
        }
    }
    if (nd) { atomicAdd(ndiff, nd); atomicMax(maxabs_bits, __float_as_int(mx)); }
}

struct Shape { ggml_type type; int M, K; };

static const char * type_name(ggml_type t) { return t == GGML_TYPE_Q4_0 ? "Q4_0" : t == GGML_TYPE_Q8_0 ? "Q8_0" : "?"; }

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) { q = s.size(); }
        if (q > p) { out.push_back(s.substr(p, q - p)); }
        p = q + 1;
    }
    return out;
}

// the rule of ggml_cuda_mmq_get_J_best for the baseline table: the fewest tiles, the smallest J on a tie;
// the fallback table has J of 8, 16, 32, 64 and 128 only.
static int baseline_J(int N, bool fallback) {
    int best = 0, best_nt = 1 << 30;
    for (int J = 8; J <= 128 && best_nt > 1; J += 8) {
        if (fallback && J != 8 && J != 16 && J != 32 && J != 64 && J != 128) { continue; }
        const int nt = (N + J - 1) / J;
        if (nt < best_nt) { best = J; best_nt = nt; }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string shapes_s = "q4_0:9216:2560,q4_0:2560:9216,q8_0:8192:2560,q8_0:4096:2560,q8_0:2560:4096,q8_0:1024:2560";
    std::string ns_s = "13,26,30,43,47,51,54,67,69,84,86,94,102,108,120,129,151,158,256,512,1024,1173,1288,2048";
    std::string variants_s = "";
    std::string policies_s = "0";
    double reps_ms = 15.0;
    int trials = 3, check = 1, list = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if      (a == "--shapes")   shapes_s = val();
        else if (a == "--ns")       ns_s = val();
        else if (a == "--variants") variants_s = val();
        else if (a == "--policies") policies_s = val();
        else if (a == "--reps-ms")  reps_ms = atof(val().c_str());
        else if (a == "--trials")   trials = atoi(val().c_str());
        else if (a == "--check")    check = atoi(val().c_str());
        else if (a == "--list")     list = 1;
        else if (a == "sweep")      { /* the harness is started as `llama sweep ...`: ignore the command */ }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { fprintf(stderr, "no CUDA device\n"); return 3; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    g_info.device_count = 1;
    g_info.physical_device_count = 1;
    auto & d0 = g_info.devices[0];
    d0.cc = 100 * prop.major + 10 * prop.minor;
    d0.nsm = prop.multiProcessorCount;
    d0.smpb = prop.sharedMemPerBlock;
    d0.smpbo = prop.sharedMemPerBlockOptin;
    d0.warp_size = prop.warpSize;
    d0.total_vram = prop.totalGlobalMem;
    d0.physical_device = 0;
    d0.physical_share_count = 1;
    d0.virtual_index = 0;
    cudaSetDevice(0);

    // variants: resource data
    std::vector<std::string> vsel = split(variants_s, ',');
    for (auto & v : g_variants) {
        v.info(v.res);
        v.launchable = v.res[5] == 0 && v.res[4] > 0 && v.res[3] <= (int) d0.smpbo;
    }
    if (list) {
        printf("name\ttype\tnt\tocc_req\tI\tJ\tstreamk\tfallback\tregs\tlocal_bytes\tsmem_static\tsmem_dynamic\tblocks_per_sm\tlaunchable\n");
        for (auto & v : g_variants) {
            printf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", v.name, type_name(v.type), v.nt, v.occ, v.I, v.J, v.streamk, v.fallback,
                   v.res[0], v.res[1], v.res[2], v.res[3], v.res[4], v.launchable ? 1 : 0);
        }
        return 0;
    }

    std::vector<Shape> shapes;
    for (auto & s : split(shapes_s, ',')) {
        auto f = split(s, ':');
        if (f.size() != 3) { fprintf(stderr, "bad shape %s\n", s.c_str()); return 2; }
        Shape sh;
        sh.type = f[0] == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        sh.M = atoi(f[1].c_str());
        sh.K = atoi(f[2].c_str());
        shapes.push_back(sh);
    }
    std::vector<int> ns;
    for (auto & s : split(ns_s, ',')) { ns.push_back(atoi(s.c_str())); }
    std::vector<int> policies;
    for (auto & s : split(policies_s, ',')) { policies.push_back(atoi(s.c_str())); }

    ggml_backend_cuda_context ctx(0);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    fprintf(stderr, "device %s cc %d nsm %d smpbo %zu\n", prop.name, d0.cc, d0.nsm, d0.smpbo);
    printf("type\tM\tK\tN\tvariant\tnt\tocc_req\tI\tJ\tstreamk\tpolicy\tregs\tlocal_bytes\tsmem_dynamic\tblocks_per_sm\tntiles\tgrid\tfixup\ttime_us\tndiff\tmaxabs\tref\n");

    unsigned long long * d_ndiff;
    int * d_maxabs;
    cudaMalloc(&d_ndiff, sizeof(unsigned long long));
    cudaMalloc(&d_maxabs, sizeof(int));

    for (const Shape & sh : shapes) {
        const bool fallback = ggml_cuda_mmq_needs_fallback(sh.M);
        const int qk = 32;
        const size_t ts = sh.type == GGML_TYPE_Q4_0 ? 18 : 34;
        const size_t x_bytes = (size_t) sh.M * (sh.K / qk) * ts;
        const int nb = (int) std::max<size_t>(2, (48u << 20) / x_bytes + 1);
        std::vector<uint8_t *> xs(nb);
        for (int c = 0; c < nb; ++c) {
            cudaMalloc(&xs[c], x_bytes);
            fill_random<<<1024, 256>>>(xs[c], x_bytes, 1234 + c);
            fix_x_scales<<<1024, 256>>>(xs[c], x_bytes / ts, (int) ts, 77 + c);
        }
        const int ds_layout = mmq_get_q8_1_ds_layout(sh.type);

        for (int N : ns) {
            const size_t y_bytes = (size_t) (sh.K / 128) * N * sizeof(block_q8_1_mmq) + 4 * 128 * sizeof(block_q8_1_mmq) + 4096;
            uint8_t * y;
            cudaMalloc(&y, y_bytes);
            fill_random<<<1024, 256>>>(y, y_bytes, 999);
            fix_y_scales<<<1024, 256>>>(y, y_bytes / sizeof(block_q8_1_mmq), ds_layout, 4242);
            float * dst, * ref;
            const size_t dst_floats = (size_t) sh.M * N + 128 * 8;
            cudaMalloc(&dst, dst_floats * sizeof(float));
            cudaMalloc(&ref, dst_floats * sizeof(float));
            cudaMemset(dst, 0, dst_floats * sizeof(float));
            cudaMemset(ref, 0, dst_floats * sizeof(float));
            cudaDeviceSynchronize();

            mmq_args args = {};
            args.type_x = sh.type;
            args.y = (const int *) y;
            args.dst = dst;
            args.ncols_x = sh.K; args.nrows_x = sh.M; args.ncols_dst = N;
            args.stride_row_x = sh.K / qk;
            args.ncols_y = N; args.nrows_dst = sh.M;
            args.nchannels_x = 1; args.nchannels_y = 1;
            args.stride_channel_x = (int64_t) sh.M * (sh.K / qk);
            args.stride_channel_y = (int64_t) N * (sh.K / 128) * sizeof(block_q8_1_mmq) / sizeof(int) / 1; // unused for one channel
            args.stride_channel_dst = (int64_t) sh.M * N;
            args.nsamples_x = 1; args.nsamples_y = 1;
            args.stride_sample_x = args.stride_channel_x; args.stride_sample_y = args.stride_channel_y; args.stride_sample_dst = args.stride_channel_dst;
            args.ncols_max = N;

            const int Jb = baseline_J(N, fallback);
            bool have_ref = false;

            // the baseline variant (the table of the build) at its J is the reference of this shape
            std::vector<int> order;
            for (size_t v = 0; v < g_variants.size(); ++v) { order.push_back((int) v); }
            auto is_baseline = [&](const Variant & v) {
                return v.type == sh.type && v.nt == 256 && v.occ == 1 && v.I == 128 && v.streamk == 1 && v.J == Jb && (v.fallback != 0) == fallback;
            };
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return is_baseline(g_variants[a]) && !is_baseline(g_variants[b]); });

            for (int vi : order) {
                Variant & v = g_variants[vi];
                if (v.type != sh.type || (v.fallback != 0) != fallback || !v.launchable) { continue; }
                if (!vsel.empty() && !is_baseline(v)) {
                    bool ok = false;
                    for (auto & s : vsel) { if (strstr(v.name, s.c_str())) { ok = true; } }
                    if (!ok) { continue; }
                }
                const int nt_x = (N + v.J - 1) / v.J;
                const int N8 = (N + 7) / 8 * 8;
                if (nt_x == 1 && v.J - N8 >= 8) { continue; } // a wider tile than the tokens: only waste

                for (int pol : policies) {
                    if (!v.streamk && pol != policies[0]) { continue; }
                    g_sweep_policy = pol;
                    args.J_best = v.J;
                    // the first launch; it also gives the diagnostics
                    args.x = (const char *) xs[0];
                    cudaMemsetAsync(dst, 0, dst_floats * sizeof(float), stream);
                    v.run(ctx, args, stream);
                    cudaError_t err = cudaStreamSynchronize(stream);
                    if (err == cudaSuccess) { err = cudaGetLastError(); }
                    if (err != cudaSuccess) {
                        fprintf(stderr, "variant %s N=%d: %s\n", v.name, N, cudaGetErrorString(err));
                        printf("%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\tERR\n", type_name(sh.type), sh.M, sh.K, N, v.name, v.nt, v.occ, v.I, v.J, v.streamk, pol);
                        fflush(stdout);
                        // a sticky error ends the process (the context is gone)
                        if (err == cudaErrorIllegalAddress || err == cudaErrorLaunchFailure) { return 4; }
                        continue;
                    }
                    const int grid = g_sweep_last_grid, fixup = g_sweep_last_fixup, ntiles = g_sweep_last_ntiles, bps = g_sweep_last_bps;
                    unsigned long long ndiff = 0;
                    float maxabs = 0.0f;
                    if (check) {
                        if (!have_ref) {
                            cudaMemcpyAsync(ref, dst, dst_floats * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                            have_ref = true;
                        } else {
                            cudaMemsetAsync(d_ndiff, 0, sizeof(unsigned long long), stream);
                            cudaMemsetAsync(d_maxabs, 0, sizeof(int), stream);
                            cmp_kernel<<<512, 256, 0, stream>>>(dst, ref, (size_t) sh.M * N, d_ndiff, d_maxabs);
                            cudaMemcpyAsync(&ndiff, d_ndiff, sizeof(ndiff), cudaMemcpyDeviceToHost, stream);
                            int mb = 0;
                            cudaMemcpyAsync(&mb, d_maxabs, sizeof(mb), cudaMemcpyDeviceToHost, stream);
                            cudaStreamSynchronize(stream);
                            memcpy(&maxabs, &mb, sizeof(float));
                        }
                    }
                    // warm-up, then an estimate of the time of one launch
                    for (int r = 0; r < 3; ++r) { args.x = (const char *) xs[r % nb]; v.run(ctx, args, stream); }
                    cudaEventRecord(ev0, stream);
                    for (int r = 0; r < 5; ++r) { args.x = (const char *) xs[r % nb]; v.run(ctx, args, stream); }
                    cudaEventRecord(ev1, stream);
                    cudaEventSynchronize(ev1);
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    const double t_est_ms = std::max(ms / 5.0, 0.001);
                    const int reps = (int) std::min(400.0, std::max(5.0, reps_ms / t_est_ms));
                    double best = 1e30;
                    for (int t = 0; t < trials; ++t) {
                        cudaEventRecord(ev0, stream);
                        for (int r = 0; r < reps; ++r) { args.x = (const char *) xs[r % nb]; v.run(ctx, args, stream); }
                        cudaEventRecord(ev1, stream);
                        cudaEventSynchronize(ev1);
                        cudaEventElapsedTime(&ms, ev0, ev1);
                        best = std::min(best, (double) ms / reps);
                    }
                    printf("%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%llu\t%.6g\t%d\n",
                           type_name(sh.type), sh.M, sh.K, N, v.name, v.nt, v.occ, v.I, v.J, v.streamk, pol,
                           v.res[0], v.res[1], v.res[3], bps ? bps : v.res[4], ntiles, grid, fixup, best * 1000.0, ndiff, (double) maxabs,
                           is_baseline(v) ? 1 : 0);
                    fflush(stdout);
                }
            }
            cudaFree(y); cudaFree(dst); cudaFree(ref);
        }
        for (auto * p : xs) { cudaFree(p); }
    }
    return 0;
}
