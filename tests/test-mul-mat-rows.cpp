// ggml_mul_mat_rows computes the products of some rows of a matrix with the columns of b.
// The output head of a model uses it to compute the logits of a few tokens: the values must be
// the values of ggml_mul_mat, bit for bit, so that the answers of a model do not change.
//
// This test compares the two on the first CUDA device. It passes without a CUDA device (nothing to compare).
// Types and column counts that the graph of a model does not use are only reported (--all), they do not fail.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// the types and the number of columns that llm_graph_context::build_logits_tokens() uses, keep them in sync
static const ggml_type TYPES_EXACT[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
};
static const int N_COLS_EXACT_MAX = 4;

static const ggml_type TYPES_OTHER[] = {
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    GGML_TYPE_MXFP4, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32,
};

static int n_failed = 0;

// random rows of a, the rows are made in blocks to keep the memory small
static std::vector<uint8_t> make_weights(ggml_type type, int64_t k, int64_t n_rows, std::mt19937 & rng) {
    std::vector<uint8_t> out(ggml_row_size(type, k)*n_rows);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int64_t n_block = 1024;
    std::vector<float> rows(k*n_block);

    for (int64_t r0 = 0; r0 < n_rows; r0 += n_block) {
        const int64_t n = std::min(n_block, n_rows - r0);
        for (int64_t i = 0; i < k*n; ++i) {
            rows[i] = dist(rng);
        }

        uint8_t * dst = out.data() + ggml_row_size(type, k)*r0;
        if (type == GGML_TYPE_F32) {
            memcpy(dst, rows.data(), sizeof(float)*k*n);
        } else if (type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(rows.data(), (ggml_fp16_t *) dst, k*n);
        } else if (type == GGML_TYPE_BF16) {
            ggml_fp32_to_bf16_row(rows.data(), (ggml_bf16_t *) dst, k*n);
        } else {
            ggml_quantize_chunk(type, rows.data(), dst, 0, n, k, nullptr);
        }
    }

    return out;
}

struct case_result {
    int64_t n_values   = 0;
    int64_t n_diff     = 0;
    double  max_diff   = 0.0;
    bool    run        = false;
};

// ids[j, i] is the row of a for the column j and the result row i
// same_rows: the rows of the output head, the same list for all the columns
static case_result run_case(ggml_backend_t backend, ggml_type type, int64_t k, int64_t n_rows, int64_t n_cols, int64_t n_ids,
        bool same_rows, uint32_t seed, const std::vector<uint8_t> & wdata) {
    case_result res;

    std::mt19937 rng(seed);

    ggml_init_params ip_w = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx_w = ggml_init(ip_w);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, type, k, n_rows);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, n_cols);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_w, GGML_TYPE_I32, n_cols, n_ids);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (!buf_w) {
        fprintf(stderr, "failed to allocate the weights\n");
        ggml_free(ctx_w);
        return res;
    }

    // the weights are shared by the cases: copy them
    ggml_backend_tensor_set(w, wdata.data(), 0, ggml_nbytes(w));

    {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> xd(k*n_cols);
        for (auto & v : xd) {
            v = dist(rng);
        }
        ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));
    }

    std::vector<int32_t> idd(n_cols*n_ids);
    {
        std::uniform_int_distribution<int32_t> dist(0, (int32_t) n_rows - 1);
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t row = dist(rng);
            for (int64_t j = 0; j < n_cols; ++j) {
                idd[i*n_cols + j] = same_rows ? row : dist(rng);
            }
        }
        // the first and the last row of the matrix are always in the list
        idd[0] = 0;
        idd[(n_ids - 1)*n_cols + (n_cols - 1)] = (int32_t) n_rows - 1;
    }
    ggml_backend_tensor_set(ids, idd.data(), 0, ggml_nbytes(ids));

    // the graph
    std::vector<uint8_t> mem(ggml_tensor_overhead()*64 + ggml_graph_overhead());
    ggml_init_params ip_g = { mem.size(), mem.data(), true };
    ggml_context * ctx_g = ggml_init(ip_g);

    ggml_tensor * full = ggml_mul_mat(ctx_g, w, x);
    ggml_tensor * rows = ggml_mul_mat_rows(ctx_g, w, x, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, full);
    ggml_build_forward_expand(gf, rows);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "failed to allocate the graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx_g);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
        return res;
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "failed to compute the graph\n");
        n_failed++;
    } else {
        std::vector<float> full_d(ggml_nelements(full));
        std::vector<float> rows_d(ggml_nelements(rows));
        ggml_backend_tensor_get(full, full_d.data(), 0, ggml_nbytes(full));
        ggml_backend_tensor_get(rows, rows_d.data(), 0, ggml_nbytes(rows));

        GGML_ASSERT(rows->ne[0] == n_ids && rows->ne[1] == n_cols);

        res.run = true;
        for (int64_t j = 0; j < n_cols; ++j) {
            for (int64_t i = 0; i < n_ids; ++i) {
                const float a = rows_d[j*n_ids + i];
                const float b = full_d[j*n_rows + idd[i*n_cols + j]];
                res.n_values++;
                if (memcmp(&a, &b, sizeof(float)) != 0) {
                    res.n_diff++;
                    res.max_diff = std::max(res.max_diff, (double) std::fabs(a - b));
                }
            }
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);

    return res;
}

static bool is_exact_type(ggml_type type) {
    for (ggml_type t : TYPES_EXACT) {
        if (t == type) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    bool all  = false; // report the types that the graph does not use
    bool full = false; // the size of the output head of the model
    for (int i = 1; i < argc; ++i) {
        all  = all  || strcmp(argv[i], "--all")  == 0;
        full = full || strcmp(argv[i], "--full") == 0;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), "CUDA") == 0) {
            dev = d;
            break;
        }
    }

    if (!dev) {
        printf("no CUDA device: nothing to compare\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to init the backend\n");
        return 1;
    }

    printf("device: %s\n", ggml_backend_dev_description(dev));

    struct shape {
        int64_t k;
        int64_t n_rows;
    };

    std::vector<shape> shapes = { { 2560, 24576 }, { 4096, 8192 }, { 1536, 4000 } };
    if (full) {
        // the output head of the model: 248,320 rows of 2,560 values
        shapes.push_back({ 2560, 248320 });
    }

    std::vector<ggml_type> types(std::begin(TYPES_EXACT), std::end(TYPES_EXACT));
    if (all) {
        types.insert(types.end(), std::begin(TYPES_OTHER), std::end(TYPES_OTHER));
    }

    int n_cases = 0;
    int n_cases_diff = 0;

    for (ggml_type type : types) {
        for (const shape & s : shapes) {
            if (s.k % ggml_blck_size(type) != 0) {
                continue;
            }
            // the big shape only with the type of the model
            if (s.n_rows == 248320 && type != GGML_TYPE_Q8_0) {
                continue;
            }

            std::mt19937 rng(1234 + (uint32_t) type);
            const std::vector<uint8_t> wdata = make_weights(type, s.k, s.n_rows, rng);

            const int n_cols_max = all ? 8 : N_COLS_EXACT_MAX;

            for (int n_cols = 1; n_cols <= n_cols_max; ++n_cols) {
                for (int64_t n_ids : { 1, 2, 10, 33 }) {
                    for (bool same_rows : { true, false }) {
                        const case_result r = run_case(backend, type, s.k, s.n_rows, n_cols, n_ids, same_rows, 77 + n_cols*7 + (uint32_t) n_ids, wdata);

                        const bool must = is_exact_type(type) && n_cols <= N_COLS_EXACT_MAX;

                        n_cases++;
                        if (!r.run) {
                            n_failed++;
                            continue;
                        }
                        if (r.n_diff != 0) {
                            n_cases_diff++;
                            printf("%-8s k = %5lld rows = %6lld cols = %d ids = %2lld %s: %lld of %lld values differ, max diff %g%s\n",
                                    ggml_type_name(type), (long long) s.k, (long long) s.n_rows, n_cols, (long long) n_ids,
                                    same_rows ? "same rows" : "any rows ", (long long) r.n_diff, (long long) r.n_values, r.max_diff,
                                    must ? "  FAIL" : "  (not used)");
                            if (must) {
                                n_failed++;
                            }
                        }
                    }
                }
            }
            printf("%-8s k = %5lld rows = %6lld: done\n", ggml_type_name(type), (long long) s.k, (long long) s.n_rows);
        }
    }

    printf("%d cases, %d with different values, %d failed\n", n_cases, n_cases_diff, n_failed);

    ggml_backend_free(backend);

    return n_failed == 0 ? 0 : 1;
}
