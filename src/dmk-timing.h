#pragma once

// DMK_TIMING=1: host-side wall times of every llama_decode on stderr, prefix "DMK_TIMING" (private build, not for upstream).
// Lines (key=value, microseconds, t = CLOCK_MONOTONIC in seconds, the same clock as the CUDA backend lines):
//   DMK_TIMING ubatch  id n_tokens n_seqs n_seq_tokens n_outputs n_nodes reuse  apply_us build_us alloc_us inputs_us submit_us total_us
//       one per ubatch: memory apply, graph build (0 if the previous graph was reused), scheduler alloc, set_inputs and the
//       submit of the compute (the host time of ggml_backend_sched_graph_compute_async: op dispatch, CUDA graph capture/launch)
//   DMK_TIMING cuda_compute n_nodes mode host_us        (from ggml-cuda.cu, one per backend graph compute)
//   DMK_TIMING sync    passes tokens wait_us since_first_submit_us since_last_submit_us
//       one per synchronize() that had passes pending: the host blocked wait_us for the GPU to finish them
//   DMK_TIMING decode  n_tokens n_ubatches init_us total_us
//       one per llama_decode: memory init_batch (ubatch split) and the whole call
// Reading the logits (llama_get_logits_ith) synchronizes, so for the decision server the sync line follows the decode line.
// With the variable unset the cost is one load and one branch per ubatch.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace dmk {

inline bool timing_enabled() {
    static const bool on = [] {
        const char * env = getenv("DMK_TIMING");
        return env != nullptr && atoi(env) != 0;
    }();
    return on;
}

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// per llama_context
struct state {
    uint64_t id_ubatch  = 0;   // ubatches submitted so far
    int32_t  passes     = 0;   // submitted since the last synchronize
    int64_t  tokens     = 0;
    int64_t  t_first_ns = 0;   // start of the first pending submit
    int64_t  t_last_ns  = 0;   // end of the last pending submit
};

// marks inside process_ubatch: 0 start, 1 memory applied, 2 graph built, 3 graph allocated, 4 inputs set, 5 compute submitted
struct ubatch_timer {
    bool    on;
    bool    reused = false;
    int64_t t[6]   = {};

    ubatch_timer() : on(timing_enabled()) {
        mark(0);
    }

    void mark(int i) {
        if (on) {
            t[i] = now_ns();
        }
    }

    // time between a mark and the previous one that was set, 0 for a stage that did not run
    double us(int i) const {
        int64_t prev = t[0];
        for (int j = 1; j < i; j++) {
            if (t[j] != 0) {
                prev = t[j];
            }
        }
        return t[i] != 0 ? (t[i] - prev) / 1e3 : 0.0;
    }
};

inline void print_ubatch(state & st, const ubatch_timer & tm, uint32_t n_tokens, uint32_t n_seqs, uint32_t n_seq_tokens,
        uint32_t n_outputs, int n_nodes) {
    st.id_ubatch++;
    if (st.passes == 0) {
        st.t_first_ns = tm.t[0];
    }
    st.passes++;
    st.tokens += n_tokens;
    st.t_last_ns = tm.t[5];

    fprintf(stderr,
            "DMK_TIMING ubatch t=%.6f id=%llu n_tokens=%u n_seqs=%u n_seq_tokens=%u n_outputs=%u n_nodes=%d reuse=%d "
            "apply_us=%.0f build_us=%.0f alloc_us=%.0f inputs_us=%.0f submit_us=%.0f total_us=%.0f\n",
            tm.t[5] / 1e9, (unsigned long long) st.id_ubatch, n_tokens, n_seqs, n_seq_tokens, n_outputs, n_nodes, (int) tm.reused,
            tm.us(1), tm.us(2), tm.us(3), tm.us(4), tm.us(5), (tm.t[5] - tm.t[0]) / 1e3);
}

inline void print_sync(state & st, int64_t t_begin_ns, int64_t t_end_ns) {
    if (st.passes > 0) {
        fprintf(stderr, "DMK_TIMING sync t=%.6f passes=%d tokens=%lld wait_us=%.0f since_first_submit_us=%.0f since_last_submit_us=%.0f\n",
                t_end_ns / 1e9, st.passes, (long long) st.tokens, (t_end_ns - t_begin_ns) / 1e3,
                (t_end_ns - st.t_first_ns) / 1e3, (t_end_ns - st.t_last_ns) / 1e3);
    }
    st.passes = 0;
    st.tokens = 0;
}

// one per llama_context::decode call
struct decode_scope {
    bool           on;
    const state *  st;
    uint64_t       id_begin;
    uint64_t       n_tokens;
    int64_t        t0;
    int64_t        init_ns = 0;

    decode_scope(const state & st, uint64_t n_tokens) : on(timing_enabled()), st(&st), id_begin(st.id_ubatch), n_tokens(n_tokens), t0(on ? now_ns() : 0) {}

    ~decode_scope() {
        if (on) {
            const int64_t t1 = now_ns();
            fprintf(stderr, "DMK_TIMING decode t=%.6f n_tokens=%llu n_ubatches=%llu init_us=%.0f total_us=%.0f\n", t1 / 1e9,
                    (unsigned long long) n_tokens, (unsigned long long) (st->id_ubatch - id_begin), init_ns / 1e3, (t1 - t0) / 1e3);
        }
    }
};

} // namespace dmk
