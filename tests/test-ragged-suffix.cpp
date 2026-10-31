// Checks the ragged split of the suffixes of several sequences on a hybrid model with recurrent layers.
//
// The sequences share a prefix, then each one decodes its own suffix, all in one batch. The batch is decoded
// with the equal-length split (LLAMA_RAGGED_SPLIT=0) and with the ragged split (LLAMA_RAGGED_SPLIT=1). The test
// compares the logits of the output tokens and the conv and recurrent state of every sequence. It also
// decodes the same batch with the equal-length split a second time, to know the difference that two runs
// of the same code make (the self-repeat floor), and one sequence at a time.
//
//   test-ragged-suffix -m model.gguf [-ngl N] [--prefix N] [--lens 43,47,73] [--text file] [--all-logits]
//                      [--kl-mean X] [--kl-max X] [--state-nmse X]

#include "llama.h"
#include "common.h"

#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct run_result {
    int n_passes = 0;

    std::vector<std::vector<float>> logits; // one row for each output token

    // [sequence][layer], empty for the layers without recurrent state
    std::vector<std::vector<std::vector<float>>> conv;
    std::vector<std::vector<std::vector<float>>> ssm;
};

struct test_params {
    std::string model;
    std::string text;
    int         ngl        = 999;
    int         n_prefix   = 600;
    bool        all_logits = false;
    double      kl_budget  = 0.0;      // the KL budget of the text, 0 = no check of the absolute limit
    std::vector<int> lens  = { 43, 47, 73 };
};

// the passes (graph computes) since the last reset, counted with the eval callback on the final norm
static int g_passes = 0;

static bool count_passes_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    if (strncmp(t->name, "result_norm", 11) != 0) {
        return false;
    }

    if (!ask) {
        ++g_passes;
    }

    return true;
}

static bool decode(llama_context * ctx, common_batch & batch) {
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

// the conv and recurrent state of the sequence in every layer that has them
static void read_states(llama_context * ctx, llama_seq_id seq_id,
        std::vector<std::vector<float>> & conv, std::vector<std::vector<float>> & ssm) {
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    if (hybrid == nullptr) {
        fprintf(stderr, "the model has no hybrid memory\n");
        exit(1);
    }

    llama_memory_recurrent * recr = hybrid->get_mem_recr();

    const int32_t tail = recr->cells[seq_id].tail;
    if (tail < 0) {
        fprintf(stderr, "sequence %d has no recurrent state\n", seq_id);
        exit(1);
    }

    const auto read_row = [&](ggml_tensor * t, std::vector<float> & dst) {
        if (t == nullptr) {
            return;
        }

        if (t->type != GGML_TYPE_F32) {
            fprintf(stderr, "the state type must be f32\n");
            exit(1);
        }

        dst.resize(t->ne[0]);
        ggml_backend_tensor_get(t, dst.data(), (size_t) tail*t->nb[1], dst.size()*sizeof(float));
    };

    conv.assign(recr->r_l.size(), {});
    ssm .assign(recr->s_l.size(), {});

    for (size_t il = 0; il < recr->r_l.size(); ++il) {
        read_row(recr->r_l[il], conv[il]);
        read_row(recr->s_l[il], ssm [il]);
    }
}

// the suffix of every sequence is decoded in one batch, or each one in its own batch
static run_result run(llama_context * ctx, const std::vector<llama_token> & prefix,
        const std::vector<std::vector<llama_token>> & suffixes, bool ragged, bool one_by_one, bool all_logits) {
    llama_memory_t mem = llama_get_memory(ctx);

    llama_memory_clear(mem, true);

    const int n_seqs   = (int) suffixes.size();
    const int n_ubatch = (int) llama_n_ubatch(ctx);

    // the prefix in sequence 0, a copy of its state in the sequences of the suffixes
    common_batch batch(ctx);

    for (size_t i0 = 0; i0 < prefix.size(); i0 += n_ubatch) {
        const size_t i1 = std::min(prefix.size(), i0 + n_ubatch);

        batch.clear();
        for (size_t i = i0; i < i1; ++i) {
            batch.add(prefix[i], (llama_pos) i, 0, i + 1 == prefix.size());
        }

        if (!decode(ctx, batch)) {
            fprintf(stderr, "decode of the prefix failed\n");
            exit(1);
        }
    }

    for (int s = 1; s <= n_seqs; ++s) {
        llama_memory_seq_cp(mem, 0, s, -1, -1);
    }

    setenv("LLAMA_RAGGED_SPLIT", ragged ? "1" : "0", 1);

    run_result res;

    g_passes = 0;

    const auto decode_suffixes = [&](int s_first, int s_last) {
        batch.clear();

        std::vector<int32_t> idx;
        for (int s = s_first; s < s_last; ++s) {
            const auto & suffix = suffixes[s];
            for (size_t j = 0; j < suffix.size(); ++j) {
                const bool output = all_logits || j + 1 == suffix.size();
                const int32_t i = batch.add(suffix[j], (llama_pos) (prefix.size() + j), s + 1, output);
                if (output) {
                    idx.push_back(i);
                }
            }
        }

        if (!decode(ctx, batch)) {
            fprintf(stderr, "decode of the suffixes failed\n");
            exit(1);
        }

        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        for (int32_t i : idx) {
            const float * l = llama_get_logits_ith(ctx, i);
            res.logits.emplace_back(l, l + n_vocab);
        }
    };

    if (one_by_one) {
        for (int s = 0; s < n_seqs; ++s) {
            decode_suffixes(s, s + 1);
        }
    } else {
        decode_suffixes(0, n_seqs);
    }

    res.n_passes = g_passes;

    res.conv.resize(n_seqs);
    res.ssm .resize(n_seqs);
    for (int s = 0; s < n_seqs; ++s) {
        read_states(ctx, s + 1, res.conv[s], res.ssm[s]);
    }

    return res;
}

struct diff_stats {
    double max_abs = 0.0; // the largest absolute difference
    double nmse    = 0.0; // the largest sum((a - b)^2) / sum(b^2) of one vector
    size_t n_same  = 0;   // the vectors that are bit-identical
    size_t n_vec   = 0;
};

static void diff_vec(diff_stats & st, const std::vector<float> & a, const std::vector<float> & ref) {
    if (a.empty() && ref.empty()) {
        return;
    }

    if (a.size() != ref.size()) {
        fprintf(stderr, "the vectors have different sizes\n");
        exit(1);
    }

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) ref[i];

        st.max_abs = std::max(st.max_abs, std::fabs(d));
        num += d*d;
        den += (double) ref[i]*(double) ref[i];
    }

    st.nmse = std::max(st.nmse, den > 0.0 ? num/den : num);
    ++st.n_vec;

    if (memcmp(a.data(), ref.data(), a.size()*sizeof(float)) == 0) {
        ++st.n_same;
    }
}

static diff_stats diff_states(const std::vector<std::vector<std::vector<float>>> & a, const std::vector<std::vector<std::vector<float>>> & ref) {
    diff_stats st;
    for (size_t s = 0; s < a.size(); ++s) {
        for (size_t il = 0; il < a[s].size(); ++il) {
            diff_vec(st, a[s][il], ref[s][il]);
        }
    }

    return st;
}

// the largest NMSE of one layer over all sequences, for a table of the layers
static double layer_nmse(const std::vector<std::vector<std::vector<float>>> & a, const std::vector<std::vector<std::vector<float>>> & ref, size_t il) {
    diff_stats st;
    for (size_t s = 0; s < a.size(); ++s) {
        if (il < a[s].size()) {
            diff_vec(st, a[s][il], ref[s][il]);
        }
    }

    return st.nmse;
}

// one line for each recurrent layer: the NMSE of the conv and the recurrent state of three comparisons
static void print_layers(const run_result & ragged, const run_result & equal, const run_result & solo) {
    printf("layer  ragged-vs-equal conv / ssm   equal-vs-one-at-a-time conv / ssm   ragged-vs-one-at-a-time conv / ssm  (NMSE)\n");
    for (size_t il = 0; il < ragged.conv[0].size(); ++il) {
        if (ragged.conv[0][il].empty() && ragged.ssm[0][il].empty()) {
            continue;
        }

        printf("%5zu  %.3e %.3e        %.3e %.3e              %.3e %.3e\n", il,
                layer_nmse(ragged.conv, equal.conv, il), layer_nmse(ragged.ssm, equal.ssm, il),
                layer_nmse(equal.conv,  solo.conv,  il), layer_nmse(equal.ssm,  solo.ssm,  il),
                layer_nmse(ragged.conv, solo.conv,  il), layer_nmse(ragged.ssm, solo.ssm,  il));
    }
}

struct logit_stats {
    double kl_mean   = 0.0; // KL(ref || a), mean over the rows
    double kl_max    = 0.0;
    double max_abs   = 0.0; // the largest difference of a logit
    size_t n_top1    = 0;   // the rows with the same most likely token
    size_t n_rows    = 0;
};

static logit_stats diff_logits(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & ref) {
    logit_stats st;

    for (size_t r = 0; r < a.size(); ++r) {
        const auto & x = a[r];
        const auto & p = ref[r];

        const auto log_sum_exp = [](const std::vector<float> & v) {
            const double m = *std::max_element(v.begin(), v.end());
            double s = 0.0;
            for (float f : v) {
                s += std::exp((double) f - m);
            }
            return m + std::log(s);
        };

        const double lse_x = log_sum_exp(x);
        const double lse_p = log_sum_exp(p);

        double kl = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double lp = (double) p[i] - lse_p;
            const double lx = (double) x[i] - lse_x;

            kl += std::exp(lp)*(lp - lx);
            st.max_abs = std::max(st.max_abs, std::fabs((double) x[i] - (double) p[i]));
        }

        st.kl_mean += kl;
        st.kl_max   = std::max(st.kl_max, kl);

        if (std::max_element(x.begin(), x.end()) - x.begin() == std::max_element(p.begin(), p.end()) - p.begin()) {
            ++st.n_top1;
        }
        ++st.n_rows;
    }

    st.kl_mean /= std::max<size_t>(1, st.n_rows);

    return st;
}

static void print_diff(const char * name, const logit_stats & l, const diff_stats & conv, const diff_stats & ssm) {
    printf("%-22s kl_mean %.3e  kl_max %.3e  max_abs_logit %.3e  top1 %zu/%zu | conv: max_abs %.3e  nmse %.3e  identical %zu/%zu"
           " | ssm: max_abs %.3e  nmse %.3e  identical %zu/%zu\n",
           name, l.kl_mean, l.kl_max, l.max_abs, l.n_top1, l.n_rows,
           conv.max_abs, conv.nmse, conv.n_same, conv.n_vec, ssm.max_abs, ssm.nmse, ssm.n_same, ssm.n_vec);
}

int main(int argc, char ** argv) {
    test_params p;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m") {
            p.model = next();
        } else if (arg == "-ngl") {
            p.ngl = atoi(next());
        } else if (arg == "--prefix") {
            p.n_prefix = atoi(next());
        } else if (arg == "--text") {
            p.text = next();
        } else if (arg == "--all-logits") {
            p.all_logits = true;
        } else if (arg == "--kl-budget") {
            p.kl_budget = atof(next());
        } else if (arg == "--lens") {
            p.lens.clear();
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) {
                p.lens.push_back(atoi(item.c_str()));
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", arg.c_str());
            return 1;
        }
    }

    if (p.model.empty() || p.lens.size() < 2) {
        fprintf(stderr, "usage: %s -m model.gguf [-ngl N] [--prefix N] [--lens a,b,c] [--text file] [--all-logits] [--kl-budget X]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.ngl;

    llama_model * model = llama_model_load_from_file(p.model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "cannot load the model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the tokens: the text of a file, or a fixed pseudo-random sequence
    const int n_seqs = (int) p.lens.size();

    int n_suffix_total = 0;
    for (int len : p.lens) {
        n_suffix_total += len;
    }

    const int n_tokens_needed = p.n_prefix + 2*n_suffix_total;

    std::vector<llama_token> tokens;
    if (!p.text.empty()) {
        std::ifstream f(p.text);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();

        tokens.resize(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
        tokens.resize(std::max(n, 0));
    } else {
        uint32_t x = 12345;
        while ((int) tokens.size() < n_tokens_needed) {
            x = x*1664525u + 1013904223u;
            tokens.push_back(1000 + (llama_token) ((x >> 8) % 30000));
        }
    }

    if ((int) tokens.size() < n_tokens_needed) {
        fprintf(stderr, "the text has %zu tokens, the test needs %d\n", tokens.size(), n_tokens_needed);
        return 1;
    }

    std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + p.n_prefix);

    // the suffix of each sequence starts at a different place of the text
    std::vector<std::vector<llama_token>> suffixes;
    {
        size_t off = p.n_prefix;
        for (int len : p.lens) {
            suffixes.emplace_back(tokens.begin() + off, tokens.begin() + off + len);
            off += len;
        }
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = p.n_prefix + n_suffix_total + 256;
    cparams.n_batch           = 2048;
    cparams.n_ubatch          = 2048;
    cparams.n_seq_max         = n_seqs + 1;
    cparams.kv_unified        = true;
    cparams.cb_eval           = count_passes_cb;
    cparams.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create the context\n");
        return 1;
    }

    const auto do_run = [&](const char * name, bool ragged, bool one_by_one) {
        run_result res = run(ctx, prefix, suffixes, ragged, one_by_one, p.all_logits);

        printf("%-22s passes of the suffixes: %d\n", name, res.n_passes);

        return res;
    };

    const run_result old_split = do_run("equal split",        false, false);
    const run_result old_again = do_run("equal split again",  false, false);
    const run_result ragged    = do_run("ragged split",       true,  false);
    const run_result ragged_2  = do_run("ragged split again", true,  false);
    const run_result solo      = do_run("one sequence a time", false, true);

    printf("\nlens");
    for (int len : p.lens) {
        printf(" %d", len);
    }
    printf("  prefix %d  rows %zu\n", p.n_prefix, old_split.logits.size());

    const auto report = [&](const char * name, const run_result & a, const run_result & ref) {
        const logit_stats l = diff_logits(a.logits, ref.logits);
        const diff_stats  c = diff_states(a.conv, ref.conv);
        const diff_stats  s = diff_states(a.ssm,  ref.ssm);

        print_diff(name, l, c, s);

        return std::make_tuple(l, c, s);
    };

    // d(x) = x against the reference, the decode of one sequence at a time
    const auto fl     = report("floor (equal again)",    old_again, old_split);
    GGML_UNUSED(fl);
    const auto rep    = report("ragged again vs ragged", ragged_2,  ragged);
    const auto d_eq   = report("equal vs one at a time", old_split, solo);
    const auto d_rg   = report("ragged vs one at a time", ragged,   solo);
    const auto r_vs_e = report("ragged vs equal",        ragged,    old_split);

    print_layers(ragged, old_split, solo);

    bool ok = true;

    const auto check = [&](bool cond, const char * what) {
        printf("%s: %s\n", cond ? "ok  " : "FAIL", what);
        ok = ok && cond;
    };

    check(ragged.n_passes == 1, "the ragged split decodes all the suffixes in one pass");
    check(old_split.n_passes > 1, "the equal split needs more than one pass");
    check(std::get<0>(rep).kl_mean == 0.0 && std::get<0>(rep).kl_max == 0.0 &&
          std::get<1>(rep).n_same == std::get<1>(rep).n_vec && std::get<2>(rep).n_same == std::get<2>(rep).n_vec,
          "a second ragged run gives the same logits and the same states bit for bit");

    // the five quantities of the input: mean KL, largest KL, rows with another top token, largest NMSE of the conv and the SSM state
    struct quantity {
        const char * name;
        double       ragged_vs_seq;
        double       equal_vs_seq;
        double       ragged_vs_equal;
    };

    const auto top_miss = [](const logit_stats & l) { return (double) (l.n_rows - l.n_top1); };

    const quantity quantities[5] = {
        { "kl_mean",     std::get<0>(d_rg).kl_mean, std::get<0>(d_eq).kl_mean, std::get<0>(r_vs_e).kl_mean },
        { "kl_max",      std::get<0>(d_rg).kl_max,  std::get<0>(d_eq).kl_max,  std::get<0>(r_vs_e).kl_max  },
        { "top_miss",    top_miss(std::get<0>(d_rg)), top_miss(std::get<0>(d_eq)), top_miss(std::get<0>(r_vs_e)) },
        { "conv_nmse",   std::get<1>(d_rg).nmse,    std::get<1>(d_eq).nmse,    std::get<1>(r_vs_e).nmse    },
        { "ssm_nmse",    std::get<2>(d_rg).nmse,    std::get<2>(d_eq).nmse,    std::get<2>(r_vs_e).nmse    },
    };

    // QUANTITY lines are for the analysis: the sums over the inputs need the values of all inputs
    printf("\n%-10s %-14s %-14s %-14s %-10s %-10s\n", "QUANTITY", "ragged_vs_seq", "equal_vs_seq", "ragged_vs_eq", "x3_input", "info_x1_x2");
    for (const quantity & q : quantities) {
        const bool in_3x = q.ragged_vs_seq <= 3.0*q.equal_vs_seq;
        // information only: ragged against equal, 1 or 2 times the floor of the equal split
        const int  info  = q.ragged_vs_equal <= q.equal_vs_seq ? 1 : (q.ragged_vs_equal <= 2.0*q.equal_vs_seq ? 2 : 3);
        printf("%-10s %-14.6e %-14.6e %-14.6e %-10s %-10d\n", q.name, q.ragged_vs_seq, q.equal_vs_seq, q.ragged_vs_equal, in_3x ? "ok" : "FAIL", info);
        ok = ok && in_3x;
    }

    if (p.kl_budget > 0.0) {
        printf("budget: mean KL of ragged against one at a time %.3e, limit %.3e (10 %% of %.6f)\n",
                std::get<0>(d_rg).kl_mean, 0.1*p.kl_budget, p.kl_budget);
        check(std::get<0>(d_rg).kl_mean <= 0.1*p.kl_budget, "mean KL of the logits of the ragged split is 10 % or less of the KL budget");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return ok ? 0 : 1;
}
