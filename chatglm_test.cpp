#include "chatglm.h"
#include <filesystem>
#include <gtest/gtest.h>

#ifdef GGML_USE_CUBLAS
#include <cuda_runtime.h>
#include <ggml-cuda.h>
#endif

namespace chatglm {

namespace fs = std::filesystem;

static inline int get_num_threads() {
    const char *chatglm_num_threads_env = getenv("CHATGLM_NUM_THREADS");
    int num_threads = chatglm_num_threads_env ? std::stoi(chatglm_num_threads_env) : get_default_num_threads();
    return num_threads;
}

static inline void expect_all_close(ggml_tensor *a, ggml_tensor *b, float atol = 1e-5f, float rtol = 0.f) {
    ASSERT_EQ(a->type, b->type);
    ASSERT_EQ(a->type, GGML_TYPE_F32);
    ASSERT_EQ(ggml_nelements(a), ggml_nelements(b));
    int64_t numel = ggml_nelements(a);
    for (int64_t i = 0; i < numel; i++) {
        float ai = ((float *)a->data)[i];
        float bi = ((float *)b->data)[i];
        EXPECT_LT(std::abs(ai - bi), atol + rtol * std::abs(bi)) << "diff " << ai << " vs " << bi;
    }
}

static inline char *read_tensor_data(char *ptr, ggml_tensor *tensor) {
    memcpy(tensor->data, ptr, ggml_nbytes(tensor));
    return ptr + ggml_nbytes(tensor);
}

static inline float random() { return rand() / (float)RAND_MAX; }

static inline float random(float lo, float hi) { return lo + random() * (hi - lo); }

static inline void random_fill(ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    for (float &v : values) {
        v = random();
    }
    int64_t hist[16]{};

    if (tensor->type == GGML_TYPE_F32) {
        memcpy(tensor->data, values.data(), sizeof(float) * values.size());
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(values.data(), (ggml_fp16_t *)tensor->data, values.size());
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        ggml_quantize_q8_0(values.data(), tensor->data, ggml_nelements(tensor), tensor->ne[0], hist);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_quantize_q4_0(values.data(), tensor->data, ggml_nelements(tensor), tensor->ne[0], hist);
    } else if (tensor->type == GGML_TYPE_Q4_1) {
        ggml_quantize_q4_1(values.data(), tensor->data, ggml_nelements(tensor), tensor->ne[0], hist);
    } else {
        CHATGLM_THROW << "unsupported dtype " << ggml_type_name(tensor->type);
    }
}

// return elapsed time in milliseconds
static inline float timeit(std::function<void()> fn, int warmup, int active) {
    for (int i = 0; i < warmup; i++) {
        fn();
    }

#ifdef GGML_USE_CUBLAS
    CHATGLM_CHECK_CUDA(cudaDeviceSynchronize());
#endif
    int64_t start_us = ggml_time_us();
    for (int i = 0; i < active; i++) {
        fn();
    }
#ifdef GGML_USE_CUBLAS
    CHATGLM_CHECK_CUDA(cudaDeviceSynchronize());
#endif
    int64_t end_us = ggml_time_us();

    float elapsed_ms = (end_us - start_us) / 1000.f;
    return elapsed_ms / active;
}

static bool equal(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static std::vector<int> extract_sorted_ids(std::vector<TokenIdScore> &token_scores) {
    std::vector<int> token_ids(token_scores.size());
    for (size_t i = 0; i < token_scores.size(); i++) {
        token_ids[i] = token_scores[i].id;
    }
    std::sort(token_ids.begin(), token_ids.end());
    return token_ids;
}

TEST(Sampling, RepetitionPenalty) {
    constexpr float penalty = 1.2;
    std::vector<float> logits{0.96, 1.2, -2, -0.8, 0, 2.4, -1};
    std::vector<int> input_ids{0, 2, 5, 2};
    // reference
    std::vector<float> target{0.8, 1.2, -2.4, -0.8, 0, 2, -1};
    // test
    BaseModelForCausalLM::sampling_repetition_penalty(logits.data(), logits.data() + logits.size(), input_ids, penalty);
    // compare
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_FLOAT_EQ(logits[i], target[i]);
    }
}

TEST(DISABLED_Sampling, BenchmarkRepetitionPenalty) {
    const float penalty = 1.2;
    constexpr size_t vocab_size = 128000;
    constexpr int seq_len = 32000;
    std::vector<float> logits(vocab_size);
    for (auto &x : logits) {
        x = random(-1, 1);
    }
    std::vector<int> input_ids(seq_len);
    for (size_t i = 0; i < input_ids.size(); i++) {
        input_ids[i] = i;
    }

    auto fn = [&logits, &input_ids, penalty] {
        BaseModelForCausalLM::sampling_repetition_penalty(logits.data(), logits.data() + logits.size(), input_ids,
                                                          penalty);
    };
    auto elapsed_ms = timeit(fn, 2, 100);
    std::cout << "[" << ::testing::UnitTest::GetInstance()->current_test_info()->name() << "] " << elapsed_ms
              << " ms\n";
}

TEST(Sampling, Temperature) {
    constexpr float temp = 0.7;
    std::vector<float> logits(64);
    for (float &v : logits) {
        v = random();
    }
    // reference
    std::vector<float> target = logits;
    for (auto &v : target) {
        v /= temp;
    }
    // test
    BaseModelForCausalLM::sampling_temperature(logits.data(), logits.data() + logits.size(), temp);
    // compare
    for (size_t i = 0; i < logits.size(); i++) {
        EXPECT_FLOAT_EQ(logits[i], target[i]);
    }
}

TEST(Sampling, TopK) {
    constexpr int top_k = 20;
    std::vector<TokenIdScore> token_scores(64);
    for (size_t i = 0; i < token_scores.size(); i++) {
        token_scores[i] = TokenIdScore(i, random());
    }

    // reference
    std::vector<TokenIdScore> target = token_scores;
    std::sort(target.begin(), target.end(), std::greater<TokenIdScore>());
    target.resize(top_k);

    // test
    BaseModelForCausalLM::sampling_top_k(token_scores.data(), token_scores.data() + top_k,
                                         token_scores.data() + token_scores.size());
    token_scores.resize(top_k);

    // sort & compare
    EXPECT_TRUE(equal(extract_sorted_ids(token_scores), extract_sorted_ids(target)));
}

static void reference_top_p(std::vector<TokenIdScore> &token_scores, float top_p) {
    std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>());
    BaseModelForCausalLM::sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
    float cumsum = 0.f;
    for (size_t i = 0; i < token_scores.size(); i++) {
        cumsum += token_scores[i].score;
        if (cumsum >= top_p) {
            token_scores.resize(i + 1);
            break;
        }
    }
}

TEST(Sampling, TopP) {
    constexpr float top_p = 0.7;
    for (int i = 0; i < 10; i++) {
        std::vector<TokenIdScore> token_scores(1024);
        for (size_t i = 0; i < token_scores.size(); i++) {
            token_scores[i] = TokenIdScore(i, random());
        }

        // reference
        std::vector<TokenIdScore> target = token_scores;
        reference_top_p(target, top_p);
        EXPECT_TRUE(!token_scores.empty());

        // test
        TokenIdScore *pos =
            BaseModelForCausalLM::sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), top_p);
        token_scores.resize(pos - token_scores.data());

        // sort & compare
        auto output_ids = extract_sorted_ids(token_scores);
        auto target_ids = extract_sorted_ids(target);
        EXPECT_TRUE(equal(output_ids, target_ids)) << "size " << output_ids.size() << " vs " << target_ids.size();
    }
}

class ChatGLMTest : public ::testing::Test {
  protected:
    ModelContext ctx;

    void SetUp() override {
        ctx.dtype = GGML_TYPE_F32;
        ctx.ctx_w = make_unique_ggml_context(1024 * MB, nullptr, false);
        ctx.ctx_kv = make_unique_ggml_context(512 * MB, nullptr, false);
        ctx.ctx_b = make_unique_ggml_context(512 * MB, nullptr, false);
        ctx.scratch_buffer.resize(1 * MB);
        ctx.scratch = {0, ctx.scratch_buffer.size(), ctx.scratch_buffer.data()};
#ifdef GGML_USE_CUBLAS
        ggml_cuda_set_scratch_size(ctx.scratch_buffer.size());
#endif
        ctx.init_device_context();

        reset_cgraph();
    }

    void TearDown() override {
#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_scratch();
#endif
    }

    void reset_cgraph() { ctx.gf = {}; }

    void cpu_graph_compute(int n_threads) { ggml_graph_compute_helper(ctx.work_buffer, &ctx.gf, n_threads); }

    void device_graph_compute(int n_threads) {
#ifdef GGML_USE_METAL
        // ggml_metal_set_n_cb(ctx.ctx_metal.get(), n_threads);
        ggml_metal_graph_compute(ctx.ctx_metal.get(), &ctx.gf);
        // ggml_metal_get_tensor(ctx.ctx_metal.get(), output);
#else
        cpu_graph_compute(n_threads);
#endif
    }

    template <bool FALLBACK_CPU>
    float _perf_graph_compute_impl() {
        int num_threads = get_num_threads();
        auto fn = [this, num_threads] {
            if constexpr (FALLBACK_CPU) {
                cpu_graph_compute(num_threads);
            } else {
                device_graph_compute(num_threads);
            }
        };
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_METAL)
        return timeit(fn, 10, 100);
#else
        return timeit(fn, 1, 3);
#endif
    }

    float perf_cpu_graph_compute() { return _perf_graph_compute_impl<true>(); }
    float perf_device_graph_compute() { return _perf_graph_compute_impl<false>(); }

    template <typename Model>
    void test_model(const Model &model, const ModelConfig &config, const fs::path &data_path, int seq_len,
                    const std::vector<ggml_tensor *> &all_weights) {
        ASSERT_EQ(config.num_hidden_layers, 1);

        MappedFile mapped_file(data_path.string());
        char *ptr = mapped_file.data;

        tensor_to_device(model.layers[0].attention.k_cache);
        tensor_to_device(model.layers[0].attention.v_cache);

        ggml_tensor *x1 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_I32, seq_len);
        ggml_tensor *ref_y1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, config.hidden_size, seq_len);
        ggml_tensor *x2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_I32, 1);
        ggml_tensor *ref_y2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, config.hidden_size);
        ggml_tensor *x3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_I32, 1);
        ggml_tensor *ref_y3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, config.hidden_size);

        std::vector<ggml_tensor *> all_tensors = all_weights;
        all_tensors.insert(all_tensors.end(), {x1, ref_y1, x2, ref_y2, x3, ref_y3});

        std::vector<ggml_tensor *> cpu_tensors{model.word_embeddings.weight, x1, x2, x3};

        for (auto tensor : all_tensors) {
            ptr = read_tensor_data(ptr, tensor);
            if (std::find(cpu_tensors.begin(), cpu_tensors.end(), tensor) == cpu_tensors.end()) {
                tensor_to_device(tensor);
            }
        }

        ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

        // self attention
        {
            ggml_tensor *out_y1 = model.forward(&ctx, x1, 0, seq_len);
            EXPECT_EQ(out_y1->backend, ref_y1->backend);
            out_y1->backend = GGML_BACKEND_CPU;
            ggml_build_forward_expand(&ctx.gf, out_y1);
            device_graph_compute(1);

            expect_all_close(ref_y1, out_y1, 5e-4);
        }

        // cross attention
        reset_cgraph();
        {
            ggml_tensor *out_y2 = model.forward(&ctx, x2, seq_len, seq_len);
            EXPECT_EQ(out_y2->backend, ref_y2->backend);
            out_y2->backend = GGML_BACKEND_CPU;
            ggml_build_forward_expand(&ctx.gf, out_y2);
            device_graph_compute(1);

            expect_all_close(ref_y2, out_y2, 5e-4);
        }
        reset_cgraph();
        {
            ggml_tensor *out_y3 = model.forward(&ctx, x3, seq_len + 1, seq_len);
            EXPECT_EQ(out_y3->backend, ref_y3->backend);
            out_y3->backend = GGML_BACKEND_CPU;
            ggml_build_forward_expand(&ctx.gf, out_y3);
            device_graph_compute(1);

            expect_all_close(ref_y3, out_y3, 5e-4);
        }

        for (auto tensor : all_tensors) {
            tensor_to_cpu(tensor);
        }
        tensor_to_cpu(model.layers[0].attention.k_cache);
        tensor_to_cpu(model.layers[0].attention.v_cache);
    }
};

TEST_F(ChatGLMTest, Embedding) {
    float w_data[]{1.5410, -0.2934, -2.1788, 0.5684,  -1.0845, -1.3986,
                   0.4033, 0.8380,  -0.7193, -0.4033, -0.5966, 0.1820};
    int x_data[]{1, 3, 0, 2, 3};
    float y_data[]{0.5684,  -1.0845, -1.3986, -0.4033, -0.5966, 0.1820,  1.5410, -0.2934,
                   -2.1788, 0.4033,  0.8380,  -0.7193, -0.4033, -0.5966, 0.1820};

    ggml_tensor *x = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_I32, 5);
    memcpy(x->data, x_data, sizeof(x_data));
    Embedding model(&ctx, 4, 3);
    memcpy(model.weight->data, w_data, sizeof(w_data));
    ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 3, 5);
    ref->data = y_data;

    ggml_tensor *out = model.forward(&ctx, x);

    ggml_build_forward_expand(&ctx.gf, out);
    cpu_graph_compute(1);

    expect_all_close(ref, out);
}

TEST_F(ChatGLMTest, Linear) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/linear.data";
    MappedFile mapped_file(test_path.string());
    char *ptr = mapped_file.data;

    ggml_tensor *w = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 16);
    ptr = read_tensor_data(ptr, w);
    ggml_tensor *b = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 16);
    ptr = read_tensor_data(ptr, b);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 32, 2);
    ptr = read_tensor_data(ptr, x);
    ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 16, 2);
    ptr = read_tensor_data(ptr, ref);
    ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

    // GEMV data
    ggml_tensor *vx = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 32);
    memcpy(vx->data, x->data, 32 * sizeof(float));
    ggml_tensor *vref = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, 16);
    memcpy(vref->data, ref->data, 16 * sizeof(float));

    tensor_to_device(x);
    tensor_to_device(vx);

    struct TestCase {
        ggml_tensor *x;
        ggml_tensor *ref;
    };
    std::vector<TestCase> cases{{x, ref}, {vx, vref}};

    struct TestConfig {
        ggml_type dtype;
        float atol;
        float rtol;
    };
    std::vector<TestConfig> test_configs{
        {GGML_TYPE_F32, 1e-5, 0},
        {GGML_TYPE_F16, 1e-2, 5e-4},
        {GGML_TYPE_Q4_0, 1.0, 0.2},
    };

    for (const auto &config : test_configs) {
        ctx.dtype = config.dtype;
        Linear model(&ctx, 32, 16);

        if (config.dtype == GGML_TYPE_F32) {
            model.weight->data = w->data;
        } else if (config.dtype == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float *)w->data, (ggml_fp16_t *)model.weight->data, ggml_nelements(model.weight));
        } else if (config.dtype == GGML_TYPE_Q4_0) {
            int64_t hist[16]{};
            ggml_quantize_q4_0((float *)w->data, model.weight->data, ggml_nelements(w), w->ne[0], hist);
        } else {
            CHATGLM_THROW << "unsupported dtype " << config.dtype;
        }
        model.bias->data = b->data;
        tensor_to_device(model.weight);
        tensor_to_device(model.bias);

        for (const auto &c : cases) {
            reset_cgraph();
            ggml_tensor *out = model.forward(&ctx, c.x);
            EXPECT_EQ(out->backend, c.x->backend);
            out->backend = GGML_BACKEND_CPU;

            ggml_build_forward_expand(&ctx.gf, out);
            device_graph_compute(get_num_threads());

            EXPECT_EQ(out->type, GGML_TYPE_F32);
            expect_all_close(c.ref, out, config.atol, config.rtol);
        }

        tensor_to_cpu(model.weight);
        tensor_to_cpu(model.bias);
    }
    tensor_to_cpu(x);
    tensor_to_cpu(vx);
}

TEST_F(ChatGLMTest, BenchmarkLinear) {
    constexpr int M = 64, N = 1024, K = 1024 * 3;
    ctx.dtype = GGML_TYPE_F32;
    Linear m(&ctx, K, N);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, K, M);

    std::vector<ggml_tensor *> all_tensors{m.weight, m.bias, x};
    for (auto tensor : all_tensors) {
        random_fill(tensor);
        tensor_to_device(tensor);
    }

    ggml_tensor *y = m.forward(&ctx, x);
    ggml_build_forward_expand(&ctx.gf, y);
    std::cout << "[Benchmark] Linear " << ggml_type_name(ctx.dtype) << " time: " << perf_device_graph_compute()
              << " ms\n";

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, LayerNorm) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/layer_norm.data";
    MappedFile mapped_file(test_path.string());
    char *ptr = mapped_file.data;

    LayerNorm model(&ctx, 64);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);
    ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);

    std::vector<ggml_tensor *> all_tensors{model.weight, model.bias, x, ref};
    for (auto tensor : all_tensors) {
        ptr = read_tensor_data(ptr, tensor);
        tensor_to_device(tensor);
    }
    ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

    ggml_tensor *out = model.forward(&ctx, x);
    EXPECT_EQ(out->backend, x->backend);
    out->backend = GGML_BACKEND_CPU;

    ggml_build_forward_expand(&ctx.gf, out);
    device_graph_compute(get_num_threads());

    expect_all_close(ref, out);

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, BenchmarkLayerNorm) {
    constexpr int seq_len = 64;
    constexpr int hidden = 1024;

    LayerNorm m(&ctx, hidden);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden, seq_len);

    std::vector<ggml_tensor *> all_tensors{m.weight, m.bias, x};
    for (auto tensor : all_tensors) {
        random_fill(tensor);
        tensor_to_device(tensor);
    }

    ggml_tensor *y = m.forward(&ctx, x);
    ggml_build_forward_expand(&ctx.gf, y);
    std::cout << "[Benchmark] LayerNorm " << ggml_type_name(ctx.dtype) << " time: " << perf_device_graph_compute()
              << " ms\n";

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, RMSNorm) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/rms_norm.data";
    MappedFile mapped_file(test_path.string());
    char *ptr = mapped_file.data;

    RMSNorm model(&ctx, 64);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);
    ggml_tensor *ref = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 64, 3);

    std::vector<ggml_tensor *> all_tensors{model.weight, x, ref};
    for (auto tensor : all_tensors) {
        ptr = read_tensor_data(ptr, tensor);
        tensor_to_device(tensor);
    }
    ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

    ggml_tensor *out = model.forward(&ctx, x);
    EXPECT_EQ(out->backend, x->backend);
    out->backend = GGML_BACKEND_CPU;

    ggml_build_forward_expand(&ctx.gf, out);
    device_graph_compute(get_num_threads());

    expect_all_close(ref, out);

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, BenchmarkRMSNorm) {
    constexpr int seq_len = 64;
    constexpr int hidden = 1024;

    RMSNorm m(&ctx, hidden);
    ggml_tensor *x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden, seq_len);

    std::vector<ggml_tensor *> all_tensors{m.weight, x};
    for (auto tensor : all_tensors) {
        random_fill(tensor);
        tensor_to_device(tensor);
    }

    ggml_tensor *y = m.forward(&ctx, x);
    ggml_build_forward_expand(&ctx.gf, y);
    std::cout << "[Benchmark] RMSNorm " << ggml_type_name(ctx.dtype) << " time: " << perf_device_graph_compute()
              << " ms\n";

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, GLMModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm_model.data";

    ModelConfig config(
        ModelType::CHATGLM, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/8, /*num_hidden_layers=*/1, /*intermediate_size=*/128, /*norm_eps=*/1e-5f,
        /*hidden_act=*/ActivationType::GELU, /*use_qkv_bias=*/true, /*use_dense_bias=*/true,
        /*interleaved_qkv=*/true, /*use_alibi=*/false, /*rope_type=*/RopeType::CHATGLM, /*rope_dim_scale=*/-1,
        /*attn_mask_type=*/AttentionMaskType::CHATGLM,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    ChatGLMModel model(&ctx, config);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].input_layernorm.bias,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.query_key_value.bias,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].attention.dense.bias,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].post_attention_layernorm.bias,
                                           model.layers[0].mlp.dense_h_to_4h.weight,
                                           model.layers[0].mlp.dense_h_to_4h.bias,
                                           model.layers[0].mlp.dense_4h_to_h.weight,
                                           model.layers[0].mlp.dense_4h_to_h.bias,
                                           model.final_layernorm.weight,
                                           model.final_layernorm.bias};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, GLM2Model) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm2_model.data";

    ModelConfig config(
        ModelType::CHATGLM2, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/2, /*num_hidden_layers=*/1, /*intermediate_size=*/48, /*norm_eps=*/1e-5f,
        /*hidden_act=*/ActivationType::SILU, /*use_qkv_bias=*/true, /*use_dense_bias=*/false,
        /*interleaved_qkv=*/false, /*use_alibi=*/false, /*rope_type=*/RopeType::GPTJ, /*rope_dim_scale=*/2,
        /*attn_mask_type=*/AttentionMaskType::CAUSAL,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    ChatGLM2Model model(&ctx, config);

    tensor_to_device(model.layers[0].attention.k_cache);
    tensor_to_device(model.layers[0].attention.v_cache);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.query_key_value.bias,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].mlp.gate_proj.weight,
                                           model.layers[0].mlp.up_proj.weight,
                                           model.layers[0].mlp.down_proj.weight,
                                           model.final_layernorm.weight};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, GLM3Model) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm3_model.data";

    ModelConfig config(
        ModelType::CHATGLM3, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/2, /*num_hidden_layers=*/1, /*intermediate_size=*/48, /*norm_eps=*/1e-5f,
        /*hidden_act=*/ActivationType::SILU, /*use_qkv_bias=*/true, /*use_dense_bias=*/false,
        /*interleaved_qkv=*/false, /*use_alibi=*/false, /*rope_type=*/RopeType::GPTJ, /*rope_dim_scale=*/2,
        /*attn_mask_type=*/AttentionMaskType::CAUSAL,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    ChatGLM3Model model(&ctx, config);

    tensor_to_device(model.layers[0].attention.k_cache);
    tensor_to_device(model.layers[0].attention.v_cache);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.query_key_value.bias,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].mlp.gate_proj.weight,
                                           model.layers[0].mlp.up_proj.weight,
                                           model.layers[0].mlp.down_proj.weight,
                                           model.final_layernorm.weight};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, Baichuan7BModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/baichuan7b_model.data";

    ModelConfig config(
        ModelType::BAICHUAN7B, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/8, /*num_hidden_layers=*/1, /*intermediate_size=*/32 * 3, /*norm_eps=*/1e-6f,
        /*hidden_act=*/ActivationType::SILU, /*use_qkv_bias=*/false, /*use_dense_bias=*/false,
        /*interleaved_qkv=*/false, /*use_alibi=*/false, /*rope_type=*/RopeType::NEOX, /*rope_dim_scale=*/1,
        /*attn_mask_type=*/AttentionMaskType::CAUSAL,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    Baichuan7BModel model(&ctx, config);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].mlp.gate_proj.weight,
                                           model.layers[0].mlp.down_proj.weight,
                                           model.layers[0].mlp.up_proj.weight,
                                           model.final_layernorm.weight};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, Baichuan13BModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/baichuan13b_model.data";

    ModelConfig config(
        ModelType::BAICHUAN13B, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/8, /*num_hidden_layers=*/1, /*intermediate_size=*/32 * 3, /*norm_eps=*/1e-6f,
        /*hidden_act=*/ActivationType::SILU, /*use_qkv_bias=*/false, /*use_dense_bias=*/false,
        /*interleaved_qkv=*/false, /*use_alibi=*/true, /*rope_type=*/RopeType::DISABLED, /*rope_dim_scale=*/-1,
        /*attn_mask_type=*/AttentionMaskType::CAUSAL,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    Baichuan13BModel model(&ctx, config);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].mlp.gate_proj.weight,
                                           model.layers[0].mlp.down_proj.weight,
                                           model.layers[0].mlp.up_proj.weight,
                                           model.final_layernorm.weight};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, InternLMModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/internlm_model.data";

    ModelConfig config(
        ModelType::INTERNLM, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32, /*num_attention_heads=*/8,
        /*num_kv_heads=*/8, /*num_hidden_layers=*/1, /*intermediate_size=*/32 * 3, /*norm_eps=*/1e-6f,
        /*hidden_act=*/ActivationType::SILU, /*use_qkv_bias=*/true, /*use_dense_bias=*/true,
        /*interleaved_qkv=*/false, /*use_alibi=*/false, /*rope_type=*/RopeType::NEOX, /*rope_dim_scale=*/1,
        /*attn_mask_type=*/AttentionMaskType::CAUSAL,
        /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1, /*sep_token_id=*/-1,
        /*extra_eos_token_ids=*/{});

    constexpr int seq_len = 3;

    InternLMModel model(&ctx, config);

    std::vector<ggml_tensor *> all_weights{model.word_embeddings.weight,
                                           model.layers[0].input_layernorm.weight,
                                           model.layers[0].attention.query_key_value.weight,
                                           model.layers[0].attention.query_key_value.bias,
                                           model.layers[0].attention.dense.weight,
                                           model.layers[0].attention.dense.bias,
                                           model.layers[0].post_attention_layernorm.weight,
                                           model.layers[0].mlp.gate_proj.weight,
                                           model.layers[0].mlp.up_proj.weight,
                                           model.layers[0].mlp.down_proj.weight,
                                           model.final_layernorm.weight};

    test_model(model, config, data_path, seq_len, all_weights);
}

TEST_F(ChatGLMTest, quantize) {
    GTEST_SKIP() << "Skipping quantization data generation";
    float src_data[]{
        -1.1258e+00, -1.1524e+00, -2.5058e-01, -4.3388e-01, 8.4871e-01,  6.9201e-01,  -3.1601e-01, -2.1152e+00,
        3.2227e-01,  -1.2633e+00, 3.4998e-01,  3.0813e-01,  1.1984e-01,  1.2377e+00,  1.1168e+00,  -2.4728e-01,
        -1.3527e+00, -1.6959e+00, 5.6665e-01,  7.9351e-01,  5.9884e-01,  -1.5551e+00, -3.4136e-01, 1.8530e+00,
        7.5019e-01,  -5.8550e-01, -1.7340e-01, 1.8348e-01,  1.3894e+00,  1.5863e+00,  9.4630e-01,  -8.4368e-01,
        -6.1358e-01, 3.1593e-02,  -4.9268e-01, 2.4841e-01,  4.3970e-01,  1.1241e-01,  6.4079e-01,  4.4116e-01,
        -1.0231e-01, 7.9244e-01,  -2.8967e-01, 5.2507e-02,  5.2286e-01,  2.3022e+00,  -1.4689e+00, -1.5867e+00,
        -6.7309e-01, 8.7283e-01,  1.0554e+00,  1.7784e-01,  -2.3034e-01, -3.9175e-01, 5.4329e-01,  -3.9516e-01,
        -4.4622e-01, 7.4402e-01,  1.5210e+00,  3.4105e+00,  -1.5312e+00, -1.2341e+00, 1.8197e+00,  -5.5153e-01,
        -5.6925e-01, 9.1997e-01,  1.1108e+00,  1.2899e+00,  -1.4782e+00, 2.5672e+00,  -4.7312e-01, 3.3555e-01,
        -1.6293e+00, -5.4974e-01, -4.7983e-01, -4.9968e-01, -1.0670e+00, 1.1149e+00,  -1.4067e-01, 8.0575e-01,
        -9.3348e-02, 6.8705e-01,  -8.3832e-01, 8.9182e-04,  8.4189e-01,  -4.0003e-01, 1.0395e+00,  3.5815e-01,
        -2.4600e-01, 2.3025e+00,  -1.8817e+00, -4.9727e-02, -1.0450e+00, -9.5650e-01, 3.3532e-02,  7.1009e-01,
        1.6459e+00,  -1.3602e+00, 3.4457e-01,  5.1987e-01,  -2.6133e+00, -1.6965e+00, -2.2824e-01, 2.7995e-01,
        2.4693e-01,  7.6887e-02,  3.3801e-01,  4.5440e-01,  4.5694e-01,  -8.6537e-01, 7.8131e-01,  -9.2679e-01,
        -2.1883e-01, -2.4351e+00, -7.2915e-02, -3.3986e-02, 9.6252e-01,  3.4917e-01,  -9.2146e-01, -5.6195e-02,
        -6.2270e-01, -4.6372e-01, 1.9218e+00,  -4.0255e-01, 1.2390e-01,  1.1648e+00,  9.2337e-01,  1.3873e+00,
        -8.8338e-01, -4.1891e-01, -8.0483e-01, 5.6561e-01,  6.1036e-01,  4.6688e-01,  1.9507e+00,  -1.0631e+00,
        -7.7326e-02, 1.1640e-01,  -5.9399e-01, -1.2439e+00, -1.0209e-01, -1.0335e+00, -3.1264e-01, 2.4579e-01,
        -2.5964e-01, 1.1834e-01,  2.4396e-01,  1.1646e+00,  2.8858e-01,  3.8660e-01,  -2.0106e-01, -1.1793e-01,
        1.9220e-01,  -7.7216e-01, -1.9003e+00, 1.3068e-01,  -7.0429e-01, 3.1472e-01,  1.5739e-01,  3.8536e-01,
        9.6715e-01,  -9.9108e-01, 3.0161e-01,  -1.0732e-01, 9.9846e-01,  -4.9871e-01, 7.6111e-01,  6.1830e-01,
        3.1405e-01,  2.1333e-01,  -1.2005e-01, 3.6046e-01,  -3.1403e-01, -1.0787e+00, 2.4081e-01,  -1.3962e+00,
        -6.6144e-02, -3.5836e-01, -1.5616e+00, -3.5464e-01, 1.0811e+00,  1.3148e-01,  1.5735e+00,  7.8143e-01,
        -1.0787e+00, -7.2091e-01, 1.4708e+00,  2.7564e-01,  6.6678e-01,  -9.9439e-01, -1.1894e+00, -1.1959e+00,
        -5.5963e-01, 5.3347e-01,  4.0689e-01,  3.9459e-01,  1.7151e-01,  8.7604e-01,  -2.8709e-01, 1.0216e+00,
        -7.4395e-02, -1.0922e+00, 3.9203e-01,  5.9453e-01,  6.6227e-01,  -1.2063e+00, 6.0744e-01,  -5.4716e-01,
        1.1711e+00,  9.7496e-02,  9.6337e-01,  8.4032e-01,  -1.2537e+00, 9.8684e-01,  -4.9466e-01, -1.2830e+00,
        9.5522e-01,  1.2836e+00,  -6.6586e-01, 5.6513e-01,  2.8770e-01,  -3.3375e-02, -1.0619e+00, -1.1443e-01,
        -3.4334e-01, 1.5713e+00,  1.9161e-01,  3.7994e-01,  -1.4476e-01, 6.3762e-01,  -2.8129e-01, -1.3299e+00,
        -1.4201e-01, -5.3415e-01, -5.2338e-01, 8.6150e-01,  -8.8696e-01, 8.3877e-01,  1.1529e+00,  -1.7611e+00,
        -1.4777e+00, -1.7557e+00, 7.6166e-02,  -1.0786e+00, 1.4403e+00,  -1.1059e-01, 5.7686e-01,  -1.6917e-01,
        -6.4025e-02, 1.0384e+00,  9.0682e-01,  -4.7551e-01, -8.7074e-01, 1.4474e-01,  1.9029e+00,  3.9040e-01};

    ggml_tensor *src = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, 128, 2);
    memcpy(src->data, src_data, sizeof(src_data));

    // q8_0
    {
        ggml_tensor *q8_dst = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_Q8_0, 128, 2);
        int64_t hist[16]{};
        ggml_quantize_q8_0((float *)src->data, q8_dst->data, ggml_nelements(src), src->ne[0], hist);

        std::cout << "Q8: [";
        for (size_t i = 0; i < ggml_nbytes(q8_dst); i++) {
            std::cout << (i > 0 ? ", " : "") << (int)((char *)q8_dst->data)[i];
        }
        std::cout << "]\n";
    }
    // q4_0
    {
        ggml_tensor *q4_dst = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_Q4_0, 128, 2);
        int64_t hist[16]{};
        ggml_quantize_q4_0((float *)src->data, q4_dst->data, ggml_nelements(src), src->ne[0], hist);

        std::cout << "Q4_0: [";
        for (size_t i = 0; i < ggml_nbytes(q4_dst); i++) {
            std::cout << (i > 0 ? ", " : "") << (int)((char *)q4_dst->data)[i];
        }
        std::cout << "]\n";
    }
    // q4_1
    {
        ggml_tensor *q4_dst = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_Q4_1, 128, 2);
        int64_t hist[16]{};
        ggml_quantize_q4_1((float *)src->data, q4_dst->data, ggml_nelements(src), src->ne[0], hist);

        std::cout << "Q4_1: [";
        for (size_t i = 0; i < ggml_nbytes(q4_dst); i++) {
            std::cout << (i > 0 ? ", " : "") << (int)((char *)q4_dst->data)[i];
        }
        std::cout << "]\n";
    }
    // q5_0
    {
        ggml_tensor *q5_dst = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_Q5_0, 128, 2);
        int64_t hist[16]{};
        ggml_quantize_q5_0((float *)src->data, q5_dst->data, ggml_nelements(src), src->ne[0], hist);

        std::cout << "Q5_0: [";
        for (size_t i = 0; i < ggml_nbytes(q5_dst); i++) {
            std::cout << (i > 0 ? ", " : "") << (int)((char *)q5_dst->data)[i];
        }
        std::cout << "]\n";
    }
    // q5_1
    {
        ggml_tensor *q5_dst = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_Q5_1, 128, 2);
        int64_t hist[16]{};
        ggml_quantize_q5_1((float *)src->data, q5_dst->data, ggml_nelements(src), src->ne[0], hist);

        std::cout << "Q5_1: [";
        for (size_t i = 0; i < ggml_nbytes(q5_dst); i++) {
            std::cout << (i > 0 ? ", " : "") << (int)((char *)q5_dst->data)[i];
        }
        std::cout << "]\n";
    }
}

struct TokenizerTestCase {
    std::string prompt;
    std::vector<int> input_ids;
    bool skip_decode = false;
};

static void check_tokenizer(const BaseTokenizer *tokenizer, const std::vector<TokenizerTestCase> &cases) {
    for (const auto &c : cases) {
        // encode
        std::vector<int> input_ids = tokenizer->encode(c.prompt, 2048);
        EXPECT_TRUE(equal(input_ids, c.input_ids));
        if (!c.skip_decode) {
            // decode
            std::string output = tokenizer->decode(c.input_ids);
            EXPECT_EQ(output, c.prompt);
        }
    }
}

TEST(Pipeline, ChatGLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLMForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你好", {5, 74874, 130001, 130004}},
            {"[Round 0]\n问：你好\n答：你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。\n[Round "
             "1]\n问：晚上睡不着应该怎么办\n答：",
             {53,     6945,   5,      8,     42,    4,     64286,  12,    74874, 4,   67342,  12,    74874, 130328,
              130247, 130233, 130227, 35,    65806, 68241, 75890,  14132, 5388,  340, 11,     21,    222,   6,
              76693,  66877,  63852,  6,     66430, 68747, 102501, 63823, 4,     52,  6945,   5,     9,     42,
              4,      64286,  12,     65450, 83400, 64213, 66846,  4,     67342, 12,  130001, 130004}},
            {"def main():\n    print('hello world')\t# greeting",
             {1616, 594, 125936, 4, 130011, 2274, 89, 7283, 398, 125686, 130008, 61, 25672, 130001, 130004}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);
    }

    // prompter
    {
        EXPECT_EQ(ChatGLMTokenizer::build_prompt({{ChatMessage::ROLE_USER, "你好"}}), "你好");
        EXPECT_EQ(
            ChatGLMTokenizer::build_prompt({
                {ChatMessage::ROLE_USER, "你好"},
                {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"},
                {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
            }),
            "[Round 0]\n问：你好\n答：你好👋！我是人工智能助手 "
            "ChatGLM-6B，很高兴见到你，欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：");
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 2048;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::ostringstream oss;
        for (int i = 0; i < gen_config.max_context_length; i++) {
            oss << "你好";
        }
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, oss.str()}};
        pipeline.chat(messages, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。");
    }
}

TEST(Pipeline, ChatGLM2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm2-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM2 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLM2ForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你好", {64790, 64792, 36474, 54591}},
            {"[Round 1]\n\n问：你好\n\n答：",
             {64790, 64792, 790, 30951, 517, 30910, 30939, 30996, 13, 13, 54761, 31211, 39701, 13, 13, 55437, 31211}},
            {"[Round 1]\n\n问：你好\n\n答：你好👋！我是人工智能助手 "
             "ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。\n\n[Round 2]\n\n问：晚上睡不着应该怎么办\n\n答：",
             {64790, 64792, 790,   30951, 517,   30910, 30939, 30996, 13,    13,    54761, 31211, 39701,
              13,    13,    55437, 31211, 39701, 243,   162,   148,   142,   31404, 33030, 34797, 42481,
              22011, 10461, 30944, 30943, 30941, 30978, 30949, 31123, 48895, 35214, 54622, 31123, 32616,
              39905, 31901, 31639, 31155, 13,    13,    30995, 30951, 517,   30910, 30943, 30996, 13,
              13,    54761, 31211, 32820, 54266, 31876, 35153, 13,    13,    55437, 31211}},
            {"def main():\n    print('hello world')\t# greeting",
             {64790, 64792, 884, 1301, 9427, 13, 296, 4466, 2029, 15616, 30914, 993, 3387, 12, 31010, 30174}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);
    }

    // prompter
    {
        EXPECT_EQ(ChatGLM2Tokenizer::build_prompt({{ChatMessage::ROLE_USER, "你好"}}), "[Round 1]\n\n问：你好\n\n答：");
        EXPECT_EQ(
            ChatGLM2Tokenizer::build_prompt({
                {ChatMessage::ROLE_USER, "你好"},
                {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
            }),
            "[Round 1]\n\n问：你好\n\n答：你好👋！我是人工智能助手 "
            "ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。\n\n[Round 2]\n\n问：晚上睡不着应该怎么办\n\n答：");
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 2048;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::ostringstream oss;
        for (int i = 0; i < gen_config.max_context_length; i++) {
            oss << "你好";
        }
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, oss.str()}};
        pipeline.chat(messages, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。");
    }
}

static inline std::string read_text(const fs::path &path) {
    MappedFile mapped_file(path.string());
    return std::string(mapped_file.data, mapped_file.size);
}

TEST(Pipeline, ChatGLM3) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm3-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM3 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLM3ForCausalLM *>(pipeline.model.get()));

    const std::string system_tool_call =
        read_text(fs::path(__FILE__).parent_path() / "examples/system/function_call.txt");
    const std::string system_ci = read_text(fs::path(__FILE__).parent_path() / "examples/system/code_interpreter.txt");

    // tokenizer
    {
        std::vector<int> target_ids{64790, 64792, 36474, 54591};
        std::vector<int> input_ids = pipeline.tokenizer->encode("你好", 2048);
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_ids{64790, 64792, 64795, 30910, 13, 36474, 54591, 64796};
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_USER, "你好"},
            {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。"},
            {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_ids{64790, 64792, 64795, 30910, 13,    36474, 54591, 64796, 30910, 13,    36474, 54591,
                                    243,   162,   148,   142,   31404, 33030, 34797, 42481, 22011, 10461, 30944, 30966,
                                    30941, 30978, 30949, 31123, 48895, 35214, 54622, 31123, 32616, 39905, 31901, 31639,
                                    31155, 64795, 30910, 13,    30910, 32820, 54266, 31876, 35153, 64796};
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_SYSTEM, system_tool_call},
            {ChatMessage::ROLE_USER, "生成一个随机数"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_ids{
            64790, 64792, 64794, 30910, 13,    20115, 267,   1762,  2554,  362,   1077,  362,   344,   457,   30930,
            809,   431,   1675,  289,   267,   1762,  4159,  30954, 13,    30982, 13,    296,   30955, 16599, 30962,
            11228, 30962, 7311,  1306,  2932,  729,   13,    352,   30955, 2323,  2932,  449,   16599, 30962, 11228,
            30962, 7311,  1306,  1252,  13,    352,   30955, 16302, 2932,  449,   9398,  711,   260,   5402,  1276,
            1994,  30932, 268,   30930, 30912, 30930, 2288,  30995, 30940, 30996, 14819, 1994,  906,   2288,  30995,
            30939, 30996, 1252,  13,    352,   30955, 12209, 2932,  790,   13,    753,   30982, 13,    647,   30955,
            2323,  2932,  449,   24794, 1252,  13,    647,   30955, 16302, 2932,  449,   1036,  5402,  9352,  1050,
            422,   267,   17009, 1252,  13,    647,   30955, 3543,  2932,  449,   592,   1252,  13,    647,   30955,
            20379, 2932,  2033,  13,    753,   4143,  13,    753,   30982, 13,    647,   30955, 2323,  2932,  449,
            7855,  1252,  13,    647,   30955, 16302, 2932,  449,   1036,  2288,  290,   267,   7383,  3859,  1252,
            13,    647,   30955, 3543,  2932,  449,   30912, 16471, 30995, 592,   30932, 558,   30996, 1252,  13,
            647,   30955, 20379, 2932,  2033,  13,    753,   30983, 13,    352,   30996, 13,    296,   4143,  13,
            296,   30955, 752,   30962, 27564, 2932,  729,   13,    352,   30955, 2323,  2932,  449,   752,   30962,
            27564, 1252,  13,    352,   30955, 16302, 2932,  449,   4867,  267,   1465,  5100,  332,   4256,  17654,
            30962, 2323,  31040, 1252,  13,    352,   30955, 12209, 2932,  790,   13,    753,   30982, 13,    647,
            30955, 2323,  2932,  449,   17654, 30962, 2323,  1252,  13,    647,   30955, 16302, 2932,  449,   1036,
            1462,  290,   267,   1911,  289,   330,   580,   266,   819,   1252,  13,    647,   30955, 3543,  2932,
            449,   2069,  1252,  13,    647,   30955, 20379, 2932,  2033,  13,    753,   30983, 13,    352,   30996,
            13,    296,   30983, 13,    30983, 64795, 30910, 13,    30910, 36454, 31623, 37853, 54744, 64796};
        EXPECT_EQ(input_ids, target_ids);
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 2048;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::ostringstream oss;
        for (int i = 0; i < gen_config.max_context_length; i++) {
            oss << "你好";
        }
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, oss.str()}};
        pipeline.chat(messages, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。");
    }

    // tool call
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_SYSTEM, system_tool_call},
            {ChatMessage::ROLE_USER, "生成一个随机数"},
        };
        {
            ChatMessage output = pipeline.chat(messages, gen_config);
            EXPECT_EQ(output.role, ChatMessage::ROLE_ASSISTANT);
            EXPECT_EQ(output.content, "```python\n"
                                      "tool_call(seed=42, range=(0, 100))\n"
                                      "```");
            messages.emplace_back(std::move(output));
        }
        messages.emplace_back(ChatMessage::ROLE_OBSERVATION, "22");
        {
            ChatMessage output = pipeline.chat(messages, gen_config);
            EXPECT_EQ(output.role, ChatMessage::ROLE_ASSISTANT);
            EXPECT_EQ(output.content, "根据您的要求，我使用随机数生成器API生成了一个在0和100之间的随机数，结果为22。");
        }
    }

    // code interpreter
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_SYSTEM, system_ci},
            {ChatMessage::ROLE_USER, "列出100以内的所有质数"},
        };
        {
            ChatMessage output = pipeline.chat(messages, gen_config);
            EXPECT_EQ(output.role, ChatMessage::ROLE_ASSISTANT);
            EXPECT_EQ(output.content, "好的，我会为您列出100以内的所有质数。\n\n质数是指只能被1和它本身整除的大于1"
                                      "的整数。例如，2、3、5、7等都是质数。\n\n让我们开始吧！");
            EXPECT_EQ(output.tool_calls.front().code.input, R"(```python
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Get all prime numbers up to 100
primes_upto_100 = [i for i in range(2, 101) if is_prime(i)]
primes_upto_100
```)");
            messages.emplace_back(std::move(output));
        }
        messages.emplace_back(
            ChatMessage::ROLE_OBSERVATION,
            "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]");
        {
            ChatMessage output = pipeline.chat(messages, gen_config);
            EXPECT_EQ(output.role, ChatMessage::ROLE_ASSISTANT);
            EXPECT_EQ(output.content, R"(100以内的所有质数为：

$$
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 
$$)");
        }
    }
}

TEST(Pipeline, CodeGeeX2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "codegeex2-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping CodeGeeX2 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLM2ForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"# language: Python\n# write a bubble sort function\n",
             {64790, 64792, 31010, 3239, 30954, 16719, 13, 31010, 3072, 260, 17338, 3482, 1674, 13}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);
    }

    // generate
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        gen_config.max_length = 256;

        std::string prompt = "# language: Python\n# write a bubble sort function\n";
        std::string target = R"(

def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list


print(bubble_sort([5, 4, 3, 2, 1])))";

        std::string output = pipeline.generate(prompt, gen_config);
        EXPECT_EQ(output, target);
    }
}

TEST(Pipeline, Baichuan13B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "baichuan-13b-chat-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping Baichuan13B e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<Baichuan13BForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你是谁", {9875, 21797}},
            {"我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮"
             "助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问",
             {6323,  31161, 31745, 32213, 31175, 14830, 72,    16347, 31745, 32213, 6358,  31135, 14823, 31212, 8823,
              5114,  7234,  14830, 72,    31182, 1231,  31188, 8627,  1696,  3823,  5536,  76,    17133, 1766,  76,
              16345, 11769, 72,    4090,  13169, 8385,  76,    31840, 32195, 31135, 4137,  2781,  3317,  31188, 2285,
              1910,  73,    6011,  31169, 4315,  1766,  72,    1231,  11533, 31490, 31182, 21934}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);

        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_USER, "你好呀"},
            {ChatMessage::ROLE_ASSISTANT, "你好！很高兴和你交流。请问有什么我可以帮助你的吗？"},
            {ChatMessage::ROLE_USER, "你叫什么名字？"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_input_ids{195,   9875, 31213, 32889, 196,  9875,  31213, 74,   17318, 31906,
                                          14822, 5536, 73,    20389, 7713, 31182, 1231,  4090, 2689,  31763,
                                          75,    195,  9875,  32177, 1534, 10240, 75,    196};
        EXPECT_TRUE(equal(input_ids, target_input_ids));
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 512;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::vector<int> input_ids(gen_config.max_context_length, 128);
        pipeline.generate(input_ids, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        gen_config.repetition_penalty = 1.1;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好呀"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好！很高兴见到你。请问有什么我可以帮助你的吗？");
    }
}

TEST(Pipeline, Baichuan2_7B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "baichuan2-7b-chat-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping Baichuan2-7B e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<Baichuan7BForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你是谁", {92067}},
            {"我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮"
             "助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问",
             {6461, 70335, 92366, 9528, 65,    10879, 70335, 3932, 92333, 8832,  92414, 5034,
              3133, 5002,  9528,  65,   28756, 92385, 5243,  1697, 2559,  3341,  69,    10474,
              1754, 69,    9036,  7356, 65,    2716,  7499,  4892, 69,    24816, 92333, 2693,
              2089, 23672, 1940,  1760, 66,    4173,  23181, 1754, 65,    65351, 39975, 14590}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);

        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_USER, "你好呀"},
            {ChatMessage::ROLE_ASSISTANT, "你好！很高兴和你交流。请问有什么问题我可以帮助你解决吗？"},
            {ChatMessage::ROLE_USER, "你叫什么名字？"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_input_ids{195, 16829, 94278, 196,   16829, 67,    52160, 10329, 3341,
                                          66,  23216, 5817,  1754,  92392, 21777, 92430, 2740,  93122,
                                          68,  195,   92430, 93410, 1747,  6642,  68,    196};
        EXPECT_TRUE(equal(input_ids, target_input_ids));
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 2048;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::vector<int> input_ids(gen_config.max_context_length, 128);
        pipeline.generate(input_ids, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        gen_config.repetition_penalty = 1.05;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好呀"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好！很高兴为你服务。请问有什么问题我可以帮助你解决？");
    }
}

TEST(Pipeline, Baichuan2_13B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "baichuan2-13b-chat-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping Baichuan2-13B e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<Baichuan13BForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你是谁", {92067}},
            {"我是百川大模型，是由百川智能的工程师们创造的大语言模型，我可以和人类进行自然交流、解答问题、协助创作，帮"
             "助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问",
             {6461, 70335, 92366, 9528, 65,    10879, 70335, 3932, 92333, 8832,  92414, 5034,
              3133, 5002,  9528,  65,   28756, 92385, 5243,  1697, 2559,  3341,  69,    10474,
              1754, 69,    9036,  7356, 65,    2716,  7499,  4892, 69,    24816, 92333, 2693,
              2089, 23672, 1940,  1760, 66,    4173,  23181, 1754, 65,    65351, 39975, 14590}}};
        check_tokenizer(pipeline.tokenizer.get(), cases);

        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_USER, "你好呀"},
            {ChatMessage::ROLE_ASSISTANT, "你好！很高兴和你交流。请问有什么我可以帮助你的吗？"},
            {ChatMessage::ROLE_USER, "你叫什么名字？"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->encode_messages(messages, 2048);
        std::vector<int> target_input_ids{195,   16829, 94278, 196,   16829, 67,  52160, 10329, 3341, 66,   23216, 5817,
                                          92392, 21777, 2193,  93122, 68,    195, 92430, 93410, 1747, 6642, 68,    196};
        EXPECT_TRUE(equal(input_ids, target_input_ids));
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        gen_config.repetition_penalty = 1.05;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好呀"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好！很高兴见到你。请问有什么我可以帮助你的吗？");
    }
}

TEST(Pipeline, InternLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "internlm-chat-7b-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping InternLM e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<InternLMForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        std::vector<TokenizerTestCase> cases{
            {"你好", {1, 76379}},
            {"<|User|>:你好<eoh>\n<|Bot|>:你好，有什么我可以帮助你的吗？<eoa>\n<|User|>:晚上睡不着应该怎么办<eoh>\n<|"
             "Bot|>:",
             {1,     333,   352,   1621,  352,   27232,  76379, 103027, 364,    333,   352, 23845, 352,  27232,
              76379, 98899, 68408, 73159, 67566, 67513,  61056, 99050,  103028, 364,   333, 352,   1621, 352,
              27232, 67891, 76046, 67551, 68573, 103027, 364,   333,    352,    23845, 352, 27232},
             true}};
        check_tokenizer(pipeline.tokenizer.get(), cases);
    }

    // prompter
    {
        EXPECT_EQ(InternLMTokenizer::build_prompt({{ChatMessage::ROLE_USER, "你好"}}), "<|User|>:你好<eoh>\n<|Bot|>:");
        EXPECT_EQ(InternLMTokenizer::build_prompt({
                      {ChatMessage::ROLE_USER, "你好"},
                      {ChatMessage::ROLE_ASSISTANT, "你好，有什么我可以帮助你的吗？"},
                      {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
                  }),
                  "<|User|>:你好<eoh>\n<|Bot|>:你好，有什么我可以帮助你的吗？<eoa>\n<|User|>:晚上睡不着应该怎么办<eoh>"
                  "\n<|Bot|>:");
    }

    // memory test
    {
        GenerationConfig gen_config;
        gen_config.max_length = 2048;
        gen_config.max_context_length = gen_config.max_length - 1;
        gen_config.do_sample = false;

        std::vector<int> input_ids(gen_config.max_context_length, 128);
        pipeline.generate(input_ids, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好，有什么我可以帮助你的吗？");
    }
}

static void run_benchmark(const fs::path &model_path) {
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping benchmark test (model " << model_path << " not found)";
    }

    ggml_time_init();
    int64_t start_ms = ggml_time_ms();
    Pipeline pipeline(model_path.string());
    int64_t load_model_ms = ggml_time_ms() - start_ms;

    start_ms = ggml_time_ms();
    std::vector<ChatMessage> messages{
        {ChatMessage::ROLE_USER, "你好"},
        {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"},
        {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
    };

    GenerationConfig gen_config;
    gen_config.do_sample = false;
    gen_config.num_threads = get_num_threads();

    PerfStreamer streamer;
    start_ms = ggml_time_ms();
    pipeline.chat(messages, gen_config, &streamer);
    int64_t gen_s = (ggml_time_ms() - start_ms) / 1000.f;

    std::cout << "======== benchmark results for " << model_path.filename() << " ========\n"
              << "using #threads: " << gen_config.num_threads << "\n"
              << "model loaded within: " << load_model_ms << " ms\n"
              << "generation finished within: " << gen_s << " s\n"
              << streamer.to_string() << "\n"
              << "===========================================================\n";
}

TEST(Benchmark, ChatGLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, ChatGLM2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm2-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, Baichuan2_7B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "baichuan2-7b-chat-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, Baichuan2_13B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "baichuan2-13b-chat-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, InternLM7B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "internlm-chat-7b-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, InternLM20B) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "internlm-chat-20b-ggml.bin";
    run_benchmark(model_path);
}

} // namespace chatglm
