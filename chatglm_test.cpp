#include "chatglm.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include <ggml-cuda.h>
#endif

namespace chatglm {

namespace fs = std::filesystem;

static inline void expect_all_close(ggml_tensor *a, ggml_tensor *b, float atol = 1e-5f, float rtol = 0.f) {
    ASSERT_EQ(a->type, b->type);
    ASSERT_EQ(a->type, GGML_TYPE_F32);
    ASSERT_EQ(ggml_nelements(a), ggml_nelements(b));

    int64_t numel = ggml_nelements(a);

    std::vector<float> a_buf(numel);
    ggml_backend_tensor_get(a, a_buf.data(), 0, numel * sizeof(float));

    std::vector<float> b_buf(numel);
    ggml_backend_tensor_get(b, b_buf.data(), 0, numel * sizeof(float));

    float max_abs_diff = 0.f;
    float max_rel_diff = 0.f;
    int64_t num_mismatch = 0;
    for (int64_t i = 0; i < numel; i++) {
        float ai = a_buf[i];
        float bi = b_buf[i];
        EXPECT_TRUE(std::isfinite(ai) && std::isfinite(bi));
        float abs_diff = std::abs(ai - bi);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        if (abs_diff >= atol + rtol * std::abs(bi)) {
            num_mismatch++;
        }
        float rel_diff = abs_diff / std::abs(bi);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }
    EXPECT_TRUE(num_mismatch == 0) << "Tensors are not close!\n\n"
                                   << "Mismatched elements: " << num_mismatch << " / " << numel << " ("
                                   << num_mismatch * 100 / numel << "%)\n"
                                   << "Greatest absolute difference: " << max_abs_diff << " (up to " << std::scientific
                                   << atol << std::defaultfloat << " allowed)\n"
                                   << "Greatest relative difference: " << max_rel_diff << " (up to " << std::scientific
                                   << rtol << std::defaultfloat << " allowed)\n";
}

static inline void read_backend_tensor_data(std::istream &is, ggml_tensor *tensor) {
    std::vector<no_init<char>> buf(ggml_nbytes(tensor));
    CHATGLM_CHECK(is.read((char *)buf.data(), buf.size()));
    ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());
}

static inline void _fill(ggml_tensor *tensor, const std::vector<float> &values) {
    switch (tensor->type) {
    case GGML_TYPE_F32: {
        ggml_backend_tensor_set(tensor, values.data(), 0, sizeof(float) * values.size());
    } break;
    case GGML_TYPE_F16: {
        std::vector<ggml_fp16_t> fp16_buf(values.size());
        ggml_fp32_to_fp16_row(values.data(), fp16_buf.data(), fp16_buf.size());
        ggml_backend_tensor_set(tensor, fp16_buf.data(), 0, fp16_buf.size());
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0: {
        std::vector<no_init<char>> q_buf(ggml_nbytes(tensor));
        ggml_quantize_chunk(tensor->type, values.data(), q_buf.data(), 0, ggml_nelements(tensor) / tensor->ne[0],
                            tensor->ne[0], nullptr);
        ggml_backend_tensor_set(tensor, q_buf.data(), 0, ggml_nbytes(tensor));
    } break;
    default:
        CHATGLM_THROW << "unsupported dtype " << tensor->type;
    }
}

static inline float random() { return rand() / (float)RAND_MAX; }

static inline float random(float lo, float hi) { return lo + random() * (hi - lo); }

static inline void random_(ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    for (float &v : values) {
        v = random();
    }
    _fill(tensor, values);
}

static inline float randn() {
    thread_local std::random_device rd{};
    thread_local std::mt19937 gen{rd()};
    std::normal_distribution<float> d;
    return d(gen);
}

static inline void randn_(ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    for (float &v : values) {
        v = randn();
    }
    _fill(tensor, values);
}

// return elapsed time in milliseconds
static inline float timeit(std::function<void()> fn, int warmup, int active) {
    for (int i = 0; i < warmup; i++) {
        fn();
    }

#ifdef GGML_USE_CUDA
    CHATGLM_CHECK_CUDA(cudaDeviceSynchronize());
#endif
    int64_t start_us = ggml_time_us();
    for (int i = 0; i < active; i++) {
        fn();
    }
#ifdef GGML_USE_CUDA
    CHATGLM_CHECK_CUDA(cudaDeviceSynchronize());
#endif
    int64_t end_us = ggml_time_us();

    float elapsed_ms = (end_us - start_us) / 1000.f;
    return elapsed_ms / active;
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
    EXPECT_EQ(extract_sorted_ids(token_scores), extract_sorted_ids(target));
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
        EXPECT_EQ(output_ids, target_ids);
    }
}

static inline ggml_tensor *ggml_new_tensor_like(ggml_context *ctx, ggml_tensor *tensor) {
    return ggml_new_tensor(ctx, tensor->type, ggml_n_dims(tensor), tensor->ne);
}

class ChatGLMTest : public ::testing::Test {
  protected:
    std::unique_ptr<ModelContext> mctx_;

    void SetUp() override { mctx_ = std::make_unique<ModelContext>(); }

    float perf_graph_compute() {
        auto fn = [this] {
            CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);
        };
        if (ggml_backend_is_cpu(mctx_->backend.get())) {
            return timeit(fn, 1, 3);
        } else {
            return timeit(fn, 10, 100);
        }
    }

    template <typename Model>
    void test_model(Model &model, const ModelConfig &config, const fs::path &data_path, int seq_len,
                    const std::vector<ggml_tensor *> &all_weights) {
        ASSERT_EQ(config.num_hidden_layers, 1);

        std::ifstream ifs(data_path, std::ios::binary);
        ASSERT_TRUE(ifs) << "cannot open file " << data_path;

        ggml_tensor *x1 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, seq_len);
        ggml_tensor *ref_y1 = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size, seq_len);
        ggml_tensor *x2 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, 1);
        ggml_tensor *ref_y2 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size);
        ggml_tensor *x3 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, 1);
        ggml_tensor *ref_y3 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size);

        std::vector<ggml_tensor *> all_tensors = all_weights;
        all_tensors.insert(all_tensors.end(), {x1, ref_y1, x2, ref_y2, x3, ref_y3});

        ggml_tensor *past_key_values = nullptr;
        if (config.num_virtual_tokens > 0) {
            const int head_size = config.hidden_size / config.num_attention_heads;
            past_key_values =
                ggml_new_tensor_4d(mctx_->ctx_b.get(), GGML_TYPE_F16, head_size, config.num_virtual_tokens,
                                   config.num_key_value_heads, config.num_hidden_layers * 2); // [l * 2, #h, v, d]
        }

        auto buf_b =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
        auto buf_w =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

        if (config.num_virtual_tokens > 0) {
            read_backend_tensor_data(ifs, past_key_values);
            model.load_prefix_cache(config, past_key_values);
        }

        for (auto tensor : all_tensors) {
            read_backend_tensor_data(ifs, tensor);
        }
        ASSERT_TRUE(ifs.peek() == EOF);

        auto input_ids_to_vec = [](ggml_tensor *input_ids) {
            std::vector<int> input_ids_vec(ggml_nelements(input_ids));
            ggml_backend_tensor_get(input_ids, input_ids_vec.data(), 0, ggml_nbytes(input_ids));
            return input_ids_vec;
        };

        std::vector<int> input_ids;

        // prefill
        {
            std::vector<int> x1_vec = input_ids_to_vec(x1);
            input_ids.insert(input_ids.end(), x1_vec.begin(), x1_vec.end());
            ggml_graph_clear(mctx_->gf);
            ggml_tensor *out_y1 = model.forward(mctx_.get(), x1, nullptr, input_ids, 0);
            ggml_build_forward_expand(mctx_->gf, out_y1);
            CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
            model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, 0, seq_len);
            CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

            expect_all_close(ref_y1, out_y1, 5e-4);
        }
        // decode
        {
            std::vector<int> x2_vec = input_ids_to_vec(x2);
            input_ids.insert(input_ids.end(), x2_vec.begin(), x2_vec.end());
            ggml_graph_clear(mctx_->gf);
            ggml_tensor *out_y2 = model.forward(mctx_.get(), x2, nullptr, input_ids, seq_len);
            ggml_build_forward_expand(mctx_->gf, out_y2);
            CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
            model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, seq_len, seq_len);
            CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

            expect_all_close(ref_y2, out_y2, 5e-4);
        }
        {
            std::vector<int> x3_vec = input_ids_to_vec(x3);
            input_ids.insert(input_ids.end(), x3_vec.begin(), x3_vec.end());
            ggml_graph_clear(mctx_->gf);
            ggml_tensor *out_y3 = model.forward(mctx_.get(), x3, nullptr, input_ids, seq_len + 1);
            ggml_build_forward_expand(mctx_->gf, out_y3);
            CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
            model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, seq_len + 1, seq_len);
            CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

            expect_all_close(ref_y3, out_y3, 5e-4);
        }
    }
};

TEST_F(ChatGLMTest, Embedding) {
    float w_data[]{1.5410, -0.2934, -2.1788, 0.5684,  -1.0845, -1.3986,
                   0.4033, 0.8380,  -0.7193, -0.4033, -0.5966, 0.1820};
    int x_data[]{1, 3, 0, 2, 3};
    float y_data[]{0.5684,  -1.0845, -1.3986, -0.4033, -0.5966, 0.1820,  1.5410, -0.2934,
                   -2.1788, 0.4033,  0.8380,  -0.7193, -0.4033, -0.5966, 0.1820};

    ggml_tensor *x = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, 5);
    Embedding model(mctx_.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor *ref = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 3, 5);

    auto buf_b = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buf_w = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
    ggml_backend_tensor_set(model.weight, w_data, 0, sizeof(w_data));
    ggml_backend_tensor_set(ref, y_data, 0, sizeof(y_data));

    ggml_tensor *out = model.forward(mctx_.get(), x);

    ggml_build_forward_expand(mctx_->gf, out);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
    CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

    expect_all_close(ref, out);
}

TEST_F(ChatGLMTest, Linear) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/linear.data";
    std::ifstream ifs(test_path, std::ios::binary);
    ASSERT_TRUE(ifs) << "cannot open file " << test_path;

    ggml_tensor *w = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 32);
    ggml_tensor *b = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, 32);
    ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 2);
    ggml_tensor *ref = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 32, 2);

    ggml_tensor *vec_x = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64);
    ggml_tensor *vec_ref = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, 32);

    auto buf_b = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));

    read_backend_tensor_data(ifs, w);
    read_backend_tensor_data(ifs, b);
    read_backend_tensor_data(ifs, x);
    read_backend_tensor_data(ifs, ref);

    read_backend_tensor_data(ifs, vec_x);
    read_backend_tensor_data(ifs, vec_ref);

    ASSERT_TRUE(ifs.peek() == EOF);

    struct TestCase {
        ggml_tensor *x;
        ggml_tensor *ref;
    };
    std::vector<TestCase> cases{{x, ref}, {vec_x, vec_ref}};

    struct TestConfig {
        ggml_type dtype;
        float atol;
        float rtol;
    };
    std::vector<TestConfig> test_configs{
        {GGML_TYPE_F32, 1e-5, 0},   {GGML_TYPE_F16, 1e-2, 5e-4}, {GGML_TYPE_Q8_0, 0.2, 5e-4},
        {GGML_TYPE_Q5_0, 1.5, 0.1}, {GGML_TYPE_Q5_1, 1.5, 0.1},  {GGML_TYPE_Q4_1, 2.0, 0.2},
        {GGML_TYPE_Q4_0, 2.0, 0.2},
    };

    for (const auto &config : test_configs) {
        Linear model(mctx_.get(), config.dtype, 64, 32);
        auto buf_w =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

        auto ctx = make_unique_ggml_context(1024 * 1024, nullptr, false);
        ggml_tensor *w_cpu = ggml_new_tensor_like(ctx.get(), w);
        ggml_backend_tensor_get(w, w_cpu->data, 0, ggml_nbytes(w));

        ggml_tensor *wq_cpu = ggml_new_tensor_2d(ctx.get(), config.dtype, w_cpu->ne[0], w_cpu->ne[1]);
        if (config.dtype == GGML_TYPE_F32) {
            wq_cpu = w_cpu;
        } else if (config.dtype == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float *)w_cpu->data, (ggml_fp16_t *)wq_cpu->data, ggml_nelements(w_cpu));
        } else {
            ggml_quantize_chunk(config.dtype, (float *)w_cpu->data, wq_cpu->data, 0, w_cpu->ne[1], w_cpu->ne[0],
                                nullptr);
        }
        ggml_backend_tensor_set(model.weight, wq_cpu->data, 0, ggml_nbytes(model.weight));
        ggml_backend_tensor_copy(b, model.bias);

        for (const auto &c : cases) {
            ggml_graph_clear(mctx_->gf);
            ggml_tensor *out = model.forward(mctx_.get(), c.x);

            ggml_build_forward_expand(mctx_->gf, out);
            CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
            CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

            expect_all_close(c.ref, out, config.atol, config.rtol);
        }
    }
}

TEST_F(ChatGLMTest, BenchmarkLinear) {
    constexpr int M = 64, N = 1024, K = 1024 * 3;
    std::vector<ggml_type> dtypes{GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_Q8_0, GGML_TYPE_Q5_1,
                                  GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0};
    for (ggml_type dtype : dtypes) {
        mctx_ = std::make_unique<ModelContext>();

        Linear m(mctx_.get(), dtype, K, N);
        ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, K, M);

        ggml_tensor *y = m.forward(mctx_.get(), x);
        ggml_build_forward_expand(mctx_->gf, y);
        CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));

        std::vector<ggml_tensor *> all_tensors{m.weight, m.bias, x};
        for (auto tensor : all_tensors) {
            randn_(tensor);
        }

        std::cout << "[Benchmark] Linear " << ggml_type_name(dtype) << " time: " << perf_graph_compute() << " ms\n";
    }
}

TEST_F(ChatGLMTest, LayerNorm) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/layer_norm.data";
    std::ifstream ifs(test_path, std::ios::binary);
    ASSERT_TRUE(ifs) << "cannot open file " << test_path;

    LayerNorm model(mctx_.get(), 64);
    ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 3);
    ggml_tensor *ref = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 3);

    auto buf_b = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buf_w = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    std::vector<ggml_tensor *> all_tensors{model.weight, model.bias, x, ref};
    for (auto tensor : all_tensors) {
        read_backend_tensor_data(ifs, tensor);
    }
    ASSERT_TRUE(ifs.peek() == EOF);

    ggml_tensor *out = model.forward(mctx_.get(), x);

    ggml_build_forward_expand(mctx_->gf, out);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
    CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

    expect_all_close(ref, out);
}

TEST_F(ChatGLMTest, BenchmarkLayerNorm) {
    constexpr int seq_len = 64;
    constexpr int hidden = 1024;

    LayerNorm m(mctx_.get(), hidden);
    ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, hidden, seq_len);

    auto buffer =
        unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buffer_w =
        unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    std::vector<ggml_tensor *> all_tensors{m.weight, m.bias, x};
    for (auto tensor : all_tensors) {
        random_(tensor);
    }

    ggml_tensor *y = m.forward(mctx_.get(), x);
    ggml_build_forward_expand(mctx_->gf, y);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
    std::cout << "[Benchmark] LayerNorm " << ggml_type_name(GGML_TYPE_F32) << " time: " << perf_graph_compute()
              << " ms\n";
}

TEST_F(ChatGLMTest, RMSNorm) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/rms_norm.data";
    std::ifstream ifs(test_path, std::ios::binary);
    ASSERT_TRUE(ifs) << "cannot open file " << test_path;

    RMSNorm model(mctx_.get(), 64);
    ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 3);
    ggml_tensor *ref = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, 64, 3);

    auto buf_b = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buf_w = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    std::vector<ggml_tensor *> all_tensors{model.weight, x, ref};
    for (auto tensor : all_tensors) {
        read_backend_tensor_data(ifs, tensor);
    }
    ASSERT_TRUE(ifs.peek() == EOF);

    ggml_tensor *out = model.forward(mctx_.get(), x);

    ggml_build_forward_expand(mctx_->gf, out);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
    CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

    expect_all_close(ref, out);
}

TEST_F(ChatGLMTest, BenchmarkRMSNorm) {
    constexpr int seq_len = 64;
    constexpr int hidden = 1024;

    RMSNorm m(mctx_.get(), hidden);
    ggml_tensor *x = ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, hidden, seq_len);

    auto buffer =
        unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buffer_w =
        unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    std::vector<ggml_tensor *> all_tensors{m.weight, x};
    for (auto tensor : all_tensors) {
        random_(tensor);
    }

    ggml_tensor *y = m.forward(mctx_.get(), x);
    ggml_build_forward_expand(mctx_->gf, y);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
    std::cout << "[Benchmark] RMSNorm " << ggml_type_name(GGML_TYPE_F32) << " time: " << perf_graph_compute()
              << " ms\n";
}

TEST_F(ChatGLMTest, GLMModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm_model.data";

    ModelConfig config(ModelType::CHATGLM, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/8, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/128, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/0,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLMModel model(mctx_.get(), config);

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

TEST_F(ChatGLMTest, GLMPTuningV2Model) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm_ptuning_v2_model.data";

    ModelConfig config(ModelType::CHATGLM, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/8, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/128, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/5,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLMModel model(mctx_.get(), config);

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

    ModelConfig config(ModelType::CHATGLM2, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/0,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLM2Model model(mctx_.get(), config);

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

    ModelConfig config(ModelType::CHATGLM3, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/0,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLM3Model model(mctx_.get(), config);

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

TEST_F(ChatGLMTest, GLM3PTuningV2Model) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm3_ptuning_v2_model.data";

    ModelConfig config(ModelType::CHATGLM3, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/5,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLM3Model model(mctx_.get(), config);

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

TEST_F(ChatGLMTest, GLM4Model) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm4_model.data";

    ModelConfig config(ModelType::CHATGLM4, GGML_TYPE_F32, /*vocab_size=*/5, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/1e-5f, /*rope_theta=*/10000.f, /*num_virtual_tokens=*/0,
                       /*max_length=*/8, /*bos_token_id=*/-1, /*eos_token_id=*/-1, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/-1, /*eoi_token_id=*/-1, /*extra_eos_token_ids=*/{},
                       /*vision=*/{});

    constexpr int seq_len = 3;

    ChatGLM4Model model(mctx_.get(), config);

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

TEST_F(ChatGLMTest, GLM4VModelText) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm4v_model_text.data";

    VisionModelConfig vision(GGML_TYPE_F32, ActivationType::GELU, 128, 28, 3, 56, 1e-6, 2, 1, 17, 7, 8);

    ModelConfig config(ModelType::CHATGLM4V, GGML_TYPE_F32, /*vocab_size=*/8, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/0.00000015625f, /*rope_theta=*/10000.f,
                       /*num_virtual_tokens=*/0,
                       /*max_length=*/16, /*bos_token_id=*/-1, /*eos_token_id=*/4, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/2, /*eoi_token_id=*/4, /*extra_eos_token_ids=*/{},
                       /*vision=*/vision);

    constexpr int seq_len = 3;

    ChatGLM4VModel model(mctx_.get(), config);

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

TEST_F(ChatGLMTest, GLM4VModel) {
    fs::path data_path = fs::path(__FILE__).parent_path() / "tests/data/glm4v_model.data";

    VisionModelConfig vision(GGML_TYPE_F32, ActivationType::GELU, 128, 28, 3, 56, 1e-6, 2, 1, 17, 7, 8);

    ModelConfig config(ModelType::CHATGLM4V, GGML_TYPE_F32, /*vocab_size=*/8, /*hidden_size=*/32,
                       /*num_attention_heads=*/8, /*num_key_value_heads=*/2, /*num_hidden_layers=*/1,
                       /*intermediate_size=*/48, /*norm_eps=*/0.00000015625f, /*rope_theta=*/10000.f,
                       /*num_virtual_tokens=*/0,
                       /*max_length=*/16, /*bos_token_id=*/-1, /*eos_token_id=*/4, /*pad_token_id=*/-1,
                       /*sep_token_id=*/-1, /*boi_token_id=*/2, /*eoi_token_id=*/4, /*extra_eos_token_ids=*/{},
                       /*vision=*/vision);

    std::ifstream fin(data_path, std::ios::binary);
    ASSERT_TRUE(fin) << "cannot open file " << data_path;

    ChatGLM4VModel model(mctx_.get(), config);

    ggml_tensor *images = ggml_new_tensor_3d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.vision.image_size,
                                             config.vision.image_size, config.vision.in_channels);

    const int vision_tokens = model.num_vision_tokens();

    const int seq_len = 6;
    ggml_tensor *x1 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, seq_len);
    ggml_tensor *ref_y1 =
        ggml_new_tensor_2d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size, seq_len - 1 + vision_tokens);
    ggml_tensor *x2 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, 1);
    ggml_tensor *ref_y2 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size);
    ggml_tensor *x3 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, 1);
    ggml_tensor *ref_y3 = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.hidden_size);

    auto buf_b = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_b.get(), mctx_->backend.get()));
    auto buf_w = unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx_->ctx_w.get(), mctx_->backend.get()));

    std::vector<ggml_tensor *> all_tensors{
        // vision
        model.vision.patch_embedding.cls_embedding,
        model.vision.patch_embedding.proj.weight,
        model.vision.patch_embedding.proj.bias,
        model.vision.patch_embedding.position_embedding.weight,
        model.vision.transformer.layers[0].input_layernorm.weight,
        model.vision.transformer.layers[0].input_layernorm.bias,
        model.vision.transformer.layers[0].attention.query_key_value.weight,
        model.vision.transformer.layers[0].attention.query_key_value.bias,
        model.vision.transformer.layers[0].attention.dense.weight,
        model.vision.transformer.layers[0].attention.dense.bias,
        model.vision.transformer.layers[0].mlp.dense_h_to_4h.weight,
        model.vision.transformer.layers[0].mlp.dense_h_to_4h.bias,
        model.vision.transformer.layers[0].mlp.dense_4h_to_h.weight,
        model.vision.transformer.layers[0].mlp.dense_4h_to_h.bias,
        model.vision.transformer.layers[0].post_attention_layernorm.weight,
        model.vision.transformer.layers[0].post_attention_layernorm.bias,
        model.vision.conv.weight,
        model.vision.conv.bias,
        model.vision.linear_proj.weight,
        model.vision.norm1.weight,
        model.vision.norm1.bias,
        model.vision.glu.gate_proj.weight,
        model.vision.glu.up_proj.weight,
        model.vision.glu.down_proj.weight,
        model.vision.boi,
        model.vision.eoi,
        // text
        model.word_embeddings.weight,
        model.layers[0].input_layernorm.weight,
        model.layers[0].attention.query_key_value.weight,
        model.layers[0].attention.query_key_value.bias,
        model.layers[0].attention.dense.weight,
        model.layers[0].post_attention_layernorm.weight,
        model.layers[0].mlp.gate_proj.weight,
        model.layers[0].mlp.up_proj.weight,
        model.layers[0].mlp.down_proj.weight,
        model.final_layernorm.weight,
        // inputs & outputs
        images,
        x1,
        ref_y1,
        x2,
        ref_y2,
        x3,
        ref_y3,
    };

    for (ggml_tensor *tensor : all_tensors) {
        read_backend_tensor_data(fin, tensor);
    }
    ASSERT_TRUE(fin.peek() == EOF);

    auto input_ids_to_vec = [](ggml_tensor *input_ids) {
        std::vector<int> input_ids_vec(ggml_nelements(input_ids));
        ggml_backend_tensor_get(input_ids, input_ids_vec.data(), 0, ggml_nbytes(input_ids));
        return input_ids_vec;
    };

    std::vector<int> input_ids;

    // prefill
    {
        std::vector<int> x1_vec = input_ids_to_vec(x1);
        input_ids.insert(input_ids.end(), x1_vec.begin(), x1_vec.end());
        ggml_graph_clear(mctx_->gf);
        ggml_tensor *out_y1 = model.forward(mctx_.get(), x1, images, input_ids, 0);
        ggml_build_forward_expand(mctx_->gf, out_y1);
        CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
        model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, 0, ref_y1->ne[1]);
        CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

        expect_all_close(ref_y1, out_y1, 1e-2);
    }
    // decode
    {
        std::vector<int> x2_vec = input_ids_to_vec(x2);
        input_ids.insert(input_ids.end(), x2_vec.begin(), x2_vec.end());
        ggml_graph_clear(mctx_->gf);
        ggml_tensor *out_y2 =
            model.forward(mctx_.get(), x2, images, input_ids, seq_len - 1 + model.num_vision_tokens());
        ggml_build_forward_expand(mctx_->gf, out_y2);
        CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
        model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, seq_len, seq_len);
        CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

        expect_all_close(ref_y2, out_y2, 1e-2);
    }
    {
        std::vector<int> x3_vec = input_ids_to_vec(x3);
        input_ids.insert(input_ids.end(), x3_vec.begin(), x3_vec.end());
        ggml_graph_clear(mctx_->gf);
        ggml_tensor *out_y3 = model.forward(mctx_.get(), x3, images, input_ids, seq_len + model.num_vision_tokens());
        ggml_build_forward_expand(mctx_->gf, out_y3);
        CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));
        model.set_graph_inputs(mctx_->gf, input_ids, std::nullopt, seq_len + 1, seq_len);
        CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

        expect_all_close(ref_y3, out_y3, 1e-2);
    }
}

TEST_F(ChatGLMTest, quantize) {
    const float src_data[]{
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

    const int q8_ref[]{
        68,  36,   -68, -69,  -15, -26,  51,  42,  -19, -127, 19,  -76, 21,  19,  7,   74,  67,   -15,  -81, -102, 34,
        48,  36,   -93, -20,  111, 45,   -35, -10, 11,  83,   95,  57,  -51, -32, 38,  -23, 1,    -18,  9,   16,   4,
        24,  16,   -4,  30,   -11, 2,    19,  86,  -55, -59,  -25, 33,  39,  7,   -9,  -15, 20,   -15,  -17, 28,   57,
        127, -57,  -46, 68,   -21, 45,   37,  -28, 46,  55,   64,  -73, 127, -23, 17,  -81, -27,  -24,  -25, -53,  55,
        -7,  40,   -5,  34,   -41, 0,    42,  -20, 51,  18,   -12, 114, -93, -2,  -52, -47, 2,    35,   69,  37,   80,
        -66, 17,   25,  -127, -82, -11,  14,  12,  4,   16,   22,  22,  -42, 38,  -45, -11, -118, -4,   -2,  47,   17,
        -45, -3,   -30, -23,  93,  -20,  6,   57,  45,  67,   -35, 35,  -58, -27, -52, 37,  40,   30,   127, -69,  -5,
        8,   -39,  -81, -7,   -67, -20,  16,  -17, 8,   16,   76,  19,  25,  -13, -8,  13,  -50,  -124, 9,   -46,  20,
        10,  25,   88,  34,   78,  -80,  24,  -9,  81,  -40,  61,  50,  25,  17,  -10, 29,  -25,  -87,  19,  -113, -5,
        -29, -126, -29, 87,   11,  127,  63,  -87, -58, 119,  22,  54,  -80, -96, -97, 45,  33,   -55,  53,  40,   39,
        17,  87,   -28, 101,  -7,  -108, 39,  59,  66,  -119, 60,  -54, 116, 10,  95,  83,  -124, 98,   -49, -127, 95,
        127, -66,  56,  28,   -3,  -105, -11, -84, 35,  -23,  105, 13,  25,  -10, 43,  -19, -89,  -9,   -36, -35,  57,
        -59, 56,   77,  -118, -99, -117, 5,   -72, 96,  -7,   38,  -11, -4,  69,  61,  -32, -58,  10,   127, 26};

    const int q4_0_ref[]{59,   52,   52,   36,   -89,  -74,  -85,  43,   119,  -16,  -71,  99,   121,  -103, -40, -19,
                         -52,  87,   -46,  -74,  -87,  104,  105,  -121, -105, -104, 118,  -105, -104, 102,  73,  8,
                         -57,  -77,  75,   -100, 34,   -75,  -118, 101,  -75,  -124, 93,   -112, 89,   119,  -99, 26,
                         -23,  -118, -69,  -75,  -120, 101,  58,   53,   125,  20,   -119, -118, -80,  -109, 87,  -119,
                         105,  120,  -23,  121,  -119, -59,  -70,  -59,  -50,  -77,  -100, -118, 123,  54,   117, 102,
                         -112, -116, 120,  -72,  -6,   125,  -72,  124,  121,  103,  75,   -78,  -125, -83,  -10, -87,
                         51,   123,  4,    69,   -42,  -57,  25,   118,  90,   -35,  -25,  -17,  34,   -79,  27,  117,
                         37,   54,   -9,   35,   -70,  -14,  40,   15,   -58,  68,   100,  -113, -12,  -101, -99, -77,
                         -23,  -15,  -121, -42,  41,   -123, 105,  -98,  -119, 74,   74,   -92,  -52,  116,  3,   111};

    const int q4_1_ref[]{60,   52,   59,   -64,  52,  36,   -89,  -74,  -85,  43,   119,  -16,  -71,  99,   121, -103,
                         -40,  -19,  -52,  87,   85,  53,   89,   -66,  51,   117,  -125, 86,   70,   69,   103, 70,
                         52,   119,  -108, -11,  6,   28,   -96,  48,   -65,  52,   -121, -65,  100,  -103, 74,  107,
                         -111, 95,   -91,  -121, 97,  -28,  5,    101,  51,   58,   102,  -103, -42,  52,   58,  -63,
                         -114, 20,   -118, -102, -64, -93,  104,  -118, 121,  121,  -6,   122,  -102, -58,  -53, -42,
                         28,   52,   -102, -65,  100, -122, -124, -54,  -102, -103, 127,  115,  -121, 72,   5,   -125,
                         87,   -109, -122, -104, -80, 50,   63,   -66,  124,  99,   9,    103,  -36,  -123, -5,  -70,
                         41,   72,   -9,   -103, -74, 50,   41,   33,   122,  49,   34,   -67,  -28,  -117, -38, -54,
                         9,    -35,  86,   13,   -41, -15,  74,   -69,  -101, 112,  27,   116,  -47,  51,   11,  -65,
                         22,   14,   -120, 57,   -41, 122,  -90,  114,  119,  -75,  -75,  91,   68,   -117, -4,  -112};

    const int q5_0_ref[]{
        59,  48,  48,  125,  -100, 121, 103, 55,   78,   109,  86,   69,  -34, -32, 98,  -58, -13, 18,   -79,  -55,
        120, -82, -46, -78,  7,    -51, -79, -79,  51,   -64,  -78,  -1,  30,  47,  -35, 46,  32,  -36,  -111, 0,
        126, 101, 119, 55,   34,   -79, 81,  95,   45,   125,  20,   -54, 89,  8,   -71, 32,  -93, -18,  42,   35,
        -61, 3,   119, 105,  1,    -53, 58,  49,   -115, 95,   -68,  -12, -6,  24,  2,   3,   96,  38,   -81,  2,
        -62, -48, -62, -29,  19,   123, 101, -118, -50,  -81,  -121, 125, -63, 22,  39,  -13, -25, 107,  -21,  -36,
        32,  25,  -31, 111,  -11,  -6,  97,  -40,  -13,  -34,  75,   -82, 42,  -76, 15,  -29, 22,  74,   -3,   65,
        86,  -11, 8,   -118, -67,  126, 17,  -36,  -109, -85,  -50,  -50, 34,  -83, 65,  -93, -48, -28,  23,   -7,
        75,  107, -2,  69,   100,  -13, 65,  14,   -117, -103, -56,  15,  -40, 23,  -99, -81, -47, -105, -85,  25,
        -61, -13, -2,  -99,  65,   27,  -78, 27,   17,   116,  -124, 73,  119, -7,  6,   -33};

    const int q5_1_ref[]{25,   48,   59,   -64, 48,   125,  -100, 121,  104,  56,   95,  125,  87,   70,  -18,  -16,
                         99,   -57,  -13,  35,  -79,  -38,  -119, -81,  41,   49,   89,  -66,  0,    32,  4,    76,
                         102,  -6,   7,    -69, -115, 123,  -34,  125,  121,  -17,  56,  -6,   13,   40,  81,   96,
                         -104, 48,   -121, -65, 46,   -96,  -46,  -126, -55,  36,   117, -42,  51,   -81, 74,   15,
                         -78,  -39,  10,   -38, 102,  101,  -36,  35,   -82,  48,   58,  -63,  -51,  95,  -67,  -12,
                         13,   25,   20,   37,  -128, 70,   -64,  20,   -28,  -14,  -12, -11,  53,   -84, -121, -68,
                         -13,  47,   -102, -65, 120,  -126, 62,   -23,  -40,  12,   25,  -108, 36,   35,  -17,  -25,
                         31,   -112, 11,   5,   -82,  39,   29,   33,   121,  46,   63,  -66,  -43,  75,  -16,  28,
                         -7,   -58,  2,    -50, -87,  27,   -9,   118,  83,   -126, -18, 35,   108,  101, 66,   66,
                         76,   45,   34,   -67, -66,  92,   47,   27,   -23,  22,   -76, -92,  2,    -70, -84,  12,
                         -65,  -14,  116,  103, 55,   -15,  55,   -23,  -112, 47,   11,  -65,  46,   104, 84,   -26,
                         44,   12,   1,    98,  -66,  -28,  77,   -44,  -18,  -118, 122, -74,  -121, 6,   -7,   32};

    auto ctx = make_unique_ggml_context(1024 * 1024, nullptr, false);

    ggml_tensor *src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 2);
    memcpy(src->data, src_data, sizeof(src_data));

    [[maybe_unused]] auto qtensor_to_string = [](ggml_tensor *tensor) {
        std::ostringstream oss;
        oss << "Q8: [";
        for (size_t i = 0; i < ggml_nbytes(tensor); i++) {
            oss << (i > 0 ? ", " : "") << (int)((char *)tensor->data)[i];
        }
        oss << "]";
        return oss.str();
    };

    // q8_0
    {
        ggml_tensor *q8_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 128, 2);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, (float *)src->data, q8_dst->data, 0, src->ne[1], src->ne[0], nullptr);
        // std::cout << qtensor_to_string(q8_dst) << '\n';
        EXPECT_TRUE(memcmp(q8_dst->data, q8_ref, sizeof(q8_ref)));
    }
    // q4_0
    {
        ggml_tensor *q4_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, 128, 2);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, (float *)src->data, q4_dst->data, 0, src->ne[1], src->ne[0], nullptr);
        // std::cout << qtensor_to_string(q4_dst) << '\n';
        EXPECT_TRUE(memcmp(q4_dst->data, q4_0_ref, sizeof(q4_0_ref)));
    }
    // q4_1
    {
        ggml_tensor *q4_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_1, 128, 2);
        ggml_quantize_chunk(GGML_TYPE_Q4_1, (float *)src->data, q4_dst->data, 0, src->ne[1], src->ne[0], nullptr);
        // std::cout << qtensor_to_string(q4_dst) << '\n';
        EXPECT_TRUE(memcmp(q4_dst->data, q4_1_ref, sizeof(q4_1_ref)));
    }
    // q5_0
    {
        ggml_tensor *q5_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q5_0, 128, 2);
        ggml_quantize_chunk(GGML_TYPE_Q5_0, (float *)src->data, q5_dst->data, 0, src->ne[1], src->ne[0], nullptr);
        // std::cout << qtensor_to_string(q5_dst) << '\n';
        EXPECT_TRUE(memcmp(q5_dst->data, q5_0_ref, sizeof(q5_0_ref)));
    }
    // q5_1
    {
        ggml_tensor *q5_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q5_1, 128, 2);
        ggml_quantize_chunk(GGML_TYPE_Q5_1, (float *)src->data, q5_dst->data, 0, src->ne[1], src->ne[0], nullptr);
        // std::cout << qtensor_to_string(q5_dst) << '\n';
        EXPECT_TRUE(memcmp(q5_dst->data, q5_1_ref, sizeof(q5_1_ref)));
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
        EXPECT_EQ(input_ids, c.input_ids);
        if (!c.skip_decode) {
            // decode
            std::string output = tokenizer->decode(c.input_ids);
            EXPECT_EQ(output, c.prompt);
        }
    }
}

static void check_chat_format(const Pipeline &pipeline) {
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 1;
    EXPECT_THROW(
        {
            pipeline.chat({{ChatMessage::ROLE_USER, "user"}, {ChatMessage::ROLE_USER, "user"}}, gen_config);
        },
        std::runtime_error);
    EXPECT_THROW({ pipeline.chat({{ChatMessage::ROLE_ASSISTANT, "assistant"}}, gen_config); }, std::runtime_error);
    EXPECT_THROW(
        {
            pipeline.chat({{ChatMessage::ROLE_USER, "user"}, {ChatMessage::ROLE_ASSISTANT, "assistant"}}, gen_config);
        },
        std::runtime_error);
    // never throw with system prompt
    pipeline.chat({{ChatMessage::ROLE_SYSTEM, "system"}, {ChatMessage::ROLE_USER, "user"}}, gen_config);
}

TEST(Pipeline, ChatGLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLMTokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLMForCausalLM *>(pipeline.model.get()));

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
        EXPECT_EQ(ChatGLMTokenizer::apply_chat_template_text({{ChatMessage::ROLE_USER, "你好"}}), "你好");
        EXPECT_EQ(
            ChatGLMTokenizer::apply_chat_template_text({
                {ChatMessage::ROLE_USER, "你好"},
                {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"},
                {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
            }),
            "[Round 0]\n问：你好\n答：你好👋！我是人工智能助手 "
            "ChatGLM-6B，很高兴见到你，欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：");
    }

    // chat
    {
        check_chat_format(pipeline);
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。");
    }
}

TEST(Pipeline, ChatGLM2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm2-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM2 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLM2Tokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLM2ForCausalLM *>(pipeline.model.get()));

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
        EXPECT_EQ(ChatGLM2Tokenizer::apply_chat_template_text({{ChatMessage::ROLE_USER, "你好"}}),
                  "[Round 1]\n\n问：你好\n\n答：");
        EXPECT_EQ(
            ChatGLM2Tokenizer::apply_chat_template_text({
                {ChatMessage::ROLE_USER, "你好"},
                {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
            }),
            "[Round 1]\n\n问：你好\n\n答：你好👋！我是人工智能助手 "
            "ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。\n\n[Round 2]\n\n问：晚上睡不着应该怎么办\n\n答：");
    }

    // chat
    {
        check_chat_format(pipeline);
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。");
    }
}

static inline std::string read_text(const fs::path &path) {
    std::ifstream ifs(path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

TEST(Pipeline, ChatGLM3) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm3-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM3 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLM3Tokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLM3ForCausalLM *>(pipeline.model.get()));

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
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
        std::vector<int> target_ids{64790, 64792, 64795, 30910, 13, 36474, 54591, 64796};
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{
            {ChatMessage::ROLE_USER, "你好"},
            {ChatMessage::ROLE_ASSISTANT, "你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。"},
            {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"},
        };
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
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
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
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

    // chat
    {
        // check_chat_format(pipeline);
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
            {ChatMessage::ROLE_USER, "求出100以内的所有质数"},
        };
        {
            ChatMessage output = pipeline.chat(messages, gen_config);
            EXPECT_EQ(output.role, ChatMessage::ROLE_ASSISTANT);
            EXPECT_EQ(output.content,
                      R"(质数是只能被1和自身整除的正整数。我们可以通过遍历1到100的所有数字来找出100以内的所有质数。

下面是找出100以内的所有质数的Python代码：)");
            EXPECT_EQ(output.tool_calls.at(0).code.input, R"(```python
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
            EXPECT_EQ(output.content,
                      R"(100以内的所有质数为：

$$
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 
$$)");
        }
    }
}

TEST(Pipeline, ChatGLM4) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm4-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM4 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLM4Tokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLM4ForCausalLM *>(pipeline.model.get()));
    auto tokenizer = dynamic_cast<ChatGLM4Tokenizer *>(pipeline.tokenizer.get());

    // const std::string system_tool_call =
    //     read_text(fs::path(__FILE__).parent_path() / "examples/system/function_call.txt");
    // const std::string system_ci = read_text(fs::path(__FILE__).parent_path() /
    // "examples/system/code_interpreter.txt");

    // tiktoken
    {
        // taken from:
        // https://github.com/ggerganov/llama.cpp/blob/4bfe50f741479c1df1c377260c3ff5702586719e/convert-hf-to-gguf.py#L413
        const std::string chktxt =
            "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis "
            "concatenated) "
            "✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 "
            "កាន់តែពិសេសអាច😁 "
            "?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've "
            "been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL";

        const std::vector<int> ref_ids{
            198,    4710,   14721, 65020,  7847,   1572,  2303,   78043,  10942, 9281,   248,    222,   320,    8251,
            8,      26440,  114,   124564, 9281,   234,   104,    30423,  320,   35495,  98226,  96714, 8,      25442,
            227,    11157,  99,    247,    9281,   99,    247,    220,    18,    220,    100702, 220,   121577, 220,
            121577, 18,     220,   121577, 100702, 220,   121577, 121577, 220,   121577, 121577, 18,    220,    121577,
            121577, 100702, 220,   18,     13,     18,    220,    18,     496,   18,     220,    18,    1112,   18,
            220,    20833,  222,   96709,  241,    44002, 233,    20833,  237,   44002,  224,    20833, 244,    20833,
            115,    20833,  253,   44002,  223,    20833, 253,    20833,  95,    96709,  227,    74764, 223,    937,
            101446, 98319,  22320, 98538,  118901, 19,    99082,  16,     98411, 21168,  55088,  52883, 18625,  131040,
            13065,  146335, 78377, 3355,   4605,   4605,  13865,  13865,  73022, 3014,   3014,   28052, 17066,  2928,
            26524,  7646,   358,   3003,   1012,   364,   83,     813,    566,   594,    1052,   11,    364,    787,
            498,    2704,   30,    364,    44,     537,   2704,   358,    3278,  1281,   432,    11,    364,    35,
            498,    1075,   1045,  15231,  30,     1205,  6,      42368,  264,   63409,  43};

        const std::vector<int> out_ids = tokenizer->core_bpe.encode_ordinary(chktxt);
        EXPECT_EQ(ref_ids, out_ids);
    }
    {
        const std::string text = R"(
```c++
#include <iostream>

int main() {
    printf("hello world\n");    // say hello
}
```

```python
if __name__ == '__main__':
    print('hello world')        # say hello
```
)";
        const std::vector<int> ref_ids = {198,   73022, 66,    22879, 1067,  366,   9661,  1339, 396,   1887, 368,
                                          341,   262,   4100,  445,   14978, 1879,  1699,  5038, 262,   442,  1977,
                                          23745, 198,   532,   13865, 19288, 73022, 12663, 198,  333,   1304, 606,
                                          563,   621,   12106, 3817,  16165, 262,   1173,  492,  14978, 1879, 863,
                                          286,   671,   1977,  23745, 198,   13865, 3989};
        const std::vector<int> out_ids = tokenizer->core_bpe.encode_ordinary(text);
        EXPECT_EQ(ref_ids, out_ids);
    }
    // tokenizer
    {
        std::vector<int> target_ids{151331, 151333, 109377};
        std::vector<int> input_ids = pipeline.tokenizer->encode("你好", 2048);
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
        std::vector<int> target_ids{151331, 151333, 151336, 198, 109377, 151337};
        EXPECT_EQ(input_ids, target_ids);
    }
    {
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"},
                                          {ChatMessage::ROLE_ASSISTANT, "你好👋！很高兴见到你，有什么可以帮助你的吗？"},
                                          {ChatMessage::ROLE_USER, "晚上睡不着应该怎么办"}};
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
        std::vector<int> target_ids{151331, 151333, 151336, 198,    109377, 151337, 198,    109377, 9281,  239,
                                    233,    6313,   118295, 103810, 98406,  3837,   101665, 110368, 99444, 99212,
                                    11314,  151336, 198,    101160, 120410, 99379,  103298, 151337};
        EXPECT_EQ(input_ids, target_ids);
    }
    // {
    //     std::vector<ChatMessage> messages{
    //         {ChatMessage::ROLE_SYSTEM, system_tool_call},
    //         {ChatMessage::ROLE_USER, "生成一个随机数"},
    //     };
    //     std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
    //     std::vector<int> target_ids{
    //         64790, 64792, 64794, 30910, 13,    20115, 267,   1762,  2554,  362,   1077,  362,   344,   457,   30930,
    //         809,   431,   1675,  289,   267,   1762,  4159,  30954, 13,    30982, 13,    296,   30955, 16599, 30962,
    //         11228, 30962, 7311,  1306,  2932,  729,   13,    352,   30955, 2323,  2932,  449,   16599, 30962, 11228,
    //         30962, 7311,  1306,  1252,  13,    352,   30955, 16302, 2932,  449,   9398,  711,   260,   5402,  1276,
    //         1994,  30932, 268,   30930, 30912, 30930, 2288,  30995, 30940, 30996, 14819, 1994,  906,   2288,  30995,
    //         30939, 30996, 1252,  13,    352,   30955, 12209, 2932,  790,   13,    753,   30982, 13,    647,   30955,
    //         2323,  2932,  449,   24794, 1252,  13,    647,   30955, 16302, 2932,  449,   1036,  5402,  9352,  1050,
    //         422,   267,   17009, 1252,  13,    647,   30955, 3543,  2932,  449,   592,   1252,  13,    647,   30955,
    //         20379, 2932,  2033,  13,    753,   4143,  13,    753,   30982, 13,    647,   30955, 2323,  2932,  449,
    //         7855,  1252,  13,    647,   30955, 16302, 2932,  449,   1036,  2288,  290,   267,   7383,  3859,  1252,
    //         13,    647,   30955, 3543,  2932,  449,   30912, 16471, 30995, 592,   30932, 558,   30996, 1252,  13,
    //         647,   30955, 20379, 2932,  2033,  13,    753,   30983, 13,    352,   30996, 13,    296,   4143,  13,
    //         296,   30955, 752,   30962, 27564, 2932,  729,   13,    352,   30955, 2323,  2932,  449,   752,   30962,
    //         27564, 1252,  13,    352,   30955, 16302, 2932,  449,   4867,  267,   1465,  5100,  332,   4256,  17654,
    //         30962, 2323,  31040, 1252,  13,    352,   30955, 12209, 2932,  790,   13,    753,   30982, 13,    647,
    //         30955, 2323,  2932,  449,   17654, 30962, 2323,  1252,  13,    647,   30955, 16302, 2932,  449,   1036,
    //         1462,  290,   267,   1911,  289,   330,   580,   266,   819,   1252,  13,    647,   30955, 3543,  2932,
    //         449,   2069,  1252,  13,    647,   30955, 20379, 2932,  2033,  13,    753,   30983, 13,    352,   30996,
    //         13,    296,   30983, 13,    30983, 64795, 30910, 13,    30910, 36454, 31623, 37853, 54744, 64796};
    //     EXPECT_EQ(input_ids, target_ids);
    // }

    // chat
    {
        // check_chat_format(pipeline);
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！很高兴能帮助你，有什么问题或者需要帮助的地方吗？");
    }
}

TEST(Pipeline, CodeGeeX2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/codegeex2-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping CodeGeeX2 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLM2Tokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLM2ForCausalLM *>(pipeline.model.get()));

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

def bubble_sort(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


print(bubble_sort([5, 4, 3, 2, 1])))";

        std::string output = pipeline.generate(prompt, gen_config);
        EXPECT_EQ(output, target);
    }
}

TEST(Pipeline, ChatGLM4V) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm4v-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM4V e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    ASSERT_TRUE(dynamic_cast<ChatGLM4Tokenizer *>(pipeline.tokenizer.get()));
    ASSERT_TRUE(dynamic_cast<ChatGLM4VForCausalLM *>(pipeline.model.get()));

    // tokenizer
    {
        fs::path image_path = fs::path(__FILE__).parent_path() / "examples/03-Confusing-Pictures.jpg";
        Image image = Image::open(image_path.string());
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "描述这张图片", image}};
        std::vector<int> target_ids{151331, 151333, 151336, 198,    151339, 151329,
                                    151340, 100395, 108627, 100736, 151337};
        std::vector<int> input_ids = pipeline.tokenizer->apply_chat_template(messages, 2048);
        EXPECT_EQ(input_ids, target_ids);
    }
    // chat
    {
        // check_chat_format(pipeline);
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "你好"}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content, "你好👋！很高兴见到你，欢迎问我任何问题。");
    }
    // chat with image
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        fs::path image_path = fs::path(__FILE__).parent_path() / "examples/03-Confusing-Pictures.jpg";
        Image image = Image::open(image_path.string());
        std::vector<ChatMessage> messages{{ChatMessage::ROLE_USER, "这张图片有什么不寻常的地方", std::move(image)}};
        ChatMessage output = pipeline.chat(messages, gen_config);
        EXPECT_EQ(output.content,
                  "这张图片中不寻常的地方在于，男子正在一辆黄色出租车后面熨衣服。通常情况下，熨衣是在家中或洗衣店进行的"
                  "，而不是在车辆上。此外，出租车在行驶中，男子却能够稳定地熨衣，这增加了场景的荒诞感。");
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

    PerfStreamer streamer;
    start_ms = ggml_time_ms();
    pipeline.chat(messages, gen_config, &streamer);
    int64_t gen_s = (ggml_time_ms() - start_ms) / 1000.f;

    std::cout << "======== benchmark results for " << model_path.filename() << " ========\n"
              << "model loaded within: " << load_model_ms << " ms\n"
              << "generation finished within: " << gen_s << " s\n"
              << streamer.to_string() << "\n"
              << "===========================================================\n";
}

TEST(Benchmark, ChatGLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, ChatGLM2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm2-ggml.bin";
    run_benchmark(model_path);
}

TEST(Benchmark, ChatGLM4) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "models/chatglm4-ggml.bin";
    run_benchmark(model_path);
}

} // namespace chatglm
