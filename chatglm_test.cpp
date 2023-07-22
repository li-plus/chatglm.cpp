#include "chatglm.h"
#include <filesystem>
#include <gtest/gtest.h>

#ifdef GGML_USE_CUBLAS
#include <cuda_runtime.h>
#include <ggml-cuda.h>
#endif

namespace chatglm {

namespace fs = std::filesystem;

static inline int get_num_threads_for_benchmark() {
#ifdef GGML_USE_CUBLAS
    return 1;
#endif
    const char *chatglm_num_threads_env = getenv("CHATGLM_NUM_THREADS");
    int num_threads_for_benchmark =
        chatglm_num_threads_env ? std::stoi(chatglm_num_threads_env) : get_num_physical_cores();
    return num_threads_for_benchmark;
}

static inline void expect_all_close(ggml_tensor *a, ggml_tensor *b, float atol = 1e-5) {
    ASSERT_EQ(a->type, b->type);
    ASSERT_EQ(a->type, GGML_TYPE_F32);
    ASSERT_EQ(ggml_nelements(a), ggml_nelements(b));
    int64_t numel = ggml_nelements(a);
    for (int64_t i = 0; i < numel; i++) {
        EXPECT_LT(std::abs(((float *)a->data)[i] - ((float *)b->data)[i]), atol);
    }
}

static inline char *read_tensor_data(char *ptr, ggml_tensor *tensor) {
    memcpy(tensor->data, ptr, ggml_nbytes(tensor));
    return ptr + ggml_nbytes(tensor);
}

static inline float random() { return rand() / (float)RAND_MAX; }

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
    } else {
        CHATGLM_THROW << "unsupported dtype " << ggml_type_name(tensor->type);
    }
}

// return elapsed time in milliseconds
static inline float timeit(std::function<void()> fn, int warmup, int active) {
    float elapsed_ms;
    for (int i = 0; i < warmup; i++) {
        fn();
    }

#ifdef GGML_USE_CUBLAS
    cudaEvent_t start, stop;
    CHATGLM_CHECK_CUDA(cudaEventCreate(&start));
    CHATGLM_CHECK_CUDA(cudaEventCreate(&stop));
    CHATGLM_CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < active; i++) {
        fn();
    }
    CHATGLM_CHECK_CUDA(cudaEventRecord(stop));
    CHATGLM_CHECK_CUDA(cudaEventSynchronize(stop));
    CHATGLM_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHATGLM_CHECK_CUDA(cudaEventDestroy(start));
    CHATGLM_CHECK_CUDA(cudaEventDestroy(stop));
#else
    int64_t start_us = ggml_time_us();
    for (int i = 0; i < active; i++) {
        fn();
    }
    int64_t end_us = ggml_time_us();
    elapsed_ms = (end_us - start_us) / 1000.f;
#endif

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
    BaseModelForConditionalGeneration::sampling_temperature(logits.data(), logits.data() + logits.size(), temp);
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
    BaseModelForConditionalGeneration::sampling_top_k(token_scores.data(), token_scores.data() + top_k,
                                                      token_scores.data() + token_scores.size());
    token_scores.resize(top_k);

    // sort & compare
    EXPECT_TRUE(equal(extract_sorted_ids(token_scores), extract_sorted_ids(target)));
}

static void reference_top_p(std::vector<TokenIdScore> &token_scores, float top_p) {
    std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>());
    BaseModelForConditionalGeneration::sampling_softmax_inplace(token_scores.data(),
                                                                token_scores.data() + token_scores.size());
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
        TokenIdScore *pos = BaseModelForConditionalGeneration::sampling_top_p(
            token_scores.data(), token_scores.data() + token_scores.size(), top_p);
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
    int num_benchmark_threads_;

    void SetUp() override {
        ctx.dtype = GGML_TYPE_F32;
        ctx.ctx_w = make_unique_ggml_context(1024 * MB, nullptr, false);
        ctx.ctx_kv = make_unique_ggml_context(128 * MB, nullptr, false);
        ctx.ctx_b = make_unique_ggml_context(128 * MB, nullptr, false);
        ctx.scratch_buffer.resize(1 * MB);
        ctx.scratch = {0, ctx.scratch_buffer.size(), ctx.scratch_buffer.data()};

        reset_cgraph();

        num_benchmark_threads_ = get_num_threads_for_benchmark();
    }

    void reset_cgraph() { ctx.gf = {}; }

    float perf_compute() {
        auto fn = [this] { ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_); };
        if (ggml_cpu_has_cublas()) {
            return timeit(fn, 10, 100);
        } else {
            return timeit(fn, 1, 3);
        }
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
    ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

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

    tensor_to_device(x);

    // fp32
    {
        ctx.dtype = GGML_TYPE_F32;
        Linear model(&ctx, 32, 16);
        model.weight->data = w->data;
        model.bias->data = b->data;
        tensor_to_device(model.weight);
        tensor_to_device(model.bias);

        ggml_tensor *out = model.forward(&ctx, x);
        EXPECT_EQ(out->backend, x->backend);
        out->backend = GGML_BACKEND_CPU;

        ggml_build_forward_expand(&ctx.gf, out);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

        expect_all_close(ref, out);

        tensor_to_cpu(model.weight);
        tensor_to_cpu(model.bias);
    }
    // fp16
    {
        reset_cgraph();

        ctx.dtype = GGML_TYPE_F16;
        Linear model(&ctx, 32, 16);
        ggml_fp32_to_fp16_row((float *)w->data, (ggml_fp16_t *)model.weight->data, ggml_nelements(model.weight));
        model.bias->data = b->data;
        tensor_to_device(model.weight);
        tensor_to_device(model.bias);

        ggml_tensor *out = model.forward(&ctx, x);
        EXPECT_EQ(out->backend, x->backend);
        out->backend = GGML_BACKEND_CPU;

        ggml_build_forward_expand(&ctx.gf, out);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

        EXPECT_EQ(out->type, GGML_TYPE_F32);
        expect_all_close(ref, out, 5e-3);

        tensor_to_cpu(model.weight);
        tensor_to_cpu(model.bias);
    }
    tensor_to_cpu(x);
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
    std::cout << "[Benchmark] Linear " << ggml_type_name(ctx.dtype) << " time: " << perf_compute() << " ms\n";

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
    ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_);

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
    std::cout << "[Benchmark] LayerNorm " << ggml_type_name(ctx.dtype) << " time: " << perf_compute() << " ms\n";

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
    ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_);

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
    std::cout << "[Benchmark] RMSNorm " << ggml_type_name(ctx.dtype) << " time: " << perf_compute() << " ms\n";

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, GLMBlock) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/glm_block.data";
    MappedFile mapped_file(test_path.string());
    char *ptr = mapped_file.data;

    constexpr int hidden_size = 32;
    constexpr int num_attention_heads = 8;
    constexpr int num_hidden_layers = 28;
    constexpr int max_length = 16;
    constexpr int seq_len = 4;
    GLMBlock model(&ctx, hidden_size, num_attention_heads, num_hidden_layers, max_length);
    tensor_to_device(model.attention.k_cache);
    tensor_to_device(model.attention.v_cache);

    ggml_tensor *x1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor *ref_y1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor *x2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *ref_y2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *x3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *ref_y3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);

    std::vector<ggml_tensor *> all_tensors{model.input_layernorm.weight,
                                           model.input_layernorm.bias,
                                           model.attention.query_key_value.weight,
                                           model.attention.query_key_value.bias,
                                           model.attention.dense.weight,
                                           model.attention.dense.bias,
                                           model.post_attention_layernorm.weight,
                                           model.post_attention_layernorm.bias,
                                           model.mlp.dense_h_to_4h.weight,
                                           model.mlp.dense_h_to_4h.bias,
                                           model.mlp.dense_4h_to_h.weight,
                                           model.mlp.dense_4h_to_h.bias,
                                           x1,
                                           ref_y1,
                                           x2,
                                           ref_y2,
                                           x3,
                                           ref_y3};

    for (auto tensor : all_tensors) {
        ptr = read_tensor_data(ptr, tensor);
        tensor_to_device(tensor);
    }

    ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

    // self attention
    {
        ggml_tensor *out_y1 = model.forward(&ctx, x1, 0, seq_len);
        EXPECT_EQ(out_y1->backend, x1->backend);
        out_y1->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y1);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

        expect_all_close(ref_y1, out_y1, 5e-4);
    }

    // cross attention
    reset_cgraph();
    {
        ggml_tensor *out_y2 = model.forward(&ctx, x2, seq_len, seq_len);
        EXPECT_EQ(out_y2->backend, x2->backend);
        out_y2->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y2);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

        expect_all_close(ref_y2, out_y2, 5e-4);
    }
    reset_cgraph();
    {
        ggml_tensor *out_y3 = model.forward(&ctx, x3, seq_len + 1, seq_len);
        EXPECT_EQ(out_y3->backend, x3->backend);
        out_y3->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y3);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, 1);

        expect_all_close(ref_y3, out_y3, 5e-4);
    }

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
}

TEST_F(ChatGLMTest, BenchmarkGLMBlock) {
    constexpr int hidden_size = 4096;
    constexpr int num_attention_heads = 32;
    constexpr int num_hidden_layers = 28;
    constexpr int max_length = 2048;
    constexpr int seq_len = 64;

    ggml_type dtypes[]{GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
    for (const auto dtype : dtypes) {
        SetUp();

        ctx.dtype = dtype;
        GLMBlock model(&ctx, hidden_size, num_attention_heads, num_hidden_layers, max_length);

        ggml_tensor *self_attn_x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
        ggml_tensor *cross_attn_x = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);

        std::vector<ggml_tensor *> all_tensors{model.input_layernorm.weight,
                                               model.input_layernorm.bias,
                                               model.attention.query_key_value.weight,
                                               model.attention.query_key_value.bias,
                                               model.attention.dense.weight,
                                               model.attention.dense.bias,
                                               model.post_attention_layernorm.weight,
                                               model.post_attention_layernorm.bias,
                                               model.mlp.dense_h_to_4h.weight,
                                               model.mlp.dense_h_to_4h.bias,
                                               model.mlp.dense_4h_to_h.weight,
                                               model.mlp.dense_4h_to_h.bias,
                                               self_attn_x,
                                               cross_attn_x};

        for (auto tensor : all_tensors) {
            random_fill(tensor);
            tensor_to_device(tensor);
        }

        // self attention
        reset_cgraph();
        {
            ggml_tensor *self_attn_y = model.forward(&ctx, self_attn_x, 0, seq_len);
            ggml_build_forward_expand(&ctx.gf, self_attn_y);
            std::cout << "[Benchmark] GLMBlock " << ggml_type_name(dtype) << " self attn time: " << perf_compute()
                      << " ms\n";
        }

        // cross attention
        reset_cgraph();
        {
            ggml_tensor *cross_attn_y = model.forward(&ctx, cross_attn_x, seq_len, seq_len);
            ggml_build_forward_expand(&ctx.gf, cross_attn_y);
            std::cout << "[Benchmark] GLMBlock " << ggml_type_name(dtype) << " cross attn time: " << perf_compute()
                      << " ms\n";
        }

        for (auto tensor : all_tensors) {
            tensor_to_cpu(tensor);
        }
    }
}

TEST_F(ChatGLMTest, GLM2Block) {
    fs::path test_path = fs::path(__FILE__).parent_path() / "tests/data/glm2_block.data";
    MappedFile mapped_file(test_path.string());
    char *ptr = mapped_file.data;

    constexpr int seq_len = 3;
    constexpr int hidden_size = 32;
    constexpr int num_attention_heads = 8;
    constexpr int num_kv_heads = 2;
    constexpr int ffn_hidden_size = 6;
    constexpr int max_length = 8;

    GLM2Block model(&ctx, hidden_size, num_attention_heads, num_kv_heads, ffn_hidden_size, max_length);
    tensor_to_device(model.attention.k_cache);
    tensor_to_device(model.attention.v_cache);

    ggml_tensor *x1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor *ref_y1 = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
    ggml_tensor *x2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *ref_y2 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *x3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);
    ggml_tensor *ref_y3 = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);

    std::vector<ggml_tensor *> all_tensors{model.input_layernorm.weight,
                                           model.attention.query_key_value.weight,
                                           model.attention.query_key_value.bias,
                                           model.attention.dense.weight,
                                           model.post_attention_layernorm.weight,
                                           model.mlp.dense_h_to_4h.weight,
                                           model.mlp.dense_4h_to_h.weight,
                                           x1,
                                           ref_y1,
                                           x2,
                                           ref_y2,
                                           x3,
                                           ref_y3};

    for (auto tensor : all_tensors) {
        ptr = read_tensor_data(ptr, tensor);
        tensor_to_device(tensor);
    }
    ASSERT_EQ(ptr, mapped_file.data + mapped_file.size);

    // self attention
    reset_cgraph();
    {
        ggml_tensor *out_y1 = model.forward(&ctx, x1, 0);
        EXPECT_EQ(out_y1->backend, x1->backend);
        out_y1->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y1);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_);

        expect_all_close(ref_y1, out_y1, 1e-4);
    }

    // cross attention
    reset_cgraph();
    {
        ggml_tensor *out_y2 = model.forward(&ctx, x2, seq_len);
        EXPECT_EQ(out_y2->backend, x2->backend);
        out_y2->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y2);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_);

        expect_all_close(ref_y2, out_y2, 1e-4);
    }
    reset_cgraph();
    {
        ggml_tensor *out_y3 = model.forward(&ctx, x3, seq_len + 1);
        EXPECT_EQ(out_y3->backend, x3->backend);
        out_y3->backend = GGML_BACKEND_CPU;
        ggml_build_forward_expand(&ctx.gf, out_y3);
        ggml_graph_compute_with_ctx(ctx.ctx_b.get(), &ctx.gf, num_benchmark_threads_);

        expect_all_close(ref_y3, out_y3, 1e-4);
    }

    for (auto tensor : all_tensors) {
        tensor_to_cpu(tensor);
    }
    tensor_to_cpu(model.attention.k_cache);
    tensor_to_cpu(model.attention.v_cache);
}

TEST_F(ChatGLMTest, BenchmarkGLM2Block) {
    constexpr int seq_len = 64;
    constexpr int hidden_size = 4096;
    constexpr int num_attention_heads = 32;
    constexpr int num_kv_heads = 2;
    constexpr int ffn_hidden_size = 13696;
    constexpr int max_length = 2048;

    ggml_type dtypes[]{GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
    for (const auto dtype : dtypes) {
        SetUp();

        ctx.dtype = dtype;
        GLM2Block model(&ctx, hidden_size, num_attention_heads, num_kv_heads, ffn_hidden_size, max_length);
        tensor_to_device(model.attention.k_cache);
        tensor_to_device(model.attention.v_cache);

        ggml_tensor *self_attn_x = ggml_new_tensor_2d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size, seq_len);
        ggml_tensor *cross_attn_x = ggml_new_tensor_1d(ctx.ctx_b.get(), GGML_TYPE_F32, hidden_size);

        std::vector<ggml_tensor *> all_tensors{model.input_layernorm.weight,
                                               model.attention.query_key_value.weight,
                                               model.attention.query_key_value.bias,
                                               model.attention.dense.weight,
                                               model.post_attention_layernorm.weight,
                                               model.mlp.dense_h_to_4h.weight,
                                               model.mlp.dense_4h_to_h.weight,
                                               self_attn_x,
                                               cross_attn_x};

        for (auto tensor : all_tensors) {
            random_fill(tensor);
            tensor_to_device(tensor);
        }

        // self attention
        reset_cgraph();
        {
            ggml_tensor *self_attn_y = model.forward(&ctx, self_attn_x, 0);
            ggml_build_forward_expand(&ctx.gf, self_attn_y);
            std::cout << "[Benchmark] GLM2Block " << ggml_type_name(dtype) << " self attn time: " << perf_compute()
                      << " ms\n";
        }

        // cross attention
        reset_cgraph();
        {
            ggml_tensor *cross_attn_y = model.forward(&ctx, cross_attn_x, seq_len);
            ggml_build_forward_expand(&ctx.gf, cross_attn_y);
            std::cout << "[Benchmark] GLM2Block " << ggml_type_name(dtype) << " cross attn time: " << perf_compute()
                      << " ms\n";
        }

        for (auto tensor : all_tensors) {
            tensor_to_cpu(tensor);
        }
        tensor_to_cpu(model.attention.k_cache);
        tensor_to_cpu(model.attention.v_cache);
    }
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

TEST(Pipeline, ChatGLM) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLMForConditionalGeneration *>(pipeline.model.get()));

    // ===== tokenization =====

    struct TokenizerTestCase {
        std::string prompt;
        std::vector<int> input_ids;
    };
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

    for (const auto &c : cases) {
        // encode
        std::vector<int> input_ids = pipeline.tokenizer->encode(c.prompt);
        EXPECT_TRUE(equal(input_ids, c.input_ids));
        // decode
        std::string output = pipeline.tokenizer->decode(c.input_ids);
        EXPECT_EQ(output, c.prompt);
    }

    // ===== prompter =====
    {
        EXPECT_EQ(ChatGLMTokenizer::build_prompt({"你好"}), "你好");
        EXPECT_EQ(ChatGLMTokenizer::build_prompt(
                      {"你好", "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
                       "晚上睡不着应该怎么办"}),
                  "[Round 0]\n问：你好\n答：你好👋！我是人工智能助手 "
                  "ChatGLM-6B，很高兴见到你，欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：");
    }

    // ===== generation =====

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
        std::vector<std::string> history{oss.str()};
        pipeline.chat(history, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<std::string> history{"你好"};
        std::string output = pipeline.chat(history, gen_config);
        EXPECT_EQ(output, "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。");
    }
}

TEST(Pipeline, ChatGLM2) {
    fs::path model_path = fs::path(__FILE__).parent_path() / "chatglm2-ggml.bin";
    if (!fs::exists(model_path)) {
        GTEST_SKIP() << "Skipping ChatGLM2 e2e test (ggml model not found)";
    }
    Pipeline pipeline(model_path.string());
    EXPECT_TRUE(dynamic_cast<ChatGLM2ForConditionalGeneration *>(pipeline.model.get()));

    // ===== tokenization =====

    struct TokenizerTestCase {
        std::string prompt;
        std::vector<int> input_ids;
    };
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

    for (const auto &c : cases) {
        // encode
        std::vector<int> input_ids = pipeline.tokenizer->encode(c.prompt);
        EXPECT_TRUE(equal(input_ids, c.input_ids));
        // decode
        std::string output = pipeline.tokenizer->decode(c.input_ids);
        EXPECT_EQ(output, c.prompt);
    }

    // ===== prompter =====
    {
        EXPECT_EQ(ChatGLM2Tokenizer::build_prompt({"你好"}), "[Round 1]\n\n问：你好\n\n答：");
        EXPECT_EQ(ChatGLM2Tokenizer::build_prompt(
                      {"你好", "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。",
                       "晚上睡不着应该怎么办"}),
                  "[Round 1]\n\n问：你好\n\n答：你好👋！我是人工智能助手 "
                  "ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。\n\n[Round 2]\n\n问：晚上睡不着应该怎么办\n\n答：");
    }

    // ===== generation =====

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
        std::vector<std::string> history{oss.str()};
        pipeline.chat(history, gen_config);
    }

    // chat
    {
        GenerationConfig gen_config;
        gen_config.do_sample = false;
        std::vector<std::string> history{"你好"};
        std::string output = pipeline.chat(history, gen_config);
        EXPECT_EQ(output, "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。");
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
    std::vector<std::string> history{"你好", "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
                                     "晚上睡不着应该怎么办"};

    GenerationConfig gen_config;
    gen_config.do_sample = false;
    gen_config.num_threads = get_num_threads_for_benchmark();

    PerfStreamer streamer;
    start_ms = ggml_time_ms();
    pipeline.chat(history, gen_config, &streamer);
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

} // namespace chatglm
