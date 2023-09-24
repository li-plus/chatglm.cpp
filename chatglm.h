#pragma once

#include <cmath>
#include <ggml.h>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

namespace chatglm {

// ===== common =====

static constexpr size_t MB = 1024 * 1024;

class LogMessageFatal {
  public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    std::ostringstream &stream() { return oss_; }

  private:
    std::ostringstream oss_;
};

#define CHATGLM_THROW ::chatglm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATGLM_CHECK(cond)                                                                                            \
    if (!(cond))                                                                                                       \
    CHATGLM_THROW << "check failed (" #cond ") "

#define CHATGLM_CHECK_CUDA(call)                                                                                       \
    do {                                                                                                               \
        cudaError_t error = (call);                                                                                    \
        CHATGLM_CHECK(error == cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);                            \
    } while (0)

std::string to_string(ggml_tensor *tensor, bool with_data = true);

ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor);

ggml_tensor *tensor_to_device(ggml_tensor *tensor);

ggml_tensor *tensor_to_cpu(ggml_tensor *tensor);

struct BaseConfig {
    // common attributes
    ggml_type dtype;
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_hidden_layers;
    int intermediate_size;
    // for sequence generation
    int max_length;
    // for tokenizer
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int sep_token_id;
};

class BaseTokenizer {
  public:
    virtual ~BaseTokenizer() = default;
    virtual std::vector<int> encode(const std::string &text, int max_length) const = 0;
    virtual std::string decode(const std::vector<int> &ids) const = 0;
    virtual std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const = 0;
};

struct ggml_context_deleter_t {
    void operator()(ggml_context *ctx) const noexcept { ggml_free(ctx); }
};

using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

static inline unique_ggml_context_t make_unique_ggml_context(size_t mem_size, void *mem_buffer, bool no_alloc) {
    return unique_ggml_context_t(ggml_init({mem_size, mem_buffer, no_alloc}));
}

#ifdef GGML_USE_METAL
struct ggml_metal_context_deleter_t {
    void operator()(ggml_metal_context *ctx) const noexcept { ggml_metal_free(ctx); }
};

using unique_ggml_metal_context_t = std::unique_ptr<ggml_metal_context, ggml_metal_context_deleter_t>;

static inline unique_ggml_metal_context_t make_unique_ggml_metal_context(int n_cb) {
    return unique_ggml_metal_context_t(ggml_metal_init(n_cb));
}
#endif

// reference: https://stackoverflow.com/questions/11149665/c-vector-that-doesnt-initialize-its-members
struct uninitialized_char {
    char m;
    uninitialized_char() {}
};

void ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads);

struct ModelContext {
    ggml_type dtype;
    unique_ggml_context_t ctx_w;  // weight
    unique_ggml_context_t ctx_kv; // kv cache
    unique_ggml_context_t ctx_b;  // buffer
#ifdef GGML_USE_METAL
    unique_ggml_metal_context_t ctx_metal;
#endif
    ggml_cgraph gf;
    ggml_scratch scratch;
    std::vector<uninitialized_char> compute_buffer; // BLAS buffer
    std::vector<uninitialized_char> scratch_buffer; // intermediate tensor buffer
    std::string_view weight_buffer;                 // mapped weight
    std::vector<uninitialized_char> work_buffer;    // temporary buffer for graph computing

    void init_device_context();
};

class Embedding {
  public:
    Embedding() : weight(nullptr) {}
    Embedding(ModelContext *ctx, int num_embeddings, int embedding_dim)
        : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, embedding_dim, num_embeddings)) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight;
};

class Linear {
  public:
    Linear() : weight(nullptr), bias(nullptr) {}
    Linear(ModelContext *ctx, int in_features, int out_features, bool use_bias = true)
        : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, in_features, out_features)),
          bias(use_bias ? ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, out_features) : nullptr) {}

    int in_features() const { return weight->ne[0]; }
    int out_features() const { return weight->ne[1]; }

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [out_features, in_features]
    ggml_tensor *bias;   // [out_features]
};

class LayerNorm {
  public:
    LayerNorm() : weight(nullptr), bias(nullptr) {}
    LayerNorm(ModelContext *ctx, int normalized_shape)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)),
          bias(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input, float eps = 1e-5f) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]
};

class RMSNorm {
  public:
    RMSNorm() : weight(nullptr), inplace(true) {}
    RMSNorm(ModelContext *ctx, int normalized_shape, bool inplace = true)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input, float eps = 1e-5f) const;

  public:
    ggml_tensor *weight;
    bool inplace;
};

enum ActivationType {
    ACT_TYPE_GELU,
    ACT_TYPE_SILU,
};

template <ActivationType ACT_TYPE>
static inline ggml_tensor *apply_activation_inplace(ggml_context *ctx, ggml_tensor *hidden_states) {
    static_assert(ACT_TYPE == ACT_TYPE_GELU || ACT_TYPE == ACT_TYPE_SILU);
    if constexpr (ACT_TYPE == ACT_TYPE_GELU) {
        hidden_states = tensor_assign_buffers(ggml_gelu_inplace(ctx, hidden_states));
    } else if constexpr (ACT_TYPE == ACT_TYPE_SILU) {
        hidden_states = tensor_assign_buffers(ggml_silu_inplace(ctx, hidden_states));
    } else {
        CHATGLM_THROW << "Unknown activation type " << ACT_TYPE;
    }
    return hidden_states;
}

template <ActivationType ACT_TYPE>
class BasicMLP {
  public:
    BasicMLP() = default;
    BasicMLP(ModelContext *ctx, int hidden_size, int intermediate_size)
        : dense_h_to_4h(ctx, hidden_size, intermediate_size), dense_4h_to_h(ctx, intermediate_size, hidden_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const {
        ggml_context *gctx = ctx->ctx_b.get();
        hidden_states = dense_h_to_4h.forward(ctx, hidden_states);
        hidden_states = apply_activation_inplace<ACT_TYPE>(gctx, hidden_states);
        hidden_states = dense_4h_to_h.forward(ctx, hidden_states);
        return hidden_states;
    }

  public:
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;
};

template <ActivationType ACT_TYPE, bool USE_BIAS>
class BasicGLU {
  public:
    BasicGLU() = default;
    BasicGLU(ModelContext *ctx, int hidden_size, int intermediate_size)
        : gate_proj(ctx, hidden_size, intermediate_size, USE_BIAS),
          up_proj(ctx, hidden_size, intermediate_size, USE_BIAS),
          down_proj(ctx, intermediate_size, hidden_size, USE_BIAS) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const {
        ggml_context *gctx = ctx->ctx_b.get();
        ggml_tensor *gate = gate_proj.forward(ctx, hidden_states);
        gate = apply_activation_inplace<ACT_TYPE>(gctx, gate);
        hidden_states = up_proj.forward(ctx, hidden_states);
        hidden_states = tensor_assign_buffers(ggml_mul_inplace(gctx, hidden_states, gate));
        hidden_states = down_proj.forward(ctx, hidden_states);
        return hidden_states;
    }

  public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
};

struct CausalContextMasker {
    ggml_tensor *operator()(ModelContext *ctx, ggml_tensor *attn_scores, int n_past) const {
        return tensor_assign_buffers(ggml_diag_mask_inf_inplace(ctx->ctx_b.get(), attn_scores, n_past));
    }
};

enum RopeType {
    ROPE_TYPE_NONE = -1,
    ROPE_TYPE_DEFAULT = 0,
    ROPE_TYPE_CHATGLM = 4,
};

static inline ggml_tensor *apply_rope_inplace(ggml_context *ctx, ggml_tensor *a, int mode, int n_past, int n_ctx) {
    // tensor a (activation) is of shape [qlen, heads, head_size]
#ifdef GGML_USE_CUBLAS
    if (!ggml_is_contiguous(a)) {
        a = tensor_assign_buffers(ggml_cont(ctx, a));
    }
#endif
    const int head_size = a->ne[0];
    const int rope_dim = head_size / 2;
    a = tensor_assign_buffers(ggml_rope_inplace(ctx, a, n_past, rope_dim, mode, n_ctx)); // [qlen, heads, head_size]
    return a;
}

template <bool USE_QKV_BIAS, bool USE_DENSE_BIAS, bool INTERLEAVED_QKV, RopeType ROPE_TYPE, bool USE_ALIBI,
          typename ContextMasker>
class BasicAttention {
  public:
    BasicAttention() = default;
    BasicAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length)
        : num_attention_heads(num_attention_heads), num_kv_heads(num_kv_heads),
          query_key_value(ctx, hidden_size, hidden_size + 2 * (hidden_size / num_attention_heads) * num_kv_heads,
                          USE_QKV_BIAS),
          dense(ctx, hidden_size, hidden_size, USE_DENSE_BIAS),
          k_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                                     num_kv_heads)),
          v_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, max_length, hidden_size / num_attention_heads,
                                     num_kv_heads)) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const {
        ggml_context *gctx = ctx->ctx_b.get();

        const int hidden_size = hidden_states->ne[0];
        const int qlen = hidden_states->ne[1];
        const int head_size = hidden_size / num_attention_heads;
        const int num_shared_q_heads = num_attention_heads / num_kv_heads;
        const bool is_gqa = num_shared_q_heads > 1;

        ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, hidden + 2 * kv_hidden]

        // split mixed qkv into separate query, key and value
        ggml_tensor *query_layer; // [qlen, heads, head_size]
        ggml_tensor *key_layer;   // [qlen, kv_heads, head_size]
        ggml_tensor *value_layer; // [qlen, kv_heads, head_size]

        if constexpr (INTERLEAVED_QKV) {
            CHATGLM_CHECK(!is_gqa) << "interleaved qkv is not supported for GQA";
            query_layer = ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen,
                                       3 * head_size * ggml_element_size(qkv), qkv->nb[1], 0);
            key_layer =
                ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                             qkv->nb[1], head_size * ggml_element_size(qkv));
            value_layer =
                ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                             qkv->nb[1], 2 * head_size * ggml_element_size(qkv));
        } else {
            query_layer = ggml_view_3d(gctx, qkv, head_size, num_attention_heads, qlen,
                                       head_size * ggml_element_size(qkv), qkv->nb[1], 0);
            key_layer = ggml_view_3d(gctx, qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv),
                                     qkv->nb[1], hidden_size * ggml_element_size(qkv));
            value_layer = ggml_view_3d(gctx, qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv),
                                       qkv->nb[1], (hidden_size + head_size * num_kv_heads) * ggml_element_size(qkv));
        }

        if constexpr (ROPE_TYPE != ROPE_TYPE_NONE) {
            query_layer = apply_rope_inplace(gctx, query_layer, ROPE_TYPE, n_past, n_ctx);
            key_layer = apply_rope_inplace(gctx, key_layer, ROPE_TYPE, n_past, n_ctx);
        }

        query_layer = tensor_assign_buffers(
            ggml_cont(gctx, ggml_permute(gctx, query_layer, 0, 2, 1, 3))); // [heads, qlen, head_size]
        if (num_shared_q_heads > 1) {
            query_layer =
                tensor_assign_buffers(ggml_reshape_3d(gctx, query_layer, head_size, num_shared_q_heads * qlen,
                                                      num_kv_heads)); // [kv_heads, shared_qheads * qlen, head_size]
        }

        key_layer = tensor_assign_buffers(ggml_permute(gctx, key_layer, 0, 2, 1, 3)); // [kv_heads, qlen, head_size]

        value_layer = tensor_assign_buffers(ggml_permute(gctx, value_layer, 1, 2, 0, 3)); // [kv_heads, head_size, qlen]

        // store key & value to cache
        ggml_tensor *k_cache_view = tensor_assign_buffers(
            ggml_view_3d(gctx, k_cache, head_size, qlen, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                         n_past * head_size * ggml_element_size(k_cache))); // [kv_heads, qlen, head_size]
        ggml_build_forward_expand(&ctx->gf, ggml_cpy(gctx, key_layer, k_cache_view));
        ggml_tensor *v_cache_view = tensor_assign_buffers(
            ggml_view_3d(gctx, v_cache, qlen, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                         n_past * ggml_element_size(v_cache))); // [kv_heads, head_size, qlen]
        ggml_build_forward_expand(&ctx->gf, ggml_cpy(gctx, value_layer, v_cache_view));

        // concat key & value with past kv
        key_layer = tensor_assign_buffers(ggml_view_3d(gctx, k_cache, head_size, n_past + qlen, num_kv_heads,
                                                       k_cache->nb[1], k_cache->nb[2],
                                                       0)); // [kv_heads, klen, head_size]
        value_layer = tensor_assign_buffers(ggml_view_3d(gctx, v_cache, n_past + qlen, head_size, num_kv_heads,
                                                         v_cache->nb[1], v_cache->nb[2],
                                                         0)); // [kv_heads, head_size, klen]

        // attention
        ggml_tensor *attn_scores =
            tensor_assign_buffers(ggml_mul_mat(gctx, key_layer, query_layer)); // [kv_heads, shared_qheads * qlen, klen]
        attn_scores = tensor_assign_buffers(
            ggml_scale_inplace(gctx, attn_scores, ggml_new_f32(gctx, 1.f / std::sqrt(head_size))));
        if constexpr (USE_ALIBI) {
            attn_scores = tensor_assign_buffers(ggml_alibi(gctx, attn_scores, n_past, num_attention_heads, 8));
        }
        if (n_past == 0) {
            // build attention mask for context input
            if (num_shared_q_heads > 1) {
                attn_scores = ggml_reshape_3d(gctx, attn_scores, n_past + qlen, qlen,
                                              num_attention_heads); // [heads, qlen, klen]
            }
            attn_scores = context_masker_(ctx, attn_scores, n_past);
            if (num_shared_q_heads > 1) {
                attn_scores = ggml_reshape_3d(gctx, attn_scores, n_past + qlen, num_shared_q_heads * qlen,
                                              num_kv_heads); // [kv_heads, shared_qheads * qlen, klen]
            }
        }
        ggml_tensor *attn_probs =
            tensor_assign_buffers(ggml_soft_max_inplace(gctx, attn_scores)); // [kv_heads, shared_qheads * qlen, klen]

        ggml_tensor *context_layer = tensor_assign_buffers(
            ggml_mul_mat(gctx, value_layer, attn_probs)); // [kv_heads, shared_qheads * qlen, head_size]
        if (num_shared_q_heads > 1) {
            context_layer = ggml_reshape_3d(gctx, context_layer, head_size, qlen,
                                            num_attention_heads); // [heads, qlen, head_size]
        }
        context_layer = tensor_assign_buffers(
            ggml_cont(gctx, ggml_permute(gctx, context_layer, 0, 2, 1, 3))); // [qlen, heads, head_size]
        context_layer =
            tensor_assign_buffers(ggml_reshape_2d(gctx, context_layer, hidden_size, qlen)); // [qlen, hidden]

        ggml_tensor *attn_output = dense.forward(ctx, context_layer);
        return attn_output;
    }

  public:
    int num_attention_heads;
    int num_kv_heads;
    Linear query_key_value;
    Linear dense;
    ggml_tensor *k_cache; // [kv_heads, max_len, head_size]
    ggml_tensor *v_cache; // [kv_heads, head_size, max_len]

  private:
    ContextMasker context_masker_;
};

template <typename Block, typename Norm>
class BasicModel {
  protected:
    BasicModel() = default;
    BasicModel(Embedding word_embeddings, std::vector<Block> layers, Norm final_layernorm)
        : word_embeddings(word_embeddings), layers(std::move(layers)), final_layernorm(final_layernorm) {}

  public:
    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const {
        ggml_context *gctx = ctx->ctx_b.get();
        ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
        for (const auto &layer : layers) {
            ggml_set_scratch(gctx, ctx->scratch);
            hidden_states = layer.forward(ctx, hidden_states, n_past, n_ctx);
        }
        ggml_scratch empty_scratch = {0, 0, nullptr};
        ggml_set_scratch(gctx, empty_scratch);
        hidden_states = final_layernorm.forward(ctx, hidden_states);
        return hidden_states;
    }

  public:
    Embedding word_embeddings;
    std::vector<Block> layers;
    Norm final_layernorm;
};

class BaseStreamer {
  public:
    virtual ~BaseStreamer() = default;
    virtual void put(const std::vector<int> &output_ids) = 0;
    virtual void end() = 0;
};

class StreamerGroup : public BaseStreamer {
  public:
    StreamerGroup(std::vector<std::shared_ptr<BaseStreamer>> streamers) : streamers_(std::move(streamers)) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    std::vector<std::shared_ptr<BaseStreamer>> streamers_;
};

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
class TextStreamer : public BaseStreamer {
  public:
    TextStreamer(std::ostream &os, BaseTokenizer *tokenizer)
        : os_(os), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    std::ostream &os_;
    BaseTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

class PerfStreamer : public BaseStreamer {
  public:
    PerfStreamer() : start_us_(0), prompt_us_(0), end_us_(0), num_prompt_tokens_(0), num_output_tokens_(0) {}

    void put(const std::vector<int> &output_ids) override;
    void end() override { end_us_ = ggml_time_us(); }

    void reset();
    std::string to_string() const;

    int64_t num_prompt_tokens() const { return num_prompt_tokens_; }
    int64_t prompt_total_time_us() const { return prompt_us_ - start_us_; }
    int64_t prompt_token_time_us() const {
        return num_prompt_tokens() ? prompt_total_time_us() / num_prompt_tokens() : 0;
    }
    int64_t num_output_tokens() const { return num_output_tokens_; }
    int64_t output_total_time_us() const { return end_us_ - prompt_us_; }
    int64_t output_token_time_us() const {
        return num_output_tokens() ? output_total_time_us() / num_output_tokens() : 0;
    }

  private:
    int64_t start_us_;
    int64_t prompt_us_;
    int64_t end_us_;
    int64_t num_prompt_tokens_;
    int64_t num_output_tokens_;
};

class MappedFile {
  public:
    MappedFile(const std::string &path);
    ~MappedFile();

  public:
    char *data;
    size_t size;
};

class ModelLoader {
  public:
    ModelLoader(char *data, size_t size) : data(data), size(size), ptr(data) {}

    int64_t tell() const { return ptr - data; }

    void seek(int64_t offset, int whence);

    template <typename T>
    T read_basic() {
        T obj = *(T *)ptr;
        ptr += sizeof(T);
        return obj;
    }

    std::string read_string(size_t length);

    void checked_read_tensor_meta(const std::string &name, int ndim, int64_t *ne, ggml_type dtype);

    void *read_tensor_data(size_t nbytes);

    void read_tensor(const std::string &name, ggml_tensor *tensor);

  public:
    char *data;
    size_t size;
    char *ptr;
};

// ===== generation =====

struct GenerationConfig {
    int max_length;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    int num_threads;

    GenerationConfig(int max_length = 2048, int max_context_length = 512, bool do_sample = true, int top_k = 0,
                     float top_p = 0.7, float temperature = 0.95, float repetition_penalty = 1.f, int num_threads = 0)
        : max_length(max_length), max_context_length(max_context_length), do_sample(do_sample), top_k(top_k),
          top_p(top_p), temperature(temperature), repetition_penalty(repetition_penalty), num_threads(num_threads) {}
};

enum ModelType {
    MODEL_TYPE_CHATGLM = 1,
    MODEL_TYPE_CHATGLM2 = 2,
    MODEL_TYPE_BAICHUAN7B = 1024,
    MODEL_TYPE_BAICHUAN13B = 1025,
};

int get_num_physical_cores();
int get_default_num_threads();

std::string to_string(ModelType model_type);

struct TokenIdScore {
    int id;
    float score;

    TokenIdScore() = default;
    TokenIdScore(int id, float score) : id(id), score(score) {}

    bool operator<(const TokenIdScore &other) const { return score < other.score; }
    bool operator>(const TokenIdScore &other) const { return score > other.score; }

    friend std::ostream &operator<<(std::ostream &os, const TokenIdScore &self) {
        return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
    }
};

class BaseModelForCausalLM {
  public:
    BaseModelForCausalLM(ModelType model_type, BaseConfig config, size_t mem_size, size_t scratch_size);
    virtual ~BaseModelForCausalLM() = default;

    virtual void load(ModelLoader &loader) = 0;
    virtual ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const = 0;

    ModelType type() const { return model_type_; }
    std::string type_name() const { return to_string(model_type_); }

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr);

    int generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, int n_past,
                            int n_ctx);

    // logits processor
    static void sampling_repetition_penalty(float *first, float *last, const std::vector<int> &input_ids,
                                            float penalty);
    // logits warper
    static void sampling_temperature(float *first, float *last, float temp);
    static void sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last);
    static TokenIdScore *sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p);

    static void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last);

  protected:
    ModelType model_type_;
    BaseConfig config_;
    ModelContext ctx_;
    std::vector<std::pair<std::string, ggml_tensor *>> state_dict_;
};

template <typename Config, typename Model>
class BasicModelForCausalLM : public BaseModelForCausalLM {
  protected:
    BasicModelForCausalLM(ModelType model_type, const Config &config, size_t mem_size, size_t scratch_size)
        : BaseModelForCausalLM(model_type, config, mem_size, scratch_size), config(config) {}

  public:
    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override {
        ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, n_past, n_ctx);
        // NOTE: only compute next_token_logits for the last token
        if (input_ids->ne[0] > 1) {
            transformer_outputs = tensor_assign_buffers(
                ggml_view_1d(ctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                             (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs)));
        }
        ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
        return lm_logits;
    }

  public:
    Config config;
    Model transformer;
    Linear lm_head;
};

// ===== ChatGLM-6B =====

struct ChatGLMConfig : public BaseConfig {};

class ChatGLMTokenizer : public BaseTokenizer {
  public:
    ChatGLMTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

  private:
    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);

  public:
    sentencepiece::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;
    int gmask_token_id;
    int pad_token_id;
};

struct GLMContextMasker {
    ggml_tensor *operator()(ModelContext *ctx, ggml_tensor *attn_scores, int n_past) const;
};

using GLMAttention = BasicAttention<true, true, true, ROPE_TYPE_CHATGLM, false, GLMContextMasker>;

using GLMMLP = BasicMLP<ACT_TYPE_GELU>;

class GLMBlock {
  public:
    GLMBlock() : num_hidden_layers(0) {}
    GLMBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_hidden_layers, int max_length)
        : input_layernorm(ctx, hidden_size),
          attention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length),
          post_attention_layernorm(ctx, hidden_size), mlp(ctx, hidden_size, 4 * hidden_size),
          num_hidden_layers(num_hidden_layers) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;

  public:
    LayerNorm input_layernorm;
    GLMAttention attention;
    LayerNorm post_attention_layernorm;
    GLMMLP mlp;
    int num_hidden_layers;
};

class ChatGLMModel : public BasicModel<GLMBlock, LayerNorm> {
  public:
    ChatGLMModel() = default;
    ChatGLMModel(ModelContext *ctx, const ChatGLMConfig &config)
        : BasicModel(Embedding(ctx, config.vocab_size, config.hidden_size), build_layers(ctx, config),
                     LayerNorm(ctx, config.hidden_size)) {}

  private:
    static std::vector<GLMBlock> build_layers(ModelContext *ctx, const ChatGLMConfig &config);
};

class ChatGLMForCausalLM : public BasicModelForCausalLM<ChatGLMConfig, ChatGLMModel> {
  public:
    ChatGLMForCausalLM(const ChatGLMConfig &config);
    ~ChatGLMForCausalLM();

    void load(ModelLoader &loader) override;

  public:
    static constexpr size_t MEM_SIZE = 512 * MB;      // 2k context
    static constexpr size_t SCRATCH_SIZE = 1024 * MB; // 2k context
};

// ===== ChatGLM2-6B =====

struct ChatGLM2Config : public BaseConfig {
    int num_kv_heads;
};

class ChatGLM2Tokenizer : public BaseTokenizer {
  public:
    ChatGLM2Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const override;

    static std::string build_prompt(const std::vector<std::string> &history);

    bool is_special_id(int id) const;

  public:
    sentencepiece::SentencePieceProcessor sp;
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
};

using GLM2Attention = BasicAttention<true, false, false, ROPE_TYPE_DEFAULT, false, CausalContextMasker>;

using GLM2MLP = BasicGLU<ACT_TYPE_SILU, false>;

class GLM2Block {
  public:
    GLM2Block() = default;
    GLM2Block(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
              int max_length)
        : input_layernorm(ctx, hidden_size, false),
          attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
          post_attention_layernorm(ctx, hidden_size, false), mlp(ctx, hidden_size, intermediate_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;

  public:
    RMSNorm input_layernorm;
    GLM2Attention attention;
    RMSNorm post_attention_layernorm;
    GLM2MLP mlp;
};

class ChatGLM2Model : public BasicModel<GLM2Block, RMSNorm> {
  public:
    ChatGLM2Model() = default;
    ChatGLM2Model(ModelContext *ctx, const ChatGLM2Config &config)
        : BasicModel(Embedding(ctx, config.vocab_size, config.hidden_size), build_layers(ctx, config),
                     RMSNorm(ctx, config.hidden_size)) {}

  private:
    static std::vector<GLM2Block> build_layers(ModelContext *ctx, const ChatGLM2Config &config);
};

class ChatGLM2ForCausalLM : public BasicModelForCausalLM<ChatGLM2Config, ChatGLM2Model> {
  public:
    ChatGLM2ForCausalLM(const ChatGLM2Config &config);
    ~ChatGLM2ForCausalLM();

    void load(ModelLoader &loader) override;

  public:
    static constexpr size_t MEM_SIZE = 512 * MB;      // 2k context
    static constexpr size_t SCRATCH_SIZE = 1280 * MB; // 2k context
};

// ===== Baichuan-13B =====

struct Baichuan13BConfig : public BaseConfig {};

class Baichuan13BTokenizer : public BaseTokenizer {
  public:
    Baichuan13BTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const override;

    bool is_special_id(int id) const;

  protected:
    static void truncate(std::vector<int> &ids, int max_length);

  public:
    static constexpr int USER_TOKEN_ID = 195;
    static constexpr int ASSISTANT_TOKEN_ID = 196;

    sentencepiece::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
};

using Baichuan13BAttention = BasicAttention<false, false, false, ROPE_TYPE_NONE, true, CausalContextMasker>;

using Baichuan13BMLP = BasicGLU<ACT_TYPE_SILU, false>;

class Baichuan13BBlock {
  public:
    Baichuan13BBlock() = default;
    Baichuan13BBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size, int max_length)
        : input_layernorm(ctx, hidden_size, false),
          attention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length),
          post_attention_layernorm(ctx, hidden_size, false), mlp(ctx, hidden_size, intermediate_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;

  public:
    RMSNorm input_layernorm;
    Baichuan13BAttention attention;
    RMSNorm post_attention_layernorm;
    Baichuan13BMLP mlp;
};

class Baichuan13BModel : public BasicModel<Baichuan13BBlock, RMSNorm> {
  public:
    Baichuan13BModel() = default;
    Baichuan13BModel(ModelContext *ctx, const Baichuan13BConfig &config)
        : BasicModel(Embedding(ctx, config.vocab_size, config.hidden_size), build_layers(ctx, config),
                     RMSNorm(ctx, config.hidden_size)) {}

  private:
    static std::vector<Baichuan13BBlock> build_layers(ModelContext *ctx, const Baichuan13BConfig &config);
};

class Baichuan13BForCausalLM : public BasicModelForCausalLM<Baichuan13BConfig, Baichuan13BModel> {
  public:
    Baichuan13BForCausalLM(const Baichuan13BConfig &config);
    ~Baichuan13BForCausalLM();

    void load(ModelLoader &loader) override;

  public:
    static constexpr size_t MEM_SIZE = 512 * MB;
    static constexpr size_t SCRATCH_SIZE = 1280 * MB;
};

// ===== pipeline =====

class Pipeline {
  public:
    Pipeline(const std::string &path);

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr) const;

    std::string generate(const std::string &prompt, const GenerationConfig &gen_config,
                         BaseStreamer *streamer = nullptr) const;

    std::string chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                     BaseStreamer *streamer = nullptr) const;

  public:
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<BaseModelForCausalLM> model;
    std::unique_ptr<MappedFile> mapped_file;
};

} // namespace chatglm
