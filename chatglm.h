#pragma once

#include <cmath>
#include <ggml.h>
#include <ggml/ggml-backend.h>
#include <iomanip>
#include <re2/re2.h>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>

namespace chatglm {

// ===== common =====

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

enum class ModelType {
    CHATGLM = 1,
    CHATGLM2 = 2,
    CHATGLM3 = 3,
    CHATGLM4 = 4,
};

std::string to_string(ModelType model_type);

// For compatibility
struct ConfigRecordV1 {
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

// For compatibility
struct ConfigRecordV1GQA : public ConfigRecordV1 {
    int num_key_value_heads;
};

// TODO: use json to serialize config
struct ConfigRecordV2 {
    ggml_type dtype;
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int intermediate_size;
    float norm_eps;
    int num_virtual_tokens;
    float rope_theta;
    int max_length;
    int eos_token_id;
    int pad_token_id;
};

enum class ActivationType {
    GELU,
    SILU,
};

enum class RopeType {
    GPTJ = 0,
    NEOX = 2,
    CHATGLM = 4,
    CHATGLM2 = 8,
    DISABLED = 10000,
};

enum class AttentionMaskType {
    CAUSAL,
    CHATGLM,
};

// Should save kv record of ModelConfig in the future
class ModelConfig {
  public:
    ModelConfig() = default;

    ModelConfig(ModelType model_type, ggml_type dtype, int vocab_size, int hidden_size, int num_attention_heads,
                int num_key_value_heads, int num_hidden_layers, int intermediate_size, float norm_eps, float rope_theta,
                int num_virtual_tokens, int max_length, int bos_token_id, int eos_token_id, int pad_token_id,
                int sep_token_id, std::vector<int> extra_eos_token_ids)
        : model_type(model_type), dtype(dtype), vocab_size(vocab_size), hidden_size(hidden_size),
          num_attention_heads(num_attention_heads), num_key_value_heads(num_key_value_heads),
          num_hidden_layers(num_hidden_layers), intermediate_size(intermediate_size), norm_eps(norm_eps),
          rope_theta(rope_theta), num_virtual_tokens(num_virtual_tokens), max_length(max_length),
          bos_token_id(bos_token_id), eos_token_id(eos_token_id), pad_token_id(pad_token_id),
          sep_token_id(sep_token_id), extra_eos_token_ids(std::move(extra_eos_token_ids)) {
        if (model_type == ModelType::CHATGLM) {
            hidden_act = ActivationType::GELU;
            use_qkv_bias = true;
            use_dense_bias = true;
            interleaved_qkv = true;
            tie_word_embeddings = true;
            rope_type = RopeType::CHATGLM;
            attn_mask_type = AttentionMaskType::CHATGLM;
        } else {
            hidden_act = ActivationType::SILU;
            use_qkv_bias = true;
            use_dense_bias = false;
            interleaved_qkv = false;
            tie_word_embeddings = false;
            rope_type = RopeType::CHATGLM2;
            attn_mask_type = AttentionMaskType::CAUSAL;
        }
    }

    ModelConfig(ModelType model_type, const ConfigRecordV1 &rec, float norm_eps, float rope_theta,
                int num_virtual_tokens)
        : ModelConfig(model_type, rec.dtype, rec.vocab_size, rec.hidden_size, rec.num_attention_heads,
                      rec.num_attention_heads, rec.num_hidden_layers, rec.intermediate_size, norm_eps, rope_theta,
                      num_virtual_tokens, rec.max_length, rec.bos_token_id, rec.eos_token_id, rec.pad_token_id,
                      rec.sep_token_id, {}) {}

    ModelConfig(ModelType model_type, const ConfigRecordV1GQA &rec, float norm_eps, float rope_theta,
                int num_virtual_tokens)
        : ModelConfig(model_type, rec.dtype, rec.vocab_size, rec.hidden_size, rec.num_attention_heads,
                      rec.num_key_value_heads, rec.num_hidden_layers, rec.intermediate_size, norm_eps, rope_theta,
                      num_virtual_tokens, rec.max_length, rec.bos_token_id, rec.eos_token_id, rec.pad_token_id,
                      rec.sep_token_id, {}) {}

    ModelConfig(ModelType model_type, const ConfigRecordV2 &rec)
        : ModelConfig(model_type, rec.dtype, rec.vocab_size, rec.hidden_size, rec.num_attention_heads,
                      rec.num_key_value_heads, rec.num_hidden_layers, rec.intermediate_size, rec.norm_eps,
                      rec.rope_theta, rec.num_virtual_tokens, rec.max_length, -1, rec.eos_token_id, rec.pad_token_id,
                      -1, {}) {}

    std::string model_type_name() const { return to_string(model_type); }

    friend std::ostream &operator<<(std::ostream &os, const ModelConfig &self) {
        os << "ModelConfig(model_type=" << (int)self.model_type << ", dtype=" << self.dtype
           << ", vocab_size=" << self.vocab_size << ", hidden_size=" << self.hidden_size
           << ", num_attention_heads=" << self.num_attention_heads
           << ", num_key_value_heads=" << self.num_key_value_heads << ", num_hidden_layers=" << self.num_hidden_layers
           << ", intermediate_size=" << self.intermediate_size << ", norm_eps=" << self.norm_eps
           << ", hidden_act=" << (int)self.hidden_act << ", use_qkv_bias=" << self.use_qkv_bias
           << ", use_dense_bias=" << self.use_dense_bias << ", interleaved_qkv=" << self.interleaved_qkv
           << ", tie_word_embeddings=" << self.tie_word_embeddings << ", rope_type=" << (int)self.rope_type
           << ", rope_theta=" << self.rope_theta << ", attn_mask_type=" << (int)self.attn_mask_type
           << ", num_virtual_tokens=" << self.num_virtual_tokens << ", max_length=" << self.max_length
           << ", bos_token_id=" << self.bos_token_id << ", eos_token_id=" << self.eos_token_id
           << ", pad_token_id=" << self.pad_token_id << ", sep_token_id=" << self.sep_token_id
           << ", extra_eos_token_ids={";
        for (size_t i = 0; i < self.extra_eos_token_ids.size(); i++) {
            os << (i > 0 ? ", " : "") << self.extra_eos_token_ids[i];
        }
        return os << "})";
    }

  public:
    ModelType model_type;
    ggml_type dtype;
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int intermediate_size;
    float norm_eps;
    ActivationType hidden_act;
    bool use_qkv_bias;
    bool use_dense_bias;
    bool interleaved_qkv;
    bool tie_word_embeddings;
    RopeType rope_type;
    float rope_theta;
    AttentionMaskType attn_mask_type;
    int num_virtual_tokens;
    int max_length;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    int sep_token_id;
    std::vector<int> extra_eos_token_ids;
};

struct FunctionMessage {
    std::string name;
    std::string arguments;

    FunctionMessage() = default;
    FunctionMessage(std::string name, std::string arguments) : name(std::move(name)), arguments(std::move(arguments)) {}

    friend std::ostream &operator<<(std::ostream &os, const FunctionMessage &self) {
        return os << "FunctionMessage(name=" << std::quoted(self.name) << ", arguments=" << std::quoted(self.arguments)
                  << ")";
    }
};

struct CodeMessage {
    std::string input;

    CodeMessage() = default;
    CodeMessage(std::string input) : input(std::move(input)) {}

    friend std::ostream &operator<<(std::ostream &os, const CodeMessage &self) {
        return os << "CodeMessage(input=" << std::quoted(self.input) << ")";
    }
};

struct ToolCallMessage {
    std::string type;
    FunctionMessage function;
    CodeMessage code;

    static const std::string TYPE_FUNCTION;
    static const std::string TYPE_CODE;

    ToolCallMessage(FunctionMessage function) : type(TYPE_FUNCTION), function(std::move(function)) {}

    ToolCallMessage(CodeMessage code) : type(TYPE_CODE), code(std::move(code)) {}

    friend std::ostream &operator<<(std::ostream &os, const ToolCallMessage &self) {
        return os << "ToolCallMessage(type=" << std::quoted(self.type) << ", function=" << self.function
                  << ", code=" << self.code << ")";
    }
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ToolCallMessage> tool_calls;

    static const std::string ROLE_USER;
    static const std::string ROLE_ASSISTANT;
    static const std::string ROLE_SYSTEM;
    static const std::string ROLE_OBSERVATION;

    ChatMessage() = default;
    ChatMessage(std::string role, std::string content, std::vector<ToolCallMessage> tool_calls = {})
        : role(std::move(role)), content(std::move(content)), tool_calls(std::move(tool_calls)) {}

    friend std::ostream &operator<<(std::ostream &os, const ChatMessage &self) {
        os << "ChatMessage(role=" << std::quoted(self.role) << ", content=" << std::quoted(self.content)
           << ", tool_calls=[";
        for (size_t i = 0; i < self.tool_calls.size(); i++) {
            os << (i > 0 ? ", " : "") << self.tool_calls[i];
        }
        return os << "])";
    }
};

class BaseTokenizer {
  public:
    virtual ~BaseTokenizer() = default;

    virtual std::vector<int> encode(const std::string &text, int max_length) const = 0;

    virtual std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const = 0;

    virtual std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const = 0;

    virtual ChatMessage decode_message(const std::vector<int> &ids) const {
        return {ChatMessage::ROLE_ASSISTANT, decode(ids)};
    }

  protected:
    static void check_chat_messages(const std::vector<ChatMessage> &messages);

    static std::vector<ChatMessage> filter_user_assistant_messages(const std::vector<ChatMessage> &messages);
};

struct ggml_context_deleter_t {
    void operator()(ggml_context *ctx) const noexcept { ggml_free(ctx); }
};

using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

inline unique_ggml_context_t make_unique_ggml_context(size_t mem_size, void *mem_buffer, bool no_alloc) {
    return unique_ggml_context_t(ggml_init({mem_size, mem_buffer, no_alloc}));
}

struct ggml_gallocr_deleter_t {
    void operator()(ggml_gallocr *galloc) const noexcept { ggml_gallocr_free(galloc); }
};

using unique_ggml_gallocr_t = std::unique_ptr<ggml_gallocr, ggml_gallocr_deleter_t>;

struct ggml_backend_deleter_t {
    void operator()(ggml_backend_t backend) const noexcept { ggml_backend_free(backend); }
};

using unique_ggml_backend_t = std::unique_ptr<ggml_backend, ggml_backend_deleter_t>;

struct ggml_backend_buffer_deleter_t {
    void operator()(ggml_backend_buffer_t buffer) const noexcept { ggml_backend_buffer_free(buffer); }
};

using unique_ggml_backend_buffer_t = std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter_t>;

// reference: https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
template <typename T>
struct no_init {
    T value;
    no_init() { /* do nothing */
    }
};

struct ModelContext {
    ggml_type dtype;

    std::vector<no_init<char>> compute_meta;

    unique_ggml_context_t ctx_w;  // weight
    unique_ggml_context_t ctx_kv; // kv cache
    unique_ggml_context_t ctx_b;  // buffer

    ggml_cgraph *gf;
    unique_ggml_backend_t backend;
    unique_ggml_gallocr_t allocr;

    unique_ggml_backend_buffer_t buf_w;
    unique_ggml_backend_buffer_t buf_kv;

    ModelContext(ggml_type dtype);
};

class Embedding {
  public:
    Embedding() : weight(nullptr) {}
    Embedding(ModelContext *mctx, int num_embeddings, int embedding_dim)
        : weight(ggml_new_tensor_2d(mctx->ctx_w.get(), mctx->dtype, embedding_dim, num_embeddings)) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight;
};

class Linear {
  public:
    Linear() : weight(nullptr), bias(nullptr) {}
    Linear(ModelContext *mctx, int in_features, int out_features, bool use_bias = true)
        : weight(ggml_new_tensor_2d(mctx->ctx_w.get(), mctx->dtype, in_features, out_features)),
          bias(use_bias ? ggml_new_tensor_1d(mctx->ctx_w.get(), GGML_TYPE_F32, out_features) : nullptr) {}

    int in_features() const { return weight->ne[0]; }
    int out_features() const { return weight->ne[1]; }

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [out_features, in_features]
    ggml_tensor *bias;   // [out_features]
};

class LayerNorm {
  public:
    LayerNorm() = default;
    LayerNorm(ModelContext *mctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(mctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)),
          bias(ggml_new_tensor_1d(mctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]
    bool inplace;
    float eps;
};

class RMSNorm {
  public:
    RMSNorm() = default;
    RMSNorm(ModelContext *mctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(mctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    bool inplace;
    float eps;
};

class BasicMLP {
  public:
    BasicMLP() = default;
    BasicMLP(ModelContext *mctx, int hidden_size, int intermediate_size, ActivationType hidden_act)
        : dense_h_to_4h(mctx, hidden_size, intermediate_size), dense_4h_to_h(mctx, intermediate_size, hidden_size),
          hidden_act(hidden_act) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *hidden_states) const;

  public:
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;
    ActivationType hidden_act;
};

class BasicGLU {
  public:
    BasicGLU() = default;
    BasicGLU(ModelContext *mctx, int hidden_size, int intermediate_size, ActivationType hidden_act)
        : gate_proj(mctx, hidden_size, intermediate_size, false), up_proj(mctx, hidden_size, intermediate_size, false),
          down_proj(mctx, intermediate_size, hidden_size, false), hidden_act(hidden_act) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *hidden_states) const;

  public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    ActivationType hidden_act;
};

class BasicAttention {
  public:
    BasicAttention() = default;
    BasicAttention(ModelContext *mctx, int hidden_size, int num_attention_heads, int num_key_value_heads,
                   int max_length, bool use_qkv_bias, bool use_dense_bias, bool interleaved_qkv, RopeType rope_type,
                   float rope_theta, AttentionMaskType attn_mask_type, int num_virtual_tokens)
        : num_attention_heads(num_attention_heads), num_key_value_heads(num_key_value_heads),
          interleaved_qkv(interleaved_qkv), rope_type(rope_type), rope_theta(rope_theta),
          attn_mask_type(attn_mask_type), num_virtual_tokens(num_virtual_tokens),
          query_key_value(mctx, hidden_size,
                          hidden_size + 2 * (hidden_size / num_attention_heads) * num_key_value_heads, use_qkv_bias),
          dense(mctx, hidden_size, hidden_size, use_dense_bias),
          k_cache(ggml_new_tensor_3d(mctx->ctx_kv.get(), GGML_TYPE_F16, hidden_size / num_attention_heads,
                                     max_length + num_virtual_tokens, num_key_value_heads)),
          v_cache(ggml_new_tensor_3d(mctx->ctx_kv.get(), GGML_TYPE_F16, max_length + num_virtual_tokens,
                                     hidden_size / num_attention_heads, num_key_value_heads)) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                         ggml_tensor *position_ids, int n_past) const;

  public:
    int num_attention_heads;
    int num_key_value_heads;
    bool interleaved_qkv;
    RopeType rope_type;
    float rope_theta;
    AttentionMaskType attn_mask_type;
    int num_virtual_tokens;
    Linear query_key_value;
    Linear dense;
    ggml_tensor *k_cache; // [#kvh, s, d]
    ggml_tensor *v_cache; // [#kvh, d, s]
};

template <typename Norm, typename MLP>
class BasicBlock {
  public:
    BasicBlock() = default;
    BasicBlock(ModelContext *mctx, int hidden_size, int num_attention_heads, int num_key_value_heads,
               int intermediate_size, int max_length, float norm_eps, ActivationType hidden_act, bool use_qkv_bias,
               bool use_dense_bias, bool interleaved_qkv, RopeType rope_type, float rope_theta,
               AttentionMaskType attn_mask_type, int num_virtual_tokens)
        : input_layernorm(mctx, hidden_size, false, norm_eps),
          attention(mctx, hidden_size, num_attention_heads, num_key_value_heads, max_length, use_qkv_bias,
                    use_dense_bias, interleaved_qkv, rope_type, rope_theta, attn_mask_type, num_virtual_tokens),
          post_attention_layernorm(mctx, hidden_size, false, norm_eps),
          mlp(mctx, hidden_size, intermediate_size, hidden_act) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                         ggml_tensor *position_ids, int n_past) const {
        ggml_context *ctx = mctx->ctx_b.get();

        ggml_tensor *residual = hidden_states;
        hidden_states = input_layernorm.forward(mctx, hidden_states);
        hidden_states = attention.forward(mctx, hidden_states, attention_mask, position_ids, n_past);
        hidden_states = ggml_add_inplace(ctx, hidden_states, residual);

        residual = hidden_states;
        hidden_states = post_attention_layernorm.forward(mctx, hidden_states);
        hidden_states = mlp.forward(mctx, hidden_states);
        hidden_states = ggml_add_inplace(ctx, hidden_states, residual);

        return hidden_states;
    }

  protected:
    BasicBlock(Norm input_layernorm, BasicAttention attention, Norm post_attention_layernorm, MLP mlp)
        : input_layernorm(input_layernorm), attention(attention), post_attention_layernorm(post_attention_layernorm),
          mlp(mlp) {}

  public:
    Norm input_layernorm;
    BasicAttention attention;
    Norm post_attention_layernorm;
    MLP mlp;
};

struct NoopPositionIdsAllocator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen) const { return nullptr; }
};

struct BasicPositionIdsAllocator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen) const { return ggml_new_tensor_1d(ctx, GGML_TYPE_I32, qlen); }
};

struct GLMPositionIdsAllocator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen) const {
        return ggml_new_tensor_1d(ctx, GGML_TYPE_I32, qlen * 2);
    }
};

struct NoopAttentionMaskAllocator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen, int kvlen) const { return nullptr; }
};

struct BasicAttentionMaskAllocator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen, int kvlen) const {
        return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kvlen, qlen);
    }
};

template <typename Block, typename Norm, typename AttentionMaskAllocator, typename PositionIdsAllocator>
class BasicModel {
  public:
    BasicModel() = default;

    BasicModel(Embedding word_embeddings, std::vector<Block> layers, Norm final_layernorm)
        : word_embeddings(word_embeddings), layers(std::move(layers)), final_layernorm(final_layernorm) {}

    BasicModel(ModelContext *mctx, const ModelConfig &config)
        : word_embeddings(mctx, config.vocab_size, config.hidden_size), layers(build_layers(mctx, config)),
          final_layernorm(mctx, config.hidden_size) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input_ids, int n_past) const {
        ggml_context *ctx = mctx->ctx_b.get();

        const int qlen = input_ids->ne[0];
        const int kvlen = layers.front().attention.num_virtual_tokens + n_past + qlen;

        ggml_tensor *position_ids = pos_ids_alloc_(ctx, qlen);
        if (position_ids) {
            ggml_set_name(position_ids, "position_ids");
            ggml_set_input(position_ids);
        }

        ggml_tensor *attention_mask = attn_mask_alloc_(ctx, qlen, kvlen);
        if (attention_mask) {
            ggml_set_name(attention_mask, "attention_mask");
            ggml_set_input(attention_mask);
        }

        ggml_tensor *hidden_states = word_embeddings.forward(mctx, input_ids);
        for (const auto &layer : layers) {
            hidden_states = layer.forward(mctx, hidden_states, attention_mask, position_ids, n_past);
        }

        hidden_states = final_layernorm.forward(mctx, hidden_states);
        return hidden_states;
    }

    void load_prefix_cache(const ModelConfig &config, ggml_tensor *past_key_values) {
        // past_key_values: [l * 2, #h, v, d]
        ModelContext mctx(config.dtype);

        ggml_tensor *backend_past_key_values = ggml_new_tensor(mctx.ctx_kv.get(), past_key_values->type,
                                                               ggml_n_dims(past_key_values), past_key_values->ne);
        auto buf_kv =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx.ctx_kv.get(), mctx.backend.get()));
        ggml_backend_tensor_set(backend_past_key_values, past_key_values->data, 0, ggml_nbytes(past_key_values));
        past_key_values = backend_past_key_values;

        const int head_size = config.hidden_size / config.num_attention_heads;
        for (size_t i = 0; i < layers.size(); i++) {
            auto &attn = layers[i].attention;
            ggml_tensor *virtual_key =
                ggml_view_3d(mctx.ctx_b.get(), past_key_values, head_size, config.num_virtual_tokens,
                             config.num_key_value_heads, past_key_values->nb[1], past_key_values->nb[2],
                             i * 2 * past_key_values->nb[3]); // [#h, v, d]
            ggml_tensor *k_cache_view =
                ggml_view_3d(mctx.ctx_b.get(), attn.k_cache, head_size, config.num_virtual_tokens,
                             config.num_key_value_heads, attn.k_cache->nb[1], attn.k_cache->nb[2], 0); // [#h, v, d]
            ggml_build_forward_expand(mctx.gf, ggml_cpy(mctx.ctx_b.get(), virtual_key, k_cache_view));

            ggml_tensor *virtual_value = ggml_view_3d(
                mctx.ctx_b.get(), past_key_values, head_size, config.num_virtual_tokens, config.num_key_value_heads,
                past_key_values->nb[1], past_key_values->nb[2], (i * 2 + 1) * past_key_values->nb[3]); // [#h, v, d]
            virtual_value = ggml_permute(mctx.ctx_b.get(), virtual_value, 1, 0, 2, 3);                 // [#h, d, v]
            ggml_tensor *v_cache_view =
                ggml_view_3d(mctx.ctx_b.get(), attn.v_cache, config.num_virtual_tokens, head_size,
                             config.num_key_value_heads, attn.v_cache->nb[1], attn.v_cache->nb[2], 0); // [#h, d, v]
            ggml_build_forward_expand(mctx.gf, ggml_cpy(mctx.ctx_b.get(), virtual_value, v_cache_view));
        }

        CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx.allocr.get(), mctx.gf));
        CHATGLM_CHECK(ggml_backend_graph_compute(mctx.backend.get(), mctx.gf) == GGML_STATUS_SUCCESS);
    }

  private:
    std::vector<Block> build_layers(ModelContext *mctx, const ModelConfig &config) {
        std::vector<Block> layers;
        layers.reserve(config.num_hidden_layers);
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
            layers.emplace_back(mctx, config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
                                config.intermediate_size, config.max_length, config.norm_eps, config.hidden_act,
                                config.use_qkv_bias, config.use_dense_bias, config.interleaved_qkv, config.rope_type,
                                config.rope_theta, config.attn_mask_type, config.num_virtual_tokens);
        }
        mctx->buf_kv =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx->ctx_kv.get(), mctx->backend.get()));
        return layers;
    }

  public:
    Embedding word_embeddings;
    std::vector<Block> layers;
    Norm final_layernorm;

  private:
    AttentionMaskAllocator attn_mask_alloc_;
    PositionIdsAllocator pos_ids_alloc_;
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
        : os_(os), tokenizer_(tokenizer), is_prompt_(true), is_first_line_(true), print_len_(0) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    std::ostream &os_;
    BaseTokenizer *tokenizer_;
    bool is_prompt_;
    bool is_first_line_;
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

struct StateDict {
    unique_ggml_context_t ctx;
    unique_ggml_backend_buffer_t buf;
    std::unordered_map<std::string, ggml_tensor *> kv;
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

    StateDict read_state_dict();

  private:
    char *data;
    size_t size;
    char *ptr;
};

// ===== generation =====

struct GenerationConfig {
    int max_length;
    int max_new_tokens;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    float repetition_penalty;

    GenerationConfig(int max_length = 2048, int max_new_tokens = -1, int max_context_length = 512,
                     bool do_sample = true, int top_k = 0, float top_p = 0.7, float temperature = 0.95,
                     float repetition_penalty = 1.f)
        : max_length(max_length), max_new_tokens(max_new_tokens), max_context_length(max_context_length),
          do_sample(do_sample), top_k(top_k), top_p(top_p), temperature(temperature),
          repetition_penalty(repetition_penalty) {}
};

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
    BaseModelForCausalLM(ModelConfig config);
    virtual ~BaseModelForCausalLM() = default;

    virtual void load_state_dict(const StateDict &sd) = 0;

    virtual ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input_ids, int n_past, int n_ctx,
                                 bool is_decoding) const = 0;

    virtual void set_graph_inputs(int qlen, int n_past, int n_ctx) const = 0;

    ggml_tensor *forward_graph_compute(const std::vector<int> &input_ids, int n_past, int n_ctx, bool is_decoding);

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

  public:
    ModelConfig config;

  protected:
    std::unique_ptr<ModelContext> mctx_;
};

template <typename Model>
class BasicModelForCausalLM : public BaseModelForCausalLM {
  protected:
    BasicModelForCausalLM(const ModelConfig &config)
        : BaseModelForCausalLM(config), transformer(mctx_.get(), config),
          lm_head(mctx_.get(), config.hidden_size, config.vocab_size, false) {
        if (config.tie_word_embeddings) {
            lm_head.weight = transformer.word_embeddings.weight;
        }
    }

  public:
    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input_ids, int n_past, int n_ctx,
                         bool is_decoding) const override {
        ggml_tensor *transformer_outputs = transformer.forward(mctx, input_ids, n_past);
        // NOTE: only compute next token logits for decoding
        if (is_decoding && input_ids->ne[0] > 1) {
            transformer_outputs =
                ggml_view_1d(mctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                             (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
        }
        ggml_tensor *lm_logits = lm_head.forward(mctx, transformer_outputs);
        return lm_logits;
    }

    void load_prefix_cache(ggml_tensor *past_key_values) { transformer.load_prefix_cache(config, past_key_values); }

  public:
    Model transformer;
    Linear lm_head;
};

// ===== ChatGLM-6B =====

class ChatGLMTokenizer : public BaseTokenizer {
  public:
    ChatGLMTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const override;

    std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const override;

    static std::string apply_chat_template_text(const std::vector<ChatMessage> &messages);

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

// NOTE: disable inplace norm since it causes nonsense on cuda when sequence length >= 144
class GLMBlock : public BasicBlock<LayerNorm, BasicMLP> {
  public:
    GLMBlock() = default;
    GLMBlock(ModelContext *mctx, int hidden_size, int num_attention_heads, int num_key_value_heads,
             int intermediate_size, int max_length, float norm_eps, ActivationType hidden_act, bool use_qkv_bias,
             bool use_dense_bias, bool interleaved_qkv, RopeType rope_type, float rope_theta,
             AttentionMaskType attn_mask_type, int num_virtual_tokens)
        : BasicBlock(LayerNorm(mctx, hidden_size, false, norm_eps),
                     BasicAttention(mctx, hidden_size, num_attention_heads, num_attention_heads, max_length,
                                    use_qkv_bias, use_dense_bias, interleaved_qkv, rope_type, rope_theta,
                                    attn_mask_type, num_virtual_tokens),
                     LayerNorm(mctx, hidden_size, false, norm_eps),
                     BasicMLP(mctx, hidden_size, intermediate_size, hidden_act)),
          alpha(std::sqrt(2.f * 28)) {}

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                         ggml_tensor *position_ids, int n_past) const;

  public:
    float alpha;
};

using ChatGLMModel = BasicModel<GLMBlock, LayerNorm, BasicAttentionMaskAllocator, GLMPositionIdsAllocator>;

class ChatGLMForCausalLM : public BasicModelForCausalLM<ChatGLMModel> {
  public:
    ChatGLMForCausalLM(const ModelConfig &config);

    void load_state_dict(const StateDict &sd) override;

    void set_graph_inputs(int qlen, int n_past, int n_ctx) const override;

    static void set_graph_inputs(ggml_cgraph *gf, int qlen, int n_past, int n_ctx);

  private:
    StateDict state_dict() const;
};

// ===== ChatGLM2-6B =====

class ChatGLM2Tokenizer : public BaseTokenizer {
  public:
    ChatGLM2Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const override;

    std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const override;

    static std::string apply_chat_template_text(const std::vector<ChatMessage> &messages);

  private:
    bool is_special_id(int id) const;

  public:
    sentencepiece::SentencePieceProcessor sp;
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
};

using GLM2Block = BasicBlock<RMSNorm, BasicGLU>;

using ChatGLM2Model = BasicModel<GLM2Block, RMSNorm, NoopAttentionMaskAllocator, BasicPositionIdsAllocator>;

class ChatGLM2ForCausalLM : public BasicModelForCausalLM<ChatGLM2Model> {
  public:
    ChatGLM2ForCausalLM(const ModelConfig &config);

    void load_state_dict(const StateDict &sd) override;

    void set_graph_inputs(int qlen, int n_past, int n_ctx) const override;

    static void set_graph_inputs(ggml_cgraph *gf, int qlen, int n_past, int n_ctx);

  private:
    StateDict state_dict() const;
};

// ===== ChatGLM3-6B =====

class ChatGLM3Tokenizer : public BaseTokenizer {
  public:
    ChatGLM3Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const override;

    std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const override;

    ChatMessage decode_message(const std::vector<int> &ids) const override;

  private:
    std::vector<int> encode_single_message(const std::string &role, const std::string &content) const;

    static std::string remove_special_tokens(const std::string &text);

    int get_command(const std::string &token) const;

    bool is_special_id(int id) const;

    static void truncate(std::vector<int> &ids, int max_length);

  public:
    sentencepiece::SentencePieceProcessor sp;
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;
    int system_token_id;
    int user_token_id;
    int assistant_token_id;
    int observation_token_id;
    std::unordered_map<std::string, int> special_tokens;
    std::unordered_map<int, std::string> index_special_tokens;
};

using ChatGLM3Model = ChatGLM2Model;

using ChatGLM3ForCausalLM = ChatGLM2ForCausalLM;

// ===== ChatGLM4-9B =====

// C++ port of BPE algorithm from https://github.com/openai/tiktoken/blob/main/src/lib.rs
class TiktokenCoreBPE {
  public:
    TiktokenCoreBPE() = default;

    TiktokenCoreBPE(std::unordered_map<std::string, int> encoder,
                    std::unordered_map<std::string, int> special_tokens_encoder, const std::string &pattern);

    std::vector<int> encode_ordinary(const std::string &text) const { return _encode_ordinary_native(text); }

    std::string decode(const std::vector<int> &tokens) const { return _decode_native(tokens); }

  private:
    static std::vector<std::pair<size_t, int>> _byte_pair_merge(const std::unordered_map<std::string, int> &ranks,
                                                                const std::string &piece);

    static std::vector<int> byte_pair_encode(const std::string &piece,
                                             const std::unordered_map<std::string, int> &ranks);

    std::vector<int> _encode_ordinary_native(const std::string &text) const;

    std::string _decode_native(const std::vector<int> &tokens) const;

  public:
    std::unique_ptr<RE2> regex;
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<std::string, int> special_tokens_encoder;
    std::unordered_map<int, std::string> decoder;
    std::unordered_map<int, std::string> special_tokens_decoder;
};

class ChatGLM4Tokenizer : public BaseTokenizer {
  public:
    ChatGLM4Tokenizer(const std::string &vocab_text);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const override;

    std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const override;

    ChatMessage decode_message(const std::vector<int> &ids) const override;

  private:
    static void truncate(std::vector<int> &ids, int max_length);

  public:
    TiktokenCoreBPE core_bpe;
    // int eos_token_id;
    // int mask_token_id;
    int gmask_token_id;
    // int smask_token_id;
    int sop_token_id;
    // int eop_token_id;
    // int system_token_id;
    int user_token_id;
    int assistant_token_id;
    int observation_token_id;
};

using ChatGLM4Model = ChatGLM2Model;

using ChatGLM4ForCausalLM = ChatGLM2ForCausalLM;

// ===== pipeline =====

class Pipeline {
  public:
    Pipeline(const std::string &path, int max_length = -1);

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr) const;

    std::string generate(const std::string &prompt, const GenerationConfig &gen_config,
                         BaseStreamer *streamer = nullptr) const;

    ChatMessage chat(const std::vector<ChatMessage> &messages, const GenerationConfig &gen_config,
                     BaseStreamer *streamer = nullptr) const;

  protected:
    std::unique_ptr<MappedFile> mapped_file_;

  public:
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<BaseModelForCausalLM> model;
};

} // namespace chatglm
