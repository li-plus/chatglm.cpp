#pragma once

#include <cmath>
#include <ggml.h>
#include <iomanip>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>

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

enum class ModelType {
    CHATGLM = 1,
    CHATGLM2 = 2,
    CHATGLM3 = 3,
    BAICHUAN7B = 1024,
    BAICHUAN13B = 1025,
    INTERNLM = 1280,
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
struct ConfigRecordV2 : public ConfigRecordV1 {
    int num_kv_heads;
};

enum class ActivationType {
    GELU,
    SILU,
};

enum class RopeType {
    GPTJ = 0,
    NEOX = 2,
    CHATGLM = 4,
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
                int num_kv_heads, int num_hidden_layers, int intermediate_size, float norm_eps,
                ActivationType hidden_act, bool use_qkv_bias, bool use_dense_bias, bool interleaved_qkv, bool use_alibi,
                RopeType rope_type, int rope_dim_scale, AttentionMaskType attn_mask_type, int max_length,
                int bos_token_id, int eos_token_id, int pad_token_id, int sep_token_id,
                std::vector<int> extra_eos_token_ids)
        : model_type(model_type), dtype(dtype), vocab_size(vocab_size), hidden_size(hidden_size),
          num_attention_heads(num_attention_heads), num_kv_heads(num_kv_heads), num_hidden_layers(num_hidden_layers),
          intermediate_size(intermediate_size), norm_eps(norm_eps), hidden_act(hidden_act), use_qkv_bias(use_qkv_bias),
          use_dense_bias(use_dense_bias), interleaved_qkv(interleaved_qkv), use_alibi(use_alibi), rope_type(rope_type),
          rope_dim_scale(rope_dim_scale), attn_mask_type(attn_mask_type), max_length(max_length),
          bos_token_id(bos_token_id), eos_token_id(eos_token_id), pad_token_id(pad_token_id),
          sep_token_id(sep_token_id), extra_eos_token_ids(std::move(extra_eos_token_ids)) {}

    ModelConfig(ModelType model_type, const ConfigRecordV1 &rec, float norm_eps, ActivationType hidden_act,
                bool use_qkv_bias, bool use_dense_bias, bool interleaved_qkv, bool use_alibi, RopeType rope_type,
                int rope_dim_scale, AttentionMaskType attn_mask_type)
        : ModelConfig(model_type, rec.dtype, rec.vocab_size, rec.hidden_size, rec.num_attention_heads,
                      rec.num_attention_heads, rec.num_hidden_layers, rec.intermediate_size, norm_eps, hidden_act,
                      use_qkv_bias, use_dense_bias, interleaved_qkv, use_alibi, rope_type, rope_dim_scale,
                      attn_mask_type, rec.max_length, rec.bos_token_id, rec.eos_token_id, rec.pad_token_id,
                      rec.sep_token_id, {}) {}

    ModelConfig(ModelType model_type, const ConfigRecordV2 &rec, float norm_eps, ActivationType hidden_act,
                bool use_qkv_bias, bool use_dense_bias, bool interleaved_qkv, bool use_alibi, RopeType rope_type,
                int rope_dim_scale, AttentionMaskType attn_mask_type)
        : ModelConfig(model_type, rec.dtype, rec.vocab_size, rec.hidden_size, rec.num_attention_heads, rec.num_kv_heads,
                      rec.num_hidden_layers, rec.intermediate_size, norm_eps, hidden_act, use_qkv_bias, use_dense_bias,
                      interleaved_qkv, use_alibi, rope_type, rope_dim_scale, attn_mask_type, rec.max_length,
                      rec.bos_token_id, rec.eos_token_id, rec.pad_token_id, rec.sep_token_id, {}) {}

    std::string model_type_name() const { return to_string(model_type); }

  public:
    ModelType model_type;
    ggml_type dtype;
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_kv_heads;
    int num_hidden_layers;
    int intermediate_size;
    float norm_eps;
    ActivationType hidden_act;
    bool use_qkv_bias;
    bool use_dense_bias;
    bool interleaved_qkv;
    bool use_alibi;
    RopeType rope_type;
    int rope_dim_scale;
    AttentionMaskType attn_mask_type;
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

    virtual std::string decode(const std::vector<int> &ids) const = 0;

    virtual std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const = 0;

    virtual ChatMessage decode_message(const std::vector<int> &ids) const {
        return {ChatMessage::ROLE_ASSISTANT, decode(ids)};
    }

  protected:
    static void check_chat_messages(const std::vector<ChatMessage> &messages);
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
    LayerNorm() = default;
    LayerNorm(ModelContext *ctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)),
          bias(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]
    bool inplace;
    float eps;
};

class RMSNorm {
  public:
    RMSNorm() = default;
    RMSNorm(ModelContext *ctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    bool inplace;
    float eps;
};

class BasicMLP {
  public:
    BasicMLP() = default;
    BasicMLP(ModelContext *ctx, int hidden_size, int intermediate_size, ActivationType hidden_act)
        : dense_h_to_4h(ctx, hidden_size, intermediate_size), dense_4h_to_h(ctx, intermediate_size, hidden_size),
          hidden_act(hidden_act) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const;

  public:
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;
    ActivationType hidden_act;
};

class BasicGLU {
  public:
    BasicGLU() = default;
    BasicGLU(ModelContext *ctx, int hidden_size, int intermediate_size, ActivationType hidden_act)
        : gate_proj(ctx, hidden_size, intermediate_size, false), up_proj(ctx, hidden_size, intermediate_size, false),
          down_proj(ctx, intermediate_size, hidden_size, false), hidden_act(hidden_act) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const;

  public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    ActivationType hidden_act;
};

class BasicAttention {
  public:
    BasicAttention() = default;
    BasicAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length,
                   bool use_qkv_bias, bool use_dense_bias, bool interleaved_qkv, bool use_alibi, RopeType rope_type,
                   int rope_dim_scale, AttentionMaskType attn_mask_type)
        : num_attention_heads(num_attention_heads), num_kv_heads(num_kv_heads), interleaved_qkv(interleaved_qkv),
          use_alibi(use_alibi), rope_type(rope_type), rope_dim_scale(rope_dim_scale), attn_mask_type(attn_mask_type),
          query_key_value(ctx, hidden_size, hidden_size + 2 * (hidden_size / num_attention_heads) * num_kv_heads,
                          use_qkv_bias),
          dense(ctx, hidden_size, hidden_size, use_dense_bias),
          k_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                                     num_kv_heads)),
          v_cache(ggml_new_tensor_3d(ctx->ctx_kv.get(), GGML_TYPE_F16, max_length, hidden_size / num_attention_heads,
                                     num_kv_heads)) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *position_ids, int n_past,
                         int n_ctx) const;

  public:
    int num_attention_heads;
    int num_kv_heads;
    bool interleaved_qkv;
    bool use_alibi;
    RopeType rope_type;
    int rope_dim_scale;
    AttentionMaskType attn_mask_type;
    Linear query_key_value;
    Linear dense;
    ggml_tensor *k_cache; // [kv_heads, max_len, head_size]
    ggml_tensor *v_cache; // [kv_heads, head_size, max_len]
};

template <typename Norm, typename Attention, typename MLP>
class BasicBlock {
  public:
    BasicBlock() = default;
    BasicBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
               int max_length, float norm_eps, ActivationType hidden_act, bool use_qkv_bias, bool use_dense_bias,
               bool interleaved_qkv, bool use_alibi, RopeType rope_type, int rope_dim_scale,
               AttentionMaskType attn_mask_type)
        : input_layernorm(ctx, hidden_size, false, norm_eps),
          attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length, use_qkv_bias, use_dense_bias,
                    interleaved_qkv, use_alibi, rope_type, rope_dim_scale, attn_mask_type),
          post_attention_layernorm(ctx, hidden_size, false, norm_eps),
          mlp(ctx, hidden_size, intermediate_size, hidden_act) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *position_ids, int n_past,
                         int n_ctx) const {
        ggml_context *gctx = ctx->ctx_b.get();

        ggml_tensor *residual = hidden_states;
        hidden_states = input_layernorm.forward(ctx, hidden_states);
        hidden_states = attention.forward(ctx, hidden_states, position_ids, n_past, n_ctx);
        hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));

        residual = hidden_states;
        hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
        hidden_states = mlp.forward(ctx, hidden_states);
        hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, residual));

        return hidden_states;
    }

  protected:
    BasicBlock(Norm input_layernorm, Attention attention, Norm post_attention_layernorm, MLP mlp)
        : input_layernorm(input_layernorm), attention(attention), post_attention_layernorm(post_attention_layernorm),
          mlp(mlp) {}

  public:
    Norm input_layernorm;
    Attention attention;
    Norm post_attention_layernorm;
    MLP mlp;
};

struct NoopPositionIdsGenerator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen, int n_past, int n_ctx) const { return nullptr; }
};

struct BasicPositionIdsGenerator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen, int n_past, int n_ctx) const {
        ggml_tensor *position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, qlen);
        for (int i = 0; i < qlen; i++) {
            ((int *)position_ids->data)[i] = n_past + i;
        }
        return position_ids;
    }
};

struct GLMPositionIdsGenerator {
    ggml_tensor *operator()(ggml_context *ctx, int qlen, int n_past, int n_ctx) const {
        ggml_tensor *position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, qlen * 2);
        for (int i = 0; i < qlen; i++) {
            const int p = n_past + i;
            ((int *)position_ids->data)[i] = std::min(p, n_ctx - 2);
            ((int *)position_ids->data)[qlen + i] = std::max(p - (n_ctx - 2), 0);
        }
        return position_ids;
    }
};

template <typename Block, typename Norm, typename PositionIdsGenerator>
class BasicModel {
  public:
    BasicModel() = default;

    BasicModel(Embedding word_embeddings, std::vector<Block> layers, Norm final_layernorm)
        : word_embeddings(word_embeddings), layers(std::move(layers)), final_layernorm(final_layernorm) {}

    BasicModel(ModelContext *ctx, const ModelConfig &config)
        : word_embeddings(ctx, config.vocab_size, config.hidden_size), layers(build_layers(ctx, config)),
          final_layernorm(ctx, config.hidden_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const {
        ggml_context *gctx = ctx->ctx_b.get();
        ggml_tensor *position_ids = pos_ids_gen_(gctx, input_ids->ne[0], n_past, n_ctx);
        if (position_ids) {
            tensor_to_device(position_ids);
        }
        ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
        for (const auto &layer : layers) {
            ggml_set_scratch(gctx, ctx->scratch);
            hidden_states = layer.forward(ctx, hidden_states, position_ids, n_past, n_ctx);
        }
        if (position_ids) {
            tensor_to_cpu(position_ids);
        }
        ggml_scratch empty_scratch = {0, 0, nullptr};
        ggml_set_scratch(gctx, empty_scratch);
        hidden_states = final_layernorm.forward(ctx, hidden_states);
        return hidden_states;
    }

  private:
    std::vector<Block> build_layers(ModelContext *ctx, const ModelConfig &config) {
        std::vector<Block> layers;
        layers.reserve(config.num_hidden_layers);
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
            // TODO: reduce max length? 32k might be too large for cpu inference
            layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                                config.intermediate_size, config.max_length, config.norm_eps, config.hidden_act,
                                config.use_qkv_bias, config.use_dense_bias, config.interleaved_qkv, config.use_alibi,
                                config.rope_type, config.rope_dim_scale, config.attn_mask_type);
        }
        return layers;
    }

  public:
    Embedding word_embeddings;
    std::vector<Block> layers;
    Norm final_layernorm;

  private:
    PositionIdsGenerator pos_ids_gen_;
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
    int max_new_tokens;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    int num_threads;

    GenerationConfig(int max_length = 2048, int max_new_tokens = -1, int max_context_length = 512,
                     bool do_sample = true, int top_k = 0, float top_p = 0.7, float temperature = 0.95,
                     float repetition_penalty = 1.f, int num_threads = 0)
        : max_length(max_length), max_new_tokens(max_new_tokens), max_context_length(max_context_length),
          do_sample(do_sample), top_k(top_k), top_p(top_p), temperature(temperature),
          repetition_penalty(repetition_penalty), num_threads(num_threads) {}
};

int get_num_physical_cores();
int get_default_num_threads();

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
    BaseModelForCausalLM(ModelConfig config, size_t mem_size, size_t scratch_size, size_t num_weights);
    virtual ~BaseModelForCausalLM() = default;

    virtual void load(ModelLoader &loader) = 0;
    virtual ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx,
                                 bool is_decoding) const = 0;

    ggml_tensor *forward_graph_compute(const std::vector<int> &input_ids, int n_past, int n_ctx, int n_threads,
                                       bool is_decoding);

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
    ModelContext ctx_;

  public:
    ModelConfig config;
};

using StateDict = std::vector<std::pair<std::string, ggml_tensor *>>;

template <typename Model>
class BasicModelForCausalLM : public BaseModelForCausalLM {
  protected:
    BasicModelForCausalLM(const ModelConfig &config, size_t mem_size, size_t scratch_size, size_t num_weights)
        : BaseModelForCausalLM(config, mem_size, scratch_size, num_weights), transformer(&ctx_, config),
          lm_head(&ctx_, config.hidden_size, config.vocab_size, false) {
        CHATGLM_CHECK(ggml_used_mem(ctx_.ctx_w.get()) == ggml_get_mem_size(ctx_.ctx_w.get()))
            << "corrupted model weights";
        CHATGLM_CHECK(ggml_used_mem(ctx_.ctx_kv.get()) + 1 * MB == ggml_get_mem_size(ctx_.ctx_kv.get()))
            << "corrupted kv cache";
    }
    ~BasicModelForCausalLM() { to_cpu(); }

  public:
    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx,
                         bool is_decoding) const override {
        ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, n_past, n_ctx);
        // NOTE: only compute next token logits for decoding
        if (is_decoding && input_ids->ne[0] > 1) {
            transformer_outputs = tensor_assign_buffers(
                ggml_view_1d(ctx->ctx_b.get(), transformer_outputs, config.hidden_size,
                             (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs)));
        }
        ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
        return lm_logits;
    }

  protected:
    void to_cpu() {
        for (auto &item : state_dict_) {
            tensor_to_cpu(item.second);
        }

        for (auto &layer : transformer.layers) {
            tensor_to_cpu(layer.attention.k_cache);
            tensor_to_cpu(layer.attention.v_cache);
        }
    }

    void to_device() {
        for (auto &item : state_dict_) {
            ggml_tensor *tensor = item.second;
            // should not place embedding onto device
            if (tensor != transformer.word_embeddings.weight) {
                tensor_to_device(tensor);
            }
        }

        for (auto &layer : transformer.layers) {
            tensor_to_device(layer.attention.k_cache);
            tensor_to_device(layer.attention.v_cache);
        }
    }

  public:
    Model transformer;
    Linear lm_head;

  protected:
    StateDict state_dict_;
};

// ===== ChatGLM-6B =====

class ChatGLMTokenizer : public BaseTokenizer {
  public:
    ChatGLMTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

    static std::string build_prompt(const std::vector<ChatMessage> &messages);

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

// NOTE: disable inplace norm since it causes nonsense on cuda when sequence length >= 144
class GLMBlock : public BasicBlock<LayerNorm, BasicAttention, BasicMLP> {
  public:
    GLMBlock() = default;
    GLMBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
             int max_length, float norm_eps, ActivationType hidden_act, bool use_qkv_bias, bool use_dense_bias,
             bool interleaved_qkv, bool use_alibi, RopeType rope_type, int rope_dim_scale,
             AttentionMaskType attn_mask_type)
        : BasicBlock(
              LayerNorm(ctx, hidden_size, false, norm_eps),
              BasicAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, use_qkv_bias,
                             use_dense_bias, interleaved_qkv, use_alibi, rope_type, rope_dim_scale, attn_mask_type),
              LayerNorm(ctx, hidden_size, false, norm_eps), BasicMLP(ctx, hidden_size, intermediate_size, hidden_act)),
          alpha_value(std::sqrt(2.f * 28)) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, ggml_tensor *position_ids, int n_past,
                         int n_ctx) const;

  public:
    float alpha_value;
};

using ChatGLMModel = BasicModel<GLMBlock, LayerNorm, GLMPositionIdsGenerator>;

class ChatGLMForCausalLM : public BasicModelForCausalLM<ChatGLMModel> {
  public:
    ChatGLMForCausalLM(const ModelConfig &config);

    void load(ModelLoader &loader) override;

    static int num_weights(int num_hidden_layers) { return 4 + num_hidden_layers * 12; }

  private:
    StateDict state_dict() const;

  public:
    static constexpr size_t MEM_SIZE = 1280 * MB;     // 2k context
    static constexpr size_t SCRATCH_SIZE = 1024 * MB; // 2k context
};

// ===== ChatGLM2-6B =====

class ChatGLM2Tokenizer : public BaseTokenizer {
  public:
    ChatGLM2Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

    static std::string build_prompt(const std::vector<ChatMessage> &messages);

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

using GLM2Block = BasicBlock<RMSNorm, BasicAttention, BasicGLU>;

using ChatGLM2Model = BasicModel<GLM2Block, RMSNorm, BasicPositionIdsGenerator>;

class ChatGLM2ForCausalLM : public BasicModelForCausalLM<ChatGLM2Model> {
  public:
    ChatGLM2ForCausalLM(const ModelConfig &config);

    void load(ModelLoader &loader) override;

    static int num_weights(int num_hidden_layers) { return 3 + num_hidden_layers * 8; }

  private:
    StateDict state_dict() const;

  public:
    static constexpr size_t MEM_SIZE = 1280 * MB;     // 2k context
    static constexpr size_t SCRATCH_SIZE = 1280 * MB; // 2k context
};

// ===== ChatGLM3-6B =====

class ChatGLM3Tokenizer : public BaseTokenizer {
  public:
    ChatGLM3Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

    ChatMessage decode_message(const std::vector<int> &ids) const override;

  private:
    std::vector<int> encode_single_message(const std::string &role, const std::string &content) const;

    std::string decode_with_special_tokens(const std::vector<int> &ids) const;

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

// ===== Baichuan =====

class BaichuanTokenizer : public BaseTokenizer {
  public:
    BaichuanTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

  private:
    bool is_special_id(int id) const;

    static void truncate(std::vector<int> &ids, int max_length);

  public:
    static constexpr int USER_TOKEN_ID = 195;
    static constexpr int ASSISTANT_TOKEN_ID = 196;

    sentencepiece::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
};

// ===== Baichuan-7B =====

using Baichuan7BBlock = BasicBlock<RMSNorm, BasicAttention, BasicGLU>;

using Baichuan7BModel = BasicModel<Baichuan7BBlock, RMSNorm, BasicPositionIdsGenerator>;

class Baichuan7BForCausalLM : public BasicModelForCausalLM<Baichuan7BModel> {
  public:
    Baichuan7BForCausalLM(const ModelConfig &config);

    void load(ModelLoader &loader) override;

    static int num_weights(int num_hidden_layers) { return 3 + num_hidden_layers * 7; }

  private:
    StateDict state_dict() const;

  public:
    static constexpr size_t MEM_SIZE = 1280 * MB;
    static constexpr size_t SCRATCH_SIZE = 1280 * MB;
};

// ===== Baichuan-13B =====

using Baichuan13BBlock = BasicBlock<RMSNorm, BasicAttention, BasicGLU>;

using Baichuan13BModel = BasicModel<Baichuan13BBlock, RMSNorm, NoopPositionIdsGenerator>;

class Baichuan13BForCausalLM : public BasicModelForCausalLM<Baichuan13BModel> {
  public:
    Baichuan13BForCausalLM(const ModelConfig &config);

    void load(ModelLoader &loader) override;

    static int num_weights(int num_hidden_layers) { return 3 + num_hidden_layers * 7; }

  private:
    StateDict state_dict() const;

  public:
    static constexpr size_t MEM_SIZE = 1280 * MB;
    static constexpr size_t SCRATCH_SIZE = 1280 * MB;
};

// ===== InternLM =====

class InternLMTokenizer : public BaseTokenizer {
  public:
    InternLMTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text, int max_length) const override;

    std::string decode(const std::vector<int> &ids) const override;

    std::vector<int> encode_messages(const std::vector<ChatMessage> &messages, int max_length) const override;

    static std::string build_prompt(const std::vector<ChatMessage> &messages);

  private:
    bool is_special_id(int id) const { return id == unk_token_id || id == bos_token_id || id == eos_token_id; }

  public:
    sentencepiece::SentencePieceProcessor sp;
    static constexpr int unk_token_id = 0;
    static constexpr int bos_token_id = 1;
    static constexpr int eos_token_id = 2;
};

using InternLMBlock = BasicBlock<RMSNorm, BasicAttention, BasicGLU>;

using InternLMModel = BasicModel<InternLMBlock, RMSNorm, BasicPositionIdsGenerator>;

class InternLMForCausalLM : public BasicModelForCausalLM<InternLMModel> {
  public:
    InternLMForCausalLM(const ModelConfig &config);

    void load(ModelLoader &loader) override;

    static int num_weights(int num_hidden_layers, int hidden_size) {
        return 3 + num_hidden_layers * (hidden_size == 4096 ? 9 : 7);
    }

  private:
    StateDict state_dict() const;

  public:
    static constexpr size_t MEM_SIZE = 1280 * MB;
    static constexpr size_t SCRATCH_SIZE = 1280 * MB;
};

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

  public:
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<BaseModelForCausalLM> model;
    std::unique_ptr<MappedFile> mapped_file;
};

} // namespace chatglm
