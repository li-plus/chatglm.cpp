#pragma once

#include <ggml.h>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>
#include <vector>

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

void tensor_assign_buffers(ggml_tensor *tensor, bool scratch = true, bool force_inplace = false);

void tensor_to_device(ggml_tensor *tensor);

void tensor_to_cpu(ggml_tensor *tensor);

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
    virtual std::vector<int> encode(const std::string &text) const = 0;
    virtual std::string decode(const std::vector<int> &ids) const = 0;
    virtual std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const = 0;
};

struct ggml_context_deleter_t {
    void operator()(ggml_context *ctx) const noexcept { ggml_free(ctx); }
};

using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

unique_ggml_context_t make_unique_ggml_context(size_t mem_size, void *mem_buffer, bool no_alloc);

// reference: https://stackoverflow.com/questions/11149665/c-vector-that-doesnt-initialize-its-members
struct uninitialized_char {
    char m;
    uninitialized_char() {}
};

struct ModelContext {
    ggml_type dtype;
    unique_ggml_context_t ctx_w;  // weight
    unique_ggml_context_t ctx_kv; // kv cache
    unique_ggml_context_t ctx_b;  // buffer
    ggml_cgraph gf;
    ggml_scratch scratch;
    std::vector<uninitialized_char> compute_buffer; // BLAS buffer
    std::vector<uninitialized_char> scratch_buffer; // intermediate tensor buffer
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

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]
};

class RMSNorm {
  public:
    RMSNorm() : weight(nullptr), inplace(true) {}
    RMSNorm(ModelContext *ctx, int normalized_shape, bool inplace = true)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input) const;

  public:
    ggml_tensor *weight;
    bool inplace;
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
    ModelLoader(std::string_view buffer) : data(buffer.data()), size(buffer.size()), ptr(buffer.data()) {}

    int64_t tell() const { return ptr - data; }

    void seek(int64_t offset, int whence);

    template <typename T>
    T read_basic() {
        T obj = *(T *)ptr;
        ptr += sizeof(T);
        return obj;
    }

    std::string read_string(size_t length);

    void read_tensor(const std::string &name, ggml_tensor *tensor);

  public:
    const char *const data;
    size_t size;
    const char *ptr;
};

// ===== generation =====

struct GenerationConfig {
    int max_length;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    int num_threads;

    GenerationConfig(int max_length = 2048, int max_context_length = 512, bool do_sample = true, int top_k = 0,
                     float top_p = 0.7, float temperature = 0.95, int num_threads = 0)
        : max_length(max_length), max_context_length(max_context_length), do_sample(do_sample), top_k(top_k),
          top_p(top_p), temperature(temperature), num_threads(num_threads) {}
};

enum ModelType {
    MODEL_TYPE_CHATGLM = 1,
    MODEL_TYPE_CHATGLM2 = 2,
};

int get_num_physical_cores();

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

class BaseModelForConditionalGeneration {
  public:
    BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, size_t mem_size, size_t scratch_size);
    virtual ~BaseModelForConditionalGeneration() = default;

    virtual void load(ModelLoader &loader) = 0;
    virtual ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const = 0;

    ModelType type() const { return model_type_; }
    std::string type_name() const { return to_string(model_type_); }

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr);

    int generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, int n_past,
                            int n_ctx);

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

// ===== ChatGLM-6B =====

struct ChatGLMConfig : public BaseConfig {};

class ChatGLMTokenizer : public BaseTokenizer {
  public:
    ChatGLMTokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text) const override;

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

class GLMMLP {
  public:
    GLMMLP() = default;
    GLMMLP(ModelContext *ctx, int hidden_size)
        : dense_h_to_4h(ctx, hidden_size, 4 * hidden_size), dense_4h_to_h(ctx, 4 * hidden_size, hidden_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const;

  public:
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;
};

class GLMSelfAttention {
  public:
    // TODO: kv cache type
    GLMSelfAttention() : num_attention_heads(0) {}
    GLMSelfAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int max_length);

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;

  public:
    Linear query_key_value;
    Linear dense;
    int num_attention_heads;
    ggml_tensor *k_cache; // [n_head, maxlen, head_size]
    ggml_tensor *v_cache; // [n_head, head_size, maxlen]
};

class GLMBlock {
  public:
    GLMBlock() : num_hidden_layers(0) {}
    GLMBlock(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_hidden_layers, int max_length)
        : input_layernorm(ctx, hidden_size), attention(ctx, hidden_size, num_attention_heads, max_length),
          post_attention_layernorm(ctx, hidden_size), mlp(ctx, hidden_size), num_hidden_layers(num_hidden_layers) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;

  public:
    LayerNorm input_layernorm;
    GLMSelfAttention attention;
    LayerNorm post_attention_layernorm;
    GLMMLP mlp;
    int num_hidden_layers;
};

class ChatGLMModel {
  public:
    ChatGLMModel() = default;
    ChatGLMModel(ModelContext *ctx, const ChatGLMConfig &config);

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const;

  public:
    Embedding word_embeddings;
    std::vector<GLMBlock> layers;
    LayerNorm final_layernorm;
};

class ChatGLMForConditionalGeneration : public BaseModelForConditionalGeneration {
  public:
    ChatGLMForConditionalGeneration() = default;
    ChatGLMForConditionalGeneration(const ChatGLMConfig &config);
    ~ChatGLMForConditionalGeneration();

    void load(ModelLoader &loader) override;

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override;

  public:
    static constexpr size_t MEM_SIZE = 512 * MB;      // 2k context
    static constexpr size_t SCRATCH_SIZE = 1024 * MB; // 2k context
    ChatGLMConfig config;
    ChatGLMModel transformer;
    Linear lm_head;
};

// ===== ChatGLM2-6B =====

struct ChatGLM2Config : public BaseConfig {
    int num_kv_heads;
};

class ChatGLM2Tokenizer : public BaseTokenizer {
  public:
    ChatGLM2Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text) const override;

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

class GLM2SelfAttention {
  public:
    GLM2SelfAttention() : num_attention_heads(0), num_kv_heads(0) {}
    GLM2SelfAttention(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int max_length);

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past) const;

  public:
    int num_attention_heads;
    int num_kv_heads;
    Linear query_key_value;
    Linear dense;
    ggml_tensor *k_cache; // [mqa_n_head, maxlen, head_size]
    ggml_tensor *v_cache; // [mqa_n_head, head_size, maxlen]
};

class GLM2MLP {
  public:
    GLM2MLP() = default;
    GLM2MLP(ModelContext *ctx, int hidden_size, int intermediate_size)
        : dense_h_to_4h(ctx, hidden_size, intermediate_size * 2, false),
          dense_4h_to_h(ctx, intermediate_size, hidden_size, false) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states) const;

  public:
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;
};

class GLM2Block {
  public:
    GLM2Block() = default;
    GLM2Block(ModelContext *ctx, int hidden_size, int num_attention_heads, int num_kv_heads, int intermediate_size,
              int max_length)
        : input_layernorm(ctx, hidden_size, false),
          attention(ctx, hidden_size, num_attention_heads, num_kv_heads, max_length),
          post_attention_layernorm(ctx, hidden_size, false), mlp(ctx, hidden_size, intermediate_size) {}

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *hidden_states, int n_past) const;

  public:
    RMSNorm input_layernorm;
    GLM2SelfAttention attention;
    RMSNorm post_attention_layernorm;
    GLM2MLP mlp;
};

class ChatGLM2Model {
  public:
    ChatGLM2Model() = default;
    ChatGLM2Model(ModelContext *ctx, const ChatGLM2Config &config);

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past) const;

  public:
    Embedding word_embeddings;
    std::vector<GLM2Block> layers;
    RMSNorm final_layernorm;
};

class ChatGLM2ForConditionalGeneration : public BaseModelForConditionalGeneration {
  public:
    ChatGLM2ForConditionalGeneration() = default;
    ChatGLM2ForConditionalGeneration(const ChatGLM2Config &config);
    ~ChatGLM2ForConditionalGeneration();

    void load(ModelLoader &loader) override;

    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override;

  public:
    static constexpr size_t MEM_SIZE = 512 * MB;      // 2k context
    static constexpr size_t SCRATCH_SIZE = 1280 * MB; // 2k context

    ChatGLM2Config config;
    ChatGLM2Model transformer;
    Linear lm_head;
};

// ===== pipeline =====

class Pipeline {
  public:
    Pipeline(const std::string &path);

    std::string chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                     BaseStreamer *streamer = nullptr) const;

  public:
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<BaseModelForConditionalGeneration> model;
    std::unique_ptr<MappedFile> mapped_file;
};

} // namespace chatglm
