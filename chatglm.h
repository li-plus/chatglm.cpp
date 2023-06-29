#pragma once

#include <ggml.h>
#include <sentencepiece_processor.h>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace chatglm {

// ===== utils =====

struct LogMessageFatal {
    std::ostringstream oss;

    LogMessageFatal(const char *file, int line) { oss << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss.str()); }
    std::ostringstream &stream() { return oss; }
};
#define CHATGLM_THROW ::chatglm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATGLM_CHECK(cond)                                                                                            \
    if (!(cond))                                                                                                       \
    CHATGLM_THROW << "check failed (" #cond ") "

std::string to_string(ggml_tensor *tensor, bool with_data = true);

// ===== common =====

struct BaseTokenizer {
    virtual std::vector<int> encode(const std::string &text) const = 0;
    virtual std::string decode(const std::vector<int> &ids) const = 0;
};

struct BasePrompter {
    virtual std::string build_prompt(const std::vector<std::string> &history) const = 0;
};

struct InitContext {
    ggml_context *gctx;
    ggml_type dtype;
};

struct ForwardContext {
    ggml_context *gctx;
    ggml_cgraph gf;
    ggml_scratch scratch;
};

struct Embedding {
    ggml_tensor *weight;

    Embedding() : weight(nullptr) {}
    Embedding(InitContext *ctx, int num_embeddings, int embedding_dim)
        : weight(ggml_new_tensor_2d(ctx->gctx, ctx->dtype, embedding_dim, num_embeddings)) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) const;
};

struct Linear {
    ggml_tensor *weight; // [out_features, in_features]
    ggml_tensor *bias;   // [out_features]

    Linear() : weight(nullptr), bias(nullptr) {}
    Linear(InitContext *ctx, int in_features, int out_features, bool use_bias = true)
        : weight(ggml_new_tensor_2d(ctx->gctx, ctx->dtype, in_features, out_features)),
          bias(use_bias ? ggml_new_tensor_1d(ctx->gctx, GGML_TYPE_F32, out_features) : nullptr) {}

    int in_features() const { return weight->ne[0]; }
    int out_features() const { return weight->ne[1]; }

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) const;
};

struct LayerNorm {
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]

    LayerNorm() : weight(nullptr), bias(nullptr) {}
    LayerNorm(InitContext *ctx, int normalized_shape)
        : weight(ggml_new_tensor_1d(ctx->gctx, GGML_TYPE_F32, normalized_shape)),
          bias(ggml_new_tensor_1d(ctx->gctx, GGML_TYPE_F32, normalized_shape)) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) const;
};

struct RMSNorm {
    ggml_tensor *weight;

    RMSNorm() : weight(nullptr) {}
    RMSNorm(InitContext *ctx, int normalized_shape)
        : weight(ggml_new_tensor_1d(ctx->gctx, GGML_TYPE_F32, normalized_shape)) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input) const;
};

struct BaseStreamer {
    virtual void put(const std::vector<int> &output_ids) = 0;
    virtual void end() = 0;
};

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
struct TextStreamer : public BaseStreamer {
    TextStreamer(BaseTokenizer *tokenizer) : tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    BaseTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

struct MappedFile {
    char *data;
    size_t size;

    MappedFile(const std::string &path);
    ~MappedFile();
};

struct ModelLoader {
    const char *const data;
    size_t size;
    const char *ptr;

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

struct BaseModelForConditionalGeneration {
    BaseModelForConditionalGeneration() : eos_token_id_(0), max_seq_len_(0), mem_size_(0), scratch_size_(0) {}
    BaseModelForConditionalGeneration(int eos_token_id, int max_seq_len, size_t mem_size, size_t scratch_size)
        : eos_token_id_(eos_token_id), max_seq_len_(max_seq_len), mem_size_(mem_size), mem_buffer_(new char[mem_size]),
          scratch_size_(scratch_size), scratch_buffer_(new char[scratch_size]) {}
    virtual ~BaseModelForConditionalGeneration() = default;

    virtual void load(ModelLoader &loader) = 0;
    virtual ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const = 0;

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr) const;

    int generate_next_token(const std::vector<int> &input_ids, const GenerationConfig &gen_config, int n_past,
                            int n_ctx) const;

    struct TokenIdScore {
        int id;
        float score;

        bool operator<(const TokenIdScore &other) const { return score < other.score; }
        bool operator>(const TokenIdScore &other) const { return score > other.score; }
    };

  private:
    static void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last);

  private:
    int eos_token_id_;
    int max_seq_len_;
    size_t mem_size_;
    std::unique_ptr<char[]> mem_buffer_; // BLAS buffer
    size_t scratch_size_;
    std::unique_ptr<char[]> scratch_buffer_; // intermediate tensor buffer
};

// ===== ChatGLM-6B =====

struct ChatGLMConfig {
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int num_layers;
    int max_sequence_length;
    int bos_token_id;
    int eos_token_id;
    int gmask_token_id;
    int mask_token_id;
    int pad_token_id;
    ggml_type dtype;
};

struct ChatGLMTokenizer : public BaseTokenizer {
    sentencepiece::SentencePieceProcessor sp;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;
    int gmask_token_id;
    int pad_token_id;

    ChatGLMTokenizer(std::string_view serialized_model_proto, int bos_token_id, int eos_token_id, int mask_token_id,
                     int gmask_token_id, int pad_token_id);

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

  private:
    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);
};

struct ChatGLMPrompter : public BasePrompter {
    std::string build_prompt(const std::vector<std::string> &history) const override;
};

struct GLU {
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;

    GLU() = default;
    GLU(InitContext *ctx, int hidden_size)
        : dense_h_to_4h(ctx, hidden_size, 4 * hidden_size), dense_4h_to_h(ctx, 4 * hidden_size, hidden_size) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) const;
};

struct SelfAttention {
    Linear query_key_value;
    Linear dense;
    int num_attention_heads;
    ggml_tensor *k_cache; // [n_head, maxlen, head_size]
    ggml_tensor *v_cache; // [n_head, head_size, maxlen]

    // TODO: kvcache type
    SelfAttention() : num_attention_heads(0) {}
    SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length)
        : query_key_value(ctx, hidden_size, 3 * hidden_size), dense(ctx, hidden_size, hidden_size),
          num_attention_heads(num_attention_heads),
          k_cache(ggml_new_tensor_3d(ctx->gctx, GGML_TYPE_F16, hidden_size / num_attention_heads, max_length,
                                     num_attention_heads)),
          v_cache(ggml_new_tensor_3d(ctx->gctx, GGML_TYPE_F16, max_length, hidden_size / num_attention_heads,
                                     num_attention_heads)) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;
};

struct GLMBlock {
    LayerNorm input_layernorm;
    SelfAttention attention;
    LayerNorm post_attention_layernorm;
    GLU mlp;
    int num_layers;

    GLMBlock() : num_layers(0) {}
    GLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int num_layers, int max_length)
        : input_layernorm(ctx, hidden_size), attention(ctx, hidden_size, num_attention_heads, max_length),
          post_attention_layernorm(ctx, hidden_size), mlp(ctx, hidden_size), num_layers(num_layers) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const;
};

struct ChatGLMModel {
    Embedding word_embeddings;
    std::vector<GLMBlock> layers;
    LayerNorm final_layernorm;

    ChatGLMModel() = default;
    ChatGLMModel(InitContext *ctx, const ChatGLMConfig &config);

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const;
};

struct ChatGLMForConditionalGeneration : public BaseModelForConditionalGeneration {
    static constexpr size_t MEM_SIZE = 272ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 128ull * 1024 * 1024;

    ChatGLMForConditionalGeneration() = default;
    ChatGLMForConditionalGeneration(const ChatGLMConfig &config);
    ~ChatGLMForConditionalGeneration() override;

    void load(ModelLoader &loader) override;

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override;

  public:
    ChatGLMConfig config;
    ChatGLMModel transformer;

  private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
    // InitContext kv_ctx_; // TODO: kv cache context
    std::unique_ptr<char[]> kv_cache_buffer_;
};

// ===== ChatGLM2-6B =====

struct ChatGLM2Config {
    int vocab_size;
    int hidden_size;
    int num_attention_heads;
    int multi_query_group_num;
    int ffn_hidden_size;
    int num_layers;
    int seq_length;
    int eos_token_id;
    ggml_type dtype;
};

struct ChatGLM2Tokenizer : public BaseTokenizer {
    sentencepiece::SentencePieceProcessor sp;
    int mask_token_id;
    int gmask_token_id;
    int smask_token_id;
    int sop_token_id;
    int eop_token_id;

    ChatGLM2Tokenizer(std::string_view serialized_model_proto);

    std::vector<int> encode(const std::string &text) const override;

    std::string decode(const std::vector<int> &ids) const override;

    bool is_special_id(int id) const;
};

struct ChatGLM2Prompter : public BasePrompter {
    std::string build_prompt(const std::vector<std::string> &history) const override;
};

struct GLM2SelfAttention {
    int num_attention_heads;
    int multi_query_group_num;
    Linear query_key_value;
    Linear dense;
    ggml_tensor *k_cache; // [mqa_n_head, maxlen, head_size]
    ggml_tensor *v_cache; // [mqa_n_head, head_size, maxlen]

    GLM2SelfAttention() : num_attention_heads(0), multi_query_group_num(0) {}
    GLM2SelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int multi_query_group_num,
                      int max_length)
        : num_attention_heads(num_attention_heads), multi_query_group_num(multi_query_group_num),
          query_key_value(ctx, hidden_size,
                          hidden_size + 2 * (hidden_size / num_attention_heads) * multi_query_group_num),
          dense(ctx, hidden_size, hidden_size, false),
          k_cache(ggml_new_tensor_3d(ctx->gctx, GGML_TYPE_F32, hidden_size / num_attention_heads, max_length,
                                     multi_query_group_num)),
          v_cache(ggml_new_tensor_3d(ctx->gctx, GGML_TYPE_F32, max_length, hidden_size / num_attention_heads,
                                     multi_query_group_num)) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) const;
};

struct GLM2MLP {
    Linear dense_h_to_4h;
    Linear dense_4h_to_h;

    GLM2MLP(InitContext *ctx, int hidden_size, int ffn_hidden_size)
        : dense_h_to_4h(ctx, hidden_size, ffn_hidden_size * 2, false),
          dense_4h_to_h(ctx, ffn_hidden_size, hidden_size, false) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states) const;
};

struct GLM2Block {
    RMSNorm input_layernorm;
    GLM2SelfAttention attention;
    RMSNorm post_attention_layernorm;
    GLM2MLP mlp;

    GLM2Block() = default;
    GLM2Block(InitContext *ctx, int hidden_size, int num_attention_heads, int multi_query_group_num,
              int ffn_hidden_size, int max_length)
        : input_layernorm(ctx, hidden_size),
          attention(ctx, hidden_size, num_attention_heads, multi_query_group_num, max_length),
          post_attention_layernorm(ctx, hidden_size), mlp(ctx, hidden_size, ffn_hidden_size) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) const;
};

struct ChatGLM2Model {
    Embedding word_embeddings;
    std::vector<GLM2Block> layers;
    RMSNorm final_layernorm;

    ChatGLM2Model() = default;
    ChatGLM2Model(InitContext *ctx, const ChatGLM2Config &config);

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) const;
};

struct ChatGLM2ForConditionalGeneration : public BaseModelForConditionalGeneration {
    static constexpr size_t MEM_SIZE = 272ull * 1024 * 1024;
    static constexpr size_t SCRATCH_SIZE = 144ull * 1024 * 1024;

    ChatGLM2ForConditionalGeneration() = default;
    ChatGLM2ForConditionalGeneration(const ChatGLM2Config &config);

    void load(ModelLoader &loader) override;

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override;

  public:
    ChatGLM2Config config;
    ChatGLM2Model transformer;
    Linear lm_head;

  private:
    // hold ggml_context & kv_cache
    InitContext w_ctx_; // weight context
    // InitContext kv_ctx_; // TODO: kv cache context
    std::unique_ptr<char[]> kv_cache_buffer_;
};

struct ChatGLM2Pipeline {
    std::unique_ptr<ChatGLM2Tokenizer> tokenizer;
    std::unique_ptr<ChatGLM2ForConditionalGeneration> model;
    InitContext ctx;
    std::unique_ptr<char[]> kvcache_buffer;
    std::unique_ptr<MappedFile> mapped_file;

    ChatGLM2Pipeline(const std::string &path);
    ~ChatGLM2Pipeline();

    std::string chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                     BaseStreamer *streamer = nullptr) const;
};

// ===== pipeline =====

struct Pipeline {
    std::unique_ptr<BaseTokenizer> tokenizer;
    std::unique_ptr<BaseModelForConditionalGeneration> model;
    std::unique_ptr<BasePrompter> prompter;
    std::unique_ptr<MappedFile> mapped_file;

    Pipeline(const std::string &path);

    std::string chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                     BaseStreamer *streamer = nullptr) const;

    static std::string build_prompt(const std::vector<std::string> &history);
};

} // namespace chatglm
