#pragma once

#include <ggml/ggml.h>
#include <sentencepiece_processor.h>
#include <sstream>
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

// ===== config =====

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

// ===== tokenizer =====

struct ChatGLMTokenizer {
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp;
    int bos_token_id;
    int eos_token_id;
    int mask_token_id;
    int gmask_token_id;
    int pad_token_id;

    ChatGLMTokenizer(std::string_view serialized_model_proto, int _bos_token_id, int _eos_token_id, int _mask_token_id,
                     int _gmask_token_id, int _pad_token_id);

    std::vector<int> encode(const std::string &text) const;

    std::string decode(const std::vector<int> &ids) const;

    static std::string preprocess(const std::string &text);

    static std::string postprocess(const std::string &text);
};

// ===== streamer =====

struct BaseStreamer {
    virtual void put(const std::vector<int> &output_ids) = 0;
    virtual void end() = 0;
};

// reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
struct TextStreamer : public BaseStreamer {
    TextStreamer(ChatGLMTokenizer *tokenizer) : tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
    void put(const std::vector<int> &output_ids) override;
    void end() override;

  private:
    ChatGLMTokenizer *tokenizer_;
    bool is_prompt_;
    std::vector<int> token_cache_;
    int print_len_;
};

// ===== model =====

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
    Linear(InitContext *ctx, int in_features, int out_features)
        : weight(ggml_new_tensor_2d(ctx->gctx, ctx->dtype, in_features, out_features)),
          bias(ggml_new_tensor_1d(ctx->gctx, GGML_TYPE_F32, out_features)) {}

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
    SelfAttention(InitContext *ctx, int hidden_size, int _num_attention_heads, int max_length)
        : query_key_value(ctx, hidden_size, 3 * hidden_size), dense(ctx, hidden_size, hidden_size),
          num_attention_heads(_num_attention_heads),
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
    GLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int _num_layers, int max_length)
        : input_layernorm(ctx, hidden_size), attention(ctx, hidden_size, num_attention_heads, max_length),
          post_attention_layernorm(ctx, hidden_size), mlp(ctx, hidden_size), num_layers(_num_layers) {}

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

struct GenerationConfig {
    int max_length = 2048;
    int max_context_length = 512;
    bool do_sample = true;
    int top_k = 0;
    float top_p = 0.7;
    float temperature = 0.95;
    int num_threads = 0;
};

struct ChatGLMForConditionalGeneration {
    ChatGLMConfig config;
    ChatGLMModel transformer;

    ChatGLMForConditionalGeneration() = default;
    ChatGLMForConditionalGeneration(InitContext *ctx, const ChatGLMConfig &_config)
        : config(_config), transformer(ctx, config) {}

    ggml_tensor *forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const;

    std::vector<int> generate(const std::vector<int> &input_ids, const GenerationConfig &gen_config,
                              BaseStreamer *streamer = nullptr) const;
};

struct MappedFile {
    void *data;
    size_t size;

    MappedFile(const std::string &path);
    ~MappedFile();
};

struct ChatGLMPipeline {
    std::unique_ptr<ChatGLMTokenizer> tokenizer;
    std::unique_ptr<ChatGLMForConditionalGeneration> model;
    InitContext ctx;
    std::unique_ptr<char[]> kvcache_buffer;
    std::unique_ptr<MappedFile> mapped_file;

    ChatGLMPipeline(const std::string &path);
    ~ChatGLMPipeline();

    void chat(std::vector<std::string> &history, const GenerationConfig &gen_config,
              BaseStreamer *streamer = nullptr) const;

    static std::string build_prompt(const std::vector<std::string> &history);
};

} // namespace chatglm
