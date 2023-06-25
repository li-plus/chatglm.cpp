#include "chatglm.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

namespace chatglm {

template <typename T>
static inline T read_as(std::istream &is) {
    T data;
    is.read((char *)&data, sizeof(data));
    return data;
}

static std::string shape_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = tensor->n_dims - 1; i >= 0; i--) {
        oss << tensor->ne[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

static std::string strides_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = tensor->n_dims - 1; i >= 0; i--) {
        oss << tensor->nb[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

std::string to_string(ggml_tensor *tensor, bool with_data) {
    std::ostringstream oss;
    oss << "ggml_tensor(";

    if (with_data) {
        oss << "[";
        for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                                   i0 * tensor->nb[0];
                        float val;
                        if (tensor->type == GGML_TYPE_F32) {
                            val = *(float *)ptr;
                        } else if (tensor->type == GGML_TYPE_F16) {
                            val = ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
                        } else {
                            CHATGLM_THROW << "unimplemented";
                        }
                        oss << (i3 + i2 + i1 + i0 > 0 ? ", " : "") << std::fixed << std::setprecision(4) << val;
                    }
                }
            }
        }
        oss << "], ";
    }

    oss << "shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
    return oss.str();
}

// ===== streamer =====

void TextStreamer::put(const std::vector<int> &output_ids) {
    if (is_prompt_) {
        // skip prompt
        is_prompt_ = false;
        return;
    }

    static const std::vector<char> puncts{',', '!', ':', ';', '?'};

    token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
    std::string text = tokenizer_->decode(token_cache_);
    if (text.empty()) {
        return;
    }

    std::string printable_text;
    if (text.back() == '\n') {
        // flush the cache after newline
        printable_text = text.substr(print_len_);
        token_cache_.clear();
        print_len_ = 0;
    } else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
        // last symbol is a punctuation, hold on
    } else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
    } else {
        printable_text = text.substr(print_len_);
        print_len_ = text.size();
    }

    std::cout << printable_text << std::flush;
}

void TextStreamer::end() {
    std::string text = tokenizer_->decode(token_cache_);
    std::cout << text.substr(print_len_) << std::endl;
    is_prompt_ = true;
    token_cache_.clear();
    print_len_ = 0;
}

// ===== tokenizer =====

ChatGLMTokenizer::ChatGLMTokenizer(std::string_view serialized_model_proto, int bos_token_id, int eos_token_id,
                                   int mask_token_id, int gmask_token_id, int pad_token_id)
    : bos_token_id(bos_token_id), eos_token_id(eos_token_id), mask_token_id(mask_token_id),
      gmask_token_id(gmask_token_id), pad_token_id(pad_token_id) {
    sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = sp->LoadFromSerializedProto(serialized_model_proto);
    CHATGLM_CHECK(status.ok()) << status.ToString();
}

std::vector<int> ChatGLMTokenizer::encode(const std::string &text) const {
    std::string input = preprocess(text);
    std::vector<int> ids;
    sp->Encode(input, &ids);
    ids.push_back(gmask_token_id);
    ids.push_back(bos_token_id);
    return ids;
}

std::string ChatGLMTokenizer::decode(const std::vector<int> &ids) const {
    std::string text;
    sp->Decode(ids, &text);
    text = postprocess(text);
    return text;
}

static std::string regex_replace(const std::string &input, const std::regex &regex,
                                 std::function<std::string(const std::smatch &)> format) {
    std::ostringstream oss;
    int last_index = 0;
    for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++) {
        oss << it->prefix() << format(*it);
        last_index = it->position() + it->length();
    }
    oss << input.substr(last_index);
    return oss.str();
}

std::string ChatGLMTokenizer::preprocess(const std::string &text) {
    std::string output;

    // newline token
    {
        static const std::regex newline_regex("\n");
        output = std::regex_replace(text, newline_regex, "<n>");
    }
    // tab token
    {
        static const std::regex tab_regex("\t");
        output = std::regex_replace(output, tab_regex, "<|tab|>");
    }
    // blank tokens
    {
        static const std::regex pattern(R"([ ]{2,80})");
        output = regex_replace(output, pattern, [](const std::smatch &sm) {
            std::ostringstream oss;
            oss << "<|blank_" << sm.str().size() << "|>";
            return oss.str();
        });
    }

    return output;
}

std::string ChatGLMTokenizer::postprocess(const std::string &text) {
    std::string output;

    // newline token
    {
        static const std::regex pattern(R"(<n>)");
        output = std::regex_replace(text, pattern, "\n");
    }
    // tab token
    {
        static const std::regex pattern(R"(<\|tab\|>)");
        output = std::regex_replace(output, pattern, "\t");
    }
    // blank tokens
    {
        static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
        output = regex_replace(output, pattern,
                               [](const std::smatch &sm) { return std::string(std::stoi(sm[1].str()), ' '); });
    }

    // replace punctuations
    // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    {
        static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        static const std::vector<std::pair<std::wregex, std::wstring>> punct_map{
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]),)")), converter.from_bytes("$1，")},
            {std::wregex(converter.from_bytes(R"(,([\u4e00-\u9fff]))")), converter.from_bytes("，$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])!)")), converter.from_bytes("$1！")},
            {std::wregex(converter.from_bytes(R"(!([\u4e00-\u9fff]))")), converter.from_bytes("！$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]):)")), converter.from_bytes("$1：")},
            {std::wregex(converter.from_bytes(R"(:([\u4e00-\u9fff]))")), converter.from_bytes("：$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]);)")), converter.from_bytes("$1；")},
            {std::wregex(converter.from_bytes(R"(;([\u4e00-\u9fff]))")), converter.from_bytes("；$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])\?)")), converter.from_bytes("$1？")},
            {std::wregex(converter.from_bytes(R"(\?([\u4e00-\u9fff]))")), converter.from_bytes("？$1")},
        };
        std::wstring w_output = converter.from_bytes(output);
        for (const auto &punct_pair : punct_map) {
            w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
        }
        output = converter.to_bytes(w_output);
    }

    return output;
}

// ===== model =====

ggml_tensor *Embedding::forward(ForwardContext *ctx, ggml_tensor *input) const {
    ggml_tensor *output = ggml_get_rows(ctx->gctx, weight, input);
    return output;
}

ggml_tensor *Linear::forward(ForwardContext *ctx, ggml_tensor *input) const {
    // input: [seqlen, in_features]
    ggml_tensor *output = ggml_mul_mat(ctx->gctx, weight, input); // [seqlen, out_features]
    ggml_tensor *bcast_bias = ggml_view_2d(ctx->gctx, bias, output->ne[0], output->ne[1], 0, 0);
    output = ggml_add_inplace(ctx->gctx, output, bcast_bias);
    return output;
}

ggml_tensor *LayerNorm::forward(ForwardContext *ctx, ggml_tensor *input) const {
    // input: [seqlen, normalized_shape]
    ggml_tensor *output = ggml_norm_inplace(ctx->gctx, input);
    ggml_tensor *bcast_weight = ggml_view_2d(ctx->gctx, weight, output->ne[0], output->ne[1], 0, 0);
    output = ggml_mul_inplace(ctx->gctx, output, bcast_weight);
    ggml_tensor *bcast_bias = ggml_view_2d(ctx->gctx, bias, output->ne[0], output->ne[1], 0, 0);
    output = ggml_add_inplace(ctx->gctx, output, bcast_bias);
    return output;
}

ggml_tensor *GLU::forward(ForwardContext *ctx, ggml_tensor *hidden_states) const {
    ggml_tensor *output = dense_h_to_4h.forward(ctx, hidden_states);
    output = ggml_gelu_inplace(ctx->gctx, output);
    output = dense_4h_to_h.forward(ctx, output);
    return output;
}

ggml_tensor *SelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const {
    int hidden_size = hidden_states->ne[0];
    int qlen = hidden_states->ne[1];
    int head_size = hidden_size / num_attention_heads;
    int rope_dim = head_size / 2;

    ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, 3 * hidden]

    ggml_tensor *query_layer = ggml_view_3d(ctx->gctx, qkv, head_size, num_attention_heads, qlen,
                                            3 * head_size * ggml_element_size(qkv), qkv->nb[1], 0);
    query_layer = ggml_rope_inplace(ctx->gctx, query_layer, n_past, rope_dim, 4, n_ctx); // [qlen, n_head, head_size]
    query_layer = ggml_permute(ctx->gctx, query_layer, 0, 2, 1, 3);                      // [n_head, qlen, head_size]

    ggml_tensor *key_layer =
        ggml_view_3d(ctx->gctx, qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                     qkv->nb[1], head_size * ggml_element_size(qkv));
    key_layer = ggml_rope_inplace(ctx->gctx, key_layer, n_past, rope_dim, 4, n_ctx); // [qlen, n_head, head_size]
    key_layer = ggml_permute(ctx->gctx, key_layer, 0, 2, 1, 3);                      // [n_head, qlen, head_size]

    ggml_tensor *value_layer = ggml_view_3d(ctx->gctx, qkv, head_size, num_attention_heads, qlen,
                                            3 * head_size * ggml_element_size(qkv), qkv->nb[1],
                                            2 * head_size * ggml_element_size(qkv)); // [qlen, n_head, head_size]
    value_layer = ggml_permute(ctx->gctx, value_layer, 1, 2, 0, 3);                  // [n_head, head_size, qlen]

    // store key & value to cache
    ggml_tensor *k_cache_view =
        ggml_view_3d(ctx->gctx, k_cache, head_size, qlen, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                     n_past * head_size * ggml_element_size(k_cache)); // [n_head, qlen, head_size]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx, key_layer, k_cache_view));
    ggml_tensor *v_cache_view =
        ggml_view_3d(ctx->gctx, v_cache, qlen, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                     n_past * ggml_element_size(v_cache)); // [n_head, head_size, qlen]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx, value_layer, v_cache_view));

    key_layer = ggml_view_3d(ctx->gctx, k_cache, head_size, n_past + qlen, num_attention_heads, k_cache->nb[1],
                             k_cache->nb[2], 0); // [n_head, klen, head_size]
    value_layer = ggml_view_3d(ctx->gctx, v_cache, n_past + qlen, head_size, num_attention_heads, v_cache->nb[1],
                               v_cache->nb[2], 0); // [n_head, head_size, klen]

    ggml_tensor *attn_scores = ggml_mul_mat(ctx->gctx, key_layer, query_layer); // [n_head, qlen, klen]
    if (n_past == 0) {
        // build attention mask for context input
        ggml_tensor *inf = ggml_new_tensor_3d(ctx->gctx, attn_scores->type, 1, qlen - 1, num_attention_heads);
        ggml_set_f32(inf, -INFINITY);
        ggml_tensor *masked_attn_scores = ggml_view_3d(
            ctx->gctx, attn_scores, 1, qlen - 1, num_attention_heads, qlen * ggml_element_size(attn_scores),
            qlen * qlen * ggml_element_size(attn_scores), (qlen - 1) * ggml_element_size(attn_scores));
        ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx, inf, masked_attn_scores));
    }
    attn_scores = ggml_scale_inplace(ctx->gctx, attn_scores, ggml_new_f32(ctx->gctx, 1.f / std::sqrt(head_size)));
    ggml_tensor *attn_probs = ggml_soft_max_inplace(ctx->gctx, attn_scores); // [n_head, qlen, klen]

    ggml_tensor *context_layer = ggml_mul_mat(ctx->gctx, value_layer, attn_probs); // [n_head, qlen, head_size]
    context_layer = ggml_reshape_2d(ctx->gctx, ggml_cont(ctx->gctx, ggml_permute(ctx->gctx, context_layer, 0, 2, 1, 3)),
                                    hidden_size, qlen);

    ggml_tensor *attn_output = dense.forward(ctx, context_layer);
    return attn_output;
}

ggml_tensor *GLMBlock::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const {
    ggml_tensor *alpha = ggml_new_f32(ctx->gctx, std::sqrt(2.f * num_layers));

    ggml_tensor *attn_input = input_layernorm.forward(ctx, hidden_states);
    ggml_tensor *attn_output = attention.forward(ctx, attn_input, n_past, n_ctx);
    ggml_build_forward_expand(&ctx->gf, attn_output);
    hidden_states = ggml_add_inplace(ctx->gctx, ggml_scale_inplace(ctx->gctx, attn_input, alpha), attn_output);

    ggml_tensor *mlp_input = post_attention_layernorm.forward(ctx, hidden_states);
    ggml_tensor *mlp_output = mlp.forward(ctx, mlp_input);
    ggml_build_forward_expand(&ctx->gf, mlp_output);
    ggml_tensor *output = ggml_add_inplace(ctx->gctx, ggml_scale_inplace(ctx->gctx, mlp_input, alpha), mlp_output);
    return output;
}

ChatGLMModel::ChatGLMModel(InitContext *ctx, const ChatGLMConfig &config)
    : word_embeddings(ctx, config.vocab_size, config.hidden_size), final_layernorm(ctx, config.hidden_size) {
    layers.reserve(config.num_layers);
    for (int layer_id = 0; layer_id < config.num_layers; layer_id++) {
        layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_layers,
                            config.max_sequence_length);
    }
}

ggml_tensor *ChatGLMModel::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const {
    ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
    for (const GLMBlock &layer : layers) {
        ggml_set_scratch(ctx->gctx, ctx->scratch);
        hidden_states = layer.forward(ctx, hidden_states, n_past, n_ctx);
    }
    ggml_set_scratch(ctx->gctx, {.offs = 0, .size = 0, .data = nullptr});
    hidden_states = final_layernorm.forward(ctx, hidden_states);
    return hidden_states;
}

ggml_tensor *ChatGLMForConditionalGeneration::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past,
                                                      int n_ctx) const {
    ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, n_past, n_ctx);
    // NOTE: only compute next_token_logits for the last token
    if (input_ids->ne[0] > 1) {
        transformer_outputs =
            ggml_view_1d(ctx->gctx, transformer_outputs, config.hidden_size,
                         (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
    }
    ggml_tensor *lm_head_weight = transformer.word_embeddings.weight; // tied weight
    ggml_tensor *lm_logits = ggml_mul_mat(ctx->gctx, lm_head_weight, transformer_outputs);
    return lm_logits;
}

struct TokenIdScore {
    int id;
    float score;

    friend bool operator<(const TokenIdScore &a, const TokenIdScore &b) { return a.score < b.score; }
    friend bool operator>(const TokenIdScore &a, const TokenIdScore &b) { return a.score > b.score; }
};

static void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore *p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore *p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
static inline int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

int ChatGLMForConditionalGeneration::generate_next_token(const std::vector<int> &input_ids,
                                                         const GenerationConfig &gen_config, int n_past,
                                                         int n_ctx) const {
    ForwardContext ctx;
    ctx.gctx = ggml_init({.mem_size = MEM_SIZE, .mem_buffer = mem_buffer.get(), .no_alloc = false});
    CHATGLM_CHECK(ctx.gctx) << "failed to init ggml context";
    int n_threads = gen_config.num_threads > 0 ? gen_config.num_threads : get_num_physical_cores();
    ctx.gf = {};
    ctx.gf.n_threads = input_ids.size() >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : n_threads;
    ctx.scratch = {.offs = 0, .size = SCRATCH_SIZE, .data = scratch_buffer.get()};

    ggml_tensor *input_ids_tensor = ggml_new_tensor_1d(ctx.gctx, GGML_TYPE_I32, input_ids.size());
    memcpy(input_ids_tensor->data, input_ids.data(), ggml_nbytes(input_ids_tensor));

    ggml_tensor *lm_logits = forward(&ctx, input_ids_tensor, n_past, n_ctx);

    ggml_build_forward_expand(&ctx.gf, lm_logits);
    ggml_graph_compute(ctx.gctx, &ctx.gf);

#ifdef GGML_PERF
    ggml_graph_print(&ctx.gf);
#endif

    float *next_token_logits = (float *)lm_logits->data;

    int next_token_id;
    if (gen_config.do_sample) {
        // temperature sampling
        float inv_temp = 1.f / gen_config.temperature;
        for (int i = 0; i < config.vocab_size; i++) {
            next_token_logits[i] *= inv_temp;
        }

        std::vector<TokenIdScore> token_scores(config.vocab_size);
        for (int i = 0; i < config.vocab_size; i++) {
            token_scores[i] = {.id = i, .score = next_token_logits[i]};
        }

        // top_k sampling
        if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
            std::nth_element(token_scores.begin(), token_scores.begin() + gen_config.top_k, token_scores.end(),
                             std::greater<TokenIdScore>());
            token_scores.resize(gen_config.top_k);
        }

        // top_p sampling
        if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
            std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!
            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

            float cumsum = 0.f;
            for (size_t i = 0; i < token_scores.size(); i++) {
                cumsum += token_scores[i].score;
                if (cumsum >= gen_config.top_p) {
                    token_scores.resize(i + 1);
                    break;
                }
            }
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            next_token_logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
        next_token_id = token_scores[dist(gen)].id;
    } else {
        // greedy search
        next_token_id = std::max_element(next_token_logits, next_token_logits + config.vocab_size) - next_token_logits;
    }
    ggml_free(ctx.gctx);

    return next_token_id;
}

std::vector<int> ChatGLMForConditionalGeneration::generate(const std::vector<int> &input_ids,
                                                           const GenerationConfig &gen_config,
                                                           BaseStreamer *streamer) const {
    CHATGLM_CHECK(gen_config.max_length <= config.max_sequence_length)
        << "max_length (" << gen_config.max_length << ") is larger than model max_sequence_length ("
        << config.max_sequence_length << ")";
    std::vector<int> curr_input_ids(input_ids);

    std::vector<int> output_ids;
    output_ids.reserve(gen_config.max_length);
    output_ids = input_ids;
    if (streamer) {
        streamer->put(input_ids);
    }

    int n_past = 0;
    const int n_ctx = input_ids.size();

    while ((int)output_ids.size() < gen_config.max_length) {
        int next_token_id = generate_next_token(curr_input_ids, gen_config, n_past, n_ctx);

        n_past += curr_input_ids.size();
        curr_input_ids = {next_token_id};
        output_ids.emplace_back(next_token_id);

        if (streamer) {
            streamer->put({next_token_id});
        }

        if (next_token_id == config.eos_token_id) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return output_ids;
}

MappedFile::MappedFile(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct stat sb;
    CHATGLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHATGLM_CHECK(data != MAP_FAILED) << strerror(errno);

    CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { CHATGLM_CHECK(munmap(data, size) == 0) << strerror(errno); }

ChatGLMPipeline::ChatGLMPipeline(const std::string &path) {
    mapped_file = std::make_unique<MappedFile>(path);

    std::ifstream fin(path, std::ios::binary);
    CHATGLM_CHECK(fin) << "failed to open model file " << path;

    // load magic
    std::string magic(4, '\0');
    fin.read(&magic[0], magic.size());
    CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

    // load config
    ChatGLMConfig config = read_as<ChatGLMConfig>(fin);

    // load tokenizer
    int proto_size = read_as<int>(fin);
    std::string_view serialized_model_proto((char *)mapped_file->data + fin.tellg(), proto_size);
    fin.seekg(proto_size, fin.cur);
    tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto, config.bos_token_id, config.eos_token_id,
                                                   config.mask_token_id, config.gmask_token_id, config.pad_token_id);

    // load model
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_layers * 14;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    ctx.gctx = ggml_init({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
    CHATGLM_CHECK(ctx.gctx) << "failed to init weight context";
    ctx.dtype = config.dtype;

    const size_t kvcache_size =
        config.num_layers * 2 * config.max_sequence_length * config.hidden_size * ggml_type_size(GGML_TYPE_F16);
    kvcache_buffer.reset(new char[kvcache_size]);
    char *kvcache_ptr = kvcache_buffer.get();

    auto load_tensor = [this, &fin](const std::string &name, ggml_tensor *tensor) {
        int ndim = read_as<int>(fin);
        CHATGLM_CHECK(ndim == tensor->n_dims)
            << "tensor " << name << " ndim mismatch: expect " << tensor->n_dims << ", got " << ndim;
        int name_size = read_as<int>(fin);
        ggml_type dtype = (ggml_type)read_as<int>(fin);
        CHATGLM_CHECK(dtype == tensor->type)
            << "tensor " << name << " dtype mismatch: expect " << tensor->type << ", got " << dtype;
        for (int i = 0; i < ndim; i++) {
            int dim = read_as<int>(fin);
            CHATGLM_CHECK(dim == tensor->ne[ndim - 1 - i]) << "tensor " << name << " dim " << i << " mismatch: expect "
                                                           << tensor->ne[ndim - 1 - i] << ", got " << dim;
        }
        std::string tensor_name(name_size, 0);
        fin.read(&tensor_name[0], name_size);
        CHATGLM_CHECK(tensor_name == name)
            << "tensor " << name << " name mismatch: expect " << name << ", got " << tensor_name;

        constexpr int64_t MEM_ALIGNED = 16;
        const int64_t data_offset = (fin.tellg() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
        tensor->data = (char *)mapped_file->data + data_offset;
        fin.seekg(data_offset + ggml_nbytes(tensor));
    };

    model = std::make_unique<ChatGLMForConditionalGeneration>(&ctx, config);
    load_tensor("transformer.word_embeddings.weight", model->transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_layers; i++) {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        load_tensor(layer_prefix + "input_layernorm.weight", model->transformer.layers[i].input_layernorm.weight);
        load_tensor(layer_prefix + "input_layernorm.bias", model->transformer.layers[i].input_layernorm.bias);
        load_tensor(layer_prefix + "attention.query_key_value.weight",
                    model->transformer.layers[i].attention.query_key_value.weight);
        load_tensor(layer_prefix + "attention.query_key_value.bias",
                    model->transformer.layers[i].attention.query_key_value.bias);
        load_tensor(layer_prefix + "attention.dense.weight", model->transformer.layers[i].attention.dense.weight);
        load_tensor(layer_prefix + "attention.dense.bias", model->transformer.layers[i].attention.dense.bias);
        model->transformer.layers[i].attention.k_cache->data = kvcache_ptr;
        kvcache_ptr += ggml_nbytes(model->transformer.layers[i].attention.k_cache);
        model->transformer.layers[i].attention.v_cache->data = kvcache_ptr;
        kvcache_ptr += ggml_nbytes(model->transformer.layers[i].attention.v_cache);
        load_tensor(layer_prefix + "post_attention_layernorm.weight",
                    model->transformer.layers[i].post_attention_layernorm.weight);
        load_tensor(layer_prefix + "post_attention_layernorm.bias",
                    model->transformer.layers[i].post_attention_layernorm.bias);
        load_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", model->transformer.layers[i].mlp.dense_h_to_4h.weight);
        load_tensor(layer_prefix + "mlp.dense_h_to_4h.bias", model->transformer.layers[i].mlp.dense_h_to_4h.bias);
        load_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", model->transformer.layers[i].mlp.dense_4h_to_h.weight);
        load_tensor(layer_prefix + "mlp.dense_4h_to_h.bias", model->transformer.layers[i].mlp.dense_4h_to_h.bias);
    }
    load_tensor("transformer.final_layernorm.weight", model->transformer.final_layernorm.weight);
    load_tensor("transformer.final_layernorm.bias", model->transformer.final_layernorm.bias);
    CHATGLM_CHECK(kvcache_ptr == kvcache_buffer.get() + kvcache_size) << "corrupted kv cache";
    CHATGLM_CHECK(ggml_used_mem(ctx.gctx) == ggml_get_mem_size(ctx.gctx)) << "corrupted model weights";

#ifdef GGML_USE_CUBLAS
    for (int i = 0; i < config.num_layers; i++) {
        ggml_cuda_transform_tensor(model->transformer.layers[i].attention.query_key_value.weight);
        ggml_cuda_transform_tensor(model->transformer.layers[i].attention.dense.weight);
        ggml_cuda_transform_tensor(model->transformer.layers[i].mlp.dense_h_to_4h.weight);
        ggml_cuda_transform_tensor(model->transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
#endif
}

ChatGLMPipeline::~ChatGLMPipeline() {
    if (ctx.gctx) {
        ggml_free(ctx.gctx);
        ctx.gctx = nullptr;
    }
}

std::string ChatGLMPipeline::chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                                  BaseStreamer *streamer) const {
    std::string prompt = build_prompt(history);
    std::vector<int> input_ids = tokenizer->encode(prompt);
    if ((int)input_ids.size() > gen_config.max_context_length) {
        // sliding window: always take the last max_context_length tokens
        input_ids.erase(input_ids.begin(), input_ids.end() - gen_config.max_context_length);
    }

    std::vector<int> output_ids = model->generate(input_ids, gen_config, streamer);
    std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());

    std::string output = tokenizer->decode(new_output_ids);
    return output;
}

std::string ChatGLMPipeline::build_prompt(const std::vector<std::string> &history) {
    CHATGLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;
    if (history.size() == 1) {
        oss_prompt << history.front();
    } else {
        for (size_t i = 0; i < history.size(); i += 2) {
            oss_prompt << "[Round " << i / 2 << "]\n问：" << history[i] << "\n答：";
            if (i < history.size() - 1) {
                oss_prompt << history[i + 1] << "\n";
            }
        }
    }
    return oss_prompt.str();
}

} // namespace chatglm
