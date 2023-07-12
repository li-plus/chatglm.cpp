#include "chatglm.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <thread>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>
#include <windows.h>
#endif

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

namespace chatglm {

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
        if (tensor->n_dims > 3)
            oss << "[";
        for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
            if (tensor->n_dims > 2)
                oss << (i3 > 0 ? ",\n\n[" : "[");
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                if (tensor->n_dims > 1)
                    oss << (i2 > 0 ? ",\n\n[" : "[");
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    oss << (i1 > 0 ? ",\n[" : "[");
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
                        oss << (i0 > 0 ? ", " : "") << std::setw(7) << std::fixed << std::setprecision(4) << val;
                    }
                    oss << "]";
                }
                if (tensor->n_dims > 1)
                    oss << "]";
            }
            if (tensor->n_dims > 2)
                oss << "]";
        }
        if (tensor->n_dims > 3)
            oss << "]";
    }

    oss << ", shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
    return oss.str();
}

void tensor_assign_buffers(ggml_tensor *tensor, bool scratch, bool force_inplace) {
#ifdef GGML_USE_CUBLAS
    if (scratch) {
        CHATGLM_CHECK(!force_inplace);
        ggml_cuda_assign_buffers(tensor);
    } else {
        if (force_inplace) {
            ggml_cuda_assign_buffers_force_inplace(tensor);
        } else {
            // BE CAREFUL TO USE THIS!
            ggml_cuda_assign_buffers_no_scratch(tensor);
        }
    }
#endif
}

void tensor_to_device(ggml_tensor *tensor) {
#ifdef GGML_USE_CUBLAS
    if (tensor->backend == GGML_BACKEND_GPU || tensor->backend == GGML_BACKEND_GPU_SPLIT) {
        return;
    }
    tensor->backend = GGML_BACKEND_GPU;
    ggml_cuda_transform_tensor(tensor->data, tensor);
#endif
}

void tensor_to_cpu(ggml_tensor *tensor) {
#ifdef GGML_USE_CUBLAS
    if (tensor->backend == GGML_BACKEND_CPU) {
        return;
    }
    ggml_cuda_free_data(tensor);
    tensor->backend = GGML_BACKEND_CPU;
#endif
}

// ===== streamer =====

void StreamerGroup::put(const std::vector<int> &output_ids) {
    for (auto &streamer : streamers_) {
        streamer->put(output_ids);
    }
}

void StreamerGroup::end() {
    for (auto &streamer : streamers_) {
        streamer->end();
    }
}

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

    os_ << printable_text << std::flush;
}

void TextStreamer::end() {
    std::string text = tokenizer_->decode(token_cache_);
    os_ << text.substr(print_len_) << std::endl;
    is_prompt_ = true;
    token_cache_.clear();
    print_len_ = 0;
}

void PerfStreamer::put(const std::vector<int> &output_ids) {
    CHATGLM_CHECK(!output_ids.empty());
    if (num_prompt_tokens_ == 0) {
        // before prompt eval
        start_us_ = ggml_time_us();
        num_prompt_tokens_ = output_ids.size();
    } else {
        if (num_output_tokens_ == 0) {
            // first new token
            prompt_us_ = ggml_time_us();
        }
        num_output_tokens_ += output_ids.size();
    }
}

void PerfStreamer::reset() {
    start_us_ = prompt_us_ = end_us_ = 0;
    num_prompt_tokens_ = num_output_tokens_ = 0;
}

std::string PerfStreamer::to_string() const {
    std::ostringstream oss;
    oss << "prompt time: " << prompt_total_time_us() / 1000.f << " ms / " << num_prompt_tokens() << " tokens ("
        << prompt_token_time_us() / 1000.f << " ms/token)\n"
        << "output time: " << output_total_time_us() / 1000.f << " ms / " << num_output_tokens() << " tokens ("
        << output_token_time_us() / 1000.f << " ms/token)\n"
        << "total time: " << (prompt_total_time_us() + output_total_time_us()) / 1000.f << " ms";
    return oss.str();
}

#ifdef _POSIX_MAPPED_FILES
MappedFile::MappedFile(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct stat sb;
    CHATGLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    CHATGLM_CHECK(data != MAP_FAILED) << strerror(errno);

    CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { CHATGLM_CHECK(munmap(data, size) == 0) << strerror(errno); }
#elif defined(_WIN32)
MappedFile::MappedFile(const std::string &path) {

    int fd = open(path.c_str(), O_RDONLY);
    CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct _stat64 sb;
    CHATGLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    HANDLE hFile = (HANDLE)_get_osfhandle(fd);

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CHATGLM_CHECK(hMapping != NULL) << strerror(errno);

    data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);

    CHATGLM_CHECK(data != NULL) << strerror(errno);

    CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { CHATGLM_CHECK(UnmapViewOfFile(data)) << strerror(errno); }
#endif

void ModelLoader::seek(int64_t offset, int whence) {
    if (whence == SEEK_SET) {
        ptr = data + offset;
    } else if (whence == SEEK_CUR) {
        ptr += offset;
    } else if (whence == SEEK_END) {
        ptr = data + size + offset;
    } else {
        CHATGLM_THROW << "invalid seek mode " << whence;
    }
}

std::string ModelLoader::read_string(size_t length) {
    std::string s(ptr, ptr + length);
    ptr += length;
    return s;
}

void ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor) {
    // read and check tensor name
    {
        int name_size = read_basic<int>();
        CHATGLM_CHECK(name_size == (int)name.size())
            << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size;
        std::string weight_name = read_string(name_size);
        CHATGLM_CHECK(weight_name == name) << "tensor name mismatch: expect " << name << " but got " << weight_name;
    }

    // read and check tensor shape
    {
        int ndim = read_basic<int>();
        CHATGLM_CHECK(ndim == tensor->n_dims)
            << "tensor " << name << " ndim mismatch: expect " << tensor->n_dims << " but got " << ndim;
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_size = read_basic<int>();
            CHATGLM_CHECK(dim_size == tensor->ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                     << ": expect " << tensor->ne[i] << " but got " << dim_size;
        }
    }

    // read and check tensor dtype
    {
        ggml_type dtype = (ggml_type)read_basic<int>();
        CHATGLM_CHECK(dtype == tensor->type)
            << "tensor " << name << " dtype mismatch: expect " << tensor->type << " but got " << dtype;
    }

    // map tensor data
    {
        constexpr int64_t MEM_ALIGNED = 16;
        const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
        tensor->data = const_cast<char *const>(data) + data_offset;
        seek(data_offset + ggml_nbytes(tensor), SEEK_SET);
    }
}

// ===== modules =====

ggml_tensor *Embedding::forward(ForwardContext *ctx, ggml_tensor *input) const {
    ggml_tensor *output = ggml_get_rows(ctx->gctx.get(), weight, input);
    return output;
}

ggml_tensor *Linear::forward(ForwardContext *ctx, ggml_tensor *input) const {
    // input: [seqlen, in_features]
    ggml_tensor *output = ggml_mul_mat(ctx->gctx.get(), weight, input); // [seqlen, out_features]
    tensor_assign_buffers(output);
    if (bias) {
        output = ggml_add_inplace(ctx->gctx.get(), output, bias);
        tensor_assign_buffers(output, false, true);
    }
    return output;
}

ggml_tensor *LayerNorm::forward(ForwardContext *ctx, ggml_tensor *input) const {
    // input: [seqlen, normalized_shape]
    ggml_tensor *output = ggml_norm_inplace(ctx->gctx.get(), input);
    tensor_assign_buffers(output, false, true);
    output = ggml_mul_inplace(ctx->gctx.get(), output, weight);
    tensor_assign_buffers(output, false, true);
    output = ggml_add_inplace(ctx->gctx.get(), output, bias);
    tensor_assign_buffers(output, false, true);
    return output;
}

ggml_tensor *RMSNorm::forward(ForwardContext *ctx, ggml_tensor *input) const {
    ggml_tensor *output = ggml_rms_norm_inplace(ctx->gctx.get(), input);
    tensor_assign_buffers(output, false, true);
    output = ggml_mul_inplace(ctx->gctx.get(), output, weight);
    tensor_assign_buffers(output, false, true);
    return output;
}

// ===== ChatGLM-6B =====

ChatGLMTokenizer::ChatGLMTokenizer(std::string_view serialized_model_proto) {
    const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
    CHATGLM_CHECK(status.ok()) << status.ToString();

    bos_token_id = sp.PieceToId("<sop>");
    eos_token_id = sp.PieceToId("<eop>");
    mask_token_id = sp.PieceToId("[MASK]");
    gmask_token_id = sp.PieceToId("[gMASK]");
    pad_token_id = sp.PieceToId("<pad>");
}

std::vector<int> ChatGLMTokenizer::encode(const std::string &text) const {
    std::string input = preprocess(text);
    std::vector<int> ids;
    sp.Encode(input, &ids);
    ids.insert(ids.end(), {gmask_token_id, bos_token_id});
    return ids;
}

std::vector<int> ChatGLMTokenizer::encode_history(const std::vector<std::string> &history, int max_length) const {
    std::string prompt = build_prompt(history);
    std::vector<int> input_ids = encode(prompt);
    if ((int)input_ids.size() > max_length) {
        // sliding window: always take the last max_length tokens
        input_ids.erase(input_ids.begin(), input_ids.end() - max_length);
    }
    return input_ids;
}

std::string ChatGLMTokenizer::build_prompt(const std::vector<std::string> &history) {
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

std::string ChatGLMTokenizer::decode(const std::vector<int> &ids) const {
    std::string text;
    sp.Decode(ids, &text);
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

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

std::string to_string(ModelType model_type) {
    switch (model_type) {
    case MODEL_TYPE_CHATGLM:
        return "ChatGLM";
    case MODEL_TYPE_CHATGLM2:
        return "ChatGLM2";
    default:
        CHATGLM_THROW << "unknown model type " << model_type;
    }
}

int BaseModelForConditionalGeneration::generate_next_token(const std::vector<int> &input_ids,
                                                           const GenerationConfig &gen_config, int n_past,
                                                           int n_ctx) const {
    ForwardContext ctx;
    ctx.gctx = GGMLContext(mem_size_, mem_buffer_.get(), false);
    ctx.gf = {};
    ctx.scratch = {0, scratch_size_, scratch_buffer_.get()};

    int n_threads = gen_config.num_threads > 0 ? gen_config.num_threads : get_num_physical_cores();
    if (input_ids.size() >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
        n_threads = 1; // BLAS enabled
    }

    ggml_tensor *input_ids_tensor = ggml_new_tensor_1d(ctx.gctx.get(), GGML_TYPE_I32, input_ids.size());
    memcpy(input_ids_tensor->data, input_ids.data(), ggml_nbytes(input_ids_tensor));

    ggml_tensor *lm_logits = forward(&ctx, input_ids_tensor, n_past, n_ctx);

    ggml_build_forward_expand(&ctx.gf, lm_logits);
    // TODO: upgrade to ggml_graph_compute with cplan
    ggml_graph_compute_with_ctx(ctx.gctx.get(), &ctx.gf, n_threads);

#ifdef GGML_PERF
    ggml_graph_print(&ctx.gf);
#endif

    int vocab_size = lm_logits->ne[0];
    float *next_token_logits = (float *)lm_logits->data;

    int next_token_id;
    if (gen_config.do_sample) {
        // temperature sampling
        float inv_temp = 1.f / gen_config.temperature;
        for (int i = 0; i < vocab_size; i++) {
            next_token_logits[i] *= inv_temp;
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, next_token_logits[i]);
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
        next_token_id = std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
    }

    return next_token_id;
}

void BaseModelForConditionalGeneration::sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) {
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

std::vector<int> BaseModelForConditionalGeneration::generate(const std::vector<int> &input_ids,
                                                             const GenerationConfig &gen_config,
                                                             BaseStreamer *streamer) const {
    CHATGLM_CHECK(gen_config.max_length <= config_.max_length)
        << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
        << config_.max_length << ")";

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

        if (next_token_id == config_.eos_token_id) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return output_ids;
}

ggml_tensor *GLMMLP::forward(ForwardContext *ctx, ggml_tensor *hidden_states) const {
    ggml_tensor *output = dense_h_to_4h.forward(ctx, hidden_states);
    output = ggml_gelu_inplace(ctx->gctx.get(), output);
    tensor_assign_buffers(output, false, true);
    output = dense_4h_to_h.forward(ctx, output);
    return output;
}

ggml_tensor *GLMSelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const {
    int hidden_size = hidden_states->ne[0];
    int qlen = hidden_states->ne[1];
    int head_size = hidden_size / num_attention_heads;
    int rope_dim = head_size / 2;

    ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, 3 * hidden]

    ggml_tensor *query_layer = ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen,
                                            3 * head_size * ggml_element_size(qkv), qkv->nb[1], 0);
    // TODO: MODE!!!!!!! 4!!!
    query_layer =
        ggml_rope_inplace(ctx->gctx.get(), query_layer, n_past, rope_dim, 0, n_ctx); // [qlen, heads, head_size]
    tensor_assign_buffers(query_layer, false, true);
    query_layer = ggml_permute(ctx->gctx.get(), query_layer, 0, 2, 1, 3); // [heads, qlen, head_size]

    ggml_tensor *key_layer =
        ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                     qkv->nb[1], head_size * ggml_element_size(qkv));
    key_layer = ggml_rope_inplace(ctx->gctx.get(), key_layer, n_past, rope_dim, 0, n_ctx); // [qlen, heads, head_size]
    tensor_assign_buffers(key_layer, false, true);
    key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3); // [heads, qlen, head_size]

    ggml_tensor *value_layer = ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen,
                                            3 * head_size * ggml_element_size(qkv), qkv->nb[1],
                                            2 * head_size * ggml_element_size(qkv)); // [qlen, heads, head_size]
    value_layer = ggml_permute(ctx->gctx.get(), value_layer, 1, 2, 0, 3);            // [heads, head_size, qlen]

    // store key & value to cache
    ggml_tensor *k_cache_view =
        ggml_view_3d(ctx->gctx.get(), k_cache, head_size, qlen, num_attention_heads, k_cache->nb[1], k_cache->nb[2],
                     n_past * head_size * ggml_element_size(k_cache)); // [heads, qlen, head_size]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx.get(), key_layer, k_cache_view));
    ggml_tensor *v_cache_view =
        ggml_view_3d(ctx->gctx.get(), v_cache, qlen, head_size, num_attention_heads, v_cache->nb[1], v_cache->nb[2],
                     n_past * ggml_element_size(v_cache)); // [heads, head_size, qlen]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx.get(), value_layer, v_cache_view));

    key_layer = ggml_view_3d(ctx->gctx.get(), k_cache, head_size, n_past + qlen, num_attention_heads, k_cache->nb[1],
                             k_cache->nb[2], 0); // [heads, klen, head_size]
    value_layer = ggml_view_3d(ctx->gctx.get(), v_cache, n_past + qlen, head_size, num_attention_heads, v_cache->nb[1],
                               v_cache->nb[2], 0); // [heads, head_size, klen]

    ggml_tensor *attn_scores = ggml_mul_mat(ctx->gctx.get(), key_layer, query_layer); // [heads, qlen, klen]
    tensor_assign_buffers(attn_scores);
    if (n_past == 0) {
        // build attention mask for context input
        ggml_tensor *inf = ggml_new_tensor_3d(ctx->gctx.get(), attn_scores->type, 1, qlen - 1, num_attention_heads);
        ggml_set_f32(inf, -INFINITY);
        tensor_to_device(inf);
        ggml_tensor *masked_attn_scores = ggml_view_3d(
            ctx->gctx.get(), attn_scores, 1, qlen - 1, num_attention_heads, qlen * ggml_element_size(attn_scores),
            qlen * qlen * ggml_element_size(attn_scores), (qlen - 1) * ggml_element_size(attn_scores));
        tensor_assign_buffers(masked_attn_scores, false, true);
        ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx.get(), inf, masked_attn_scores));
    }
    attn_scores =
        ggml_scale_inplace(ctx->gctx.get(), attn_scores, ggml_new_f32(ctx->gctx.get(), 1.f / std::sqrt(head_size)));
    tensor_assign_buffers(attn_scores, false, true);
    ggml_tensor *attn_probs = ggml_soft_max_inplace(ctx->gctx.get(), attn_scores); // [heads, qlen, klen]
    tensor_assign_buffers(attn_probs, false, true);

    ggml_tensor *context_layer = ggml_mul_mat(ctx->gctx.get(), value_layer, attn_probs); // [heads, qlen, head_size]
    tensor_assign_buffers(context_layer);
    context_layer = ggml_reshape_2d(
        ctx->gctx.get(), ggml_cont(ctx->gctx.get(), ggml_permute(ctx->gctx.get(), context_layer, 0, 2, 1, 3)),
        hidden_size, qlen);
    tensor_assign_buffers(context_layer);

    ggml_tensor *attn_output = dense.forward(ctx, context_layer);
    return attn_output;
}

ggml_tensor *GLMBlock::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past, int n_ctx) const {
    ggml_tensor *alpha = ggml_new_f32(ctx->gctx.get(), std::sqrt(2.f * num_hidden_layers));
    tensor_assign_buffers(alpha);

    ggml_tensor *attn_input = input_layernorm.forward(ctx, hidden_states);
    ggml_tensor *attn_output = attention.forward(ctx, attn_input, n_past, n_ctx);
    ggml_build_forward_expand(&ctx->gf, attn_output);
    attn_input = ggml_scale_inplace(ctx->gctx.get(), attn_input, alpha);
    tensor_assign_buffers(attn_input, false, true);
    hidden_states = ggml_add_inplace(ctx->gctx.get(), attn_input, attn_output);
    tensor_assign_buffers(hidden_states, false, true);

    ggml_tensor *mlp_input = post_attention_layernorm.forward(ctx, hidden_states);
    ggml_tensor *mlp_output = mlp.forward(ctx, mlp_input);
    ggml_build_forward_expand(&ctx->gf, mlp_output);
    mlp_input = ggml_scale_inplace(ctx->gctx.get(), mlp_input, alpha);
    tensor_assign_buffers(mlp_input, false, true);
    ggml_tensor *output = ggml_add_inplace(ctx->gctx.get(), mlp_input, mlp_output);
    tensor_assign_buffers(output, false, true);

    return output;
}

ChatGLMModel::ChatGLMModel(InitContext *ctx, const ChatGLMConfig &config)
    : word_embeddings(ctx, config.vocab_size, config.hidden_size), final_layernorm(ctx, config.hidden_size) {
    layers.reserve(config.num_hidden_layers);
    for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
        layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
                            config.max_length);
    }
}

ggml_tensor *ChatGLMModel::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const {
    ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
    for (const GLMBlock &layer : layers) {
        ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
        hidden_states = layer.forward(ctx, hidden_states, n_past, n_ctx);
    }
    ggml_scratch empty_scratch = {0, 0, nullptr};
    ggml_set_scratch(ctx->gctx.get(), empty_scratch);
    hidden_states = final_layernorm.forward(ctx, hidden_states);
    return hidden_states;
}

ChatGLMForConditionalGeneration::ChatGLMForConditionalGeneration(const ChatGLMConfig &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_CHATGLM, config, MEM_SIZE, SCRATCH_SIZE), config(config) {
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 14;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext(ctx_size, nullptr, true);
    w_ctx_.dtype = config.dtype;

    transformer = ChatGLMModel(&w_ctx_, config);

    const size_t kv_cache_size =
        config.num_hidden_layers * 2 * config.max_length * config.hidden_size * ggml_type_size(GGML_TYPE_F16);
    kv_cache_buffer_.reset(new char[kv_cache_size]);
    char *kv_cache_ptr = kv_cache_buffer_.get();
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        transformer.layers[i].attention.k_cache->data = kv_cache_ptr;
        kv_cache_ptr += ggml_nbytes(transformer.layers[i].attention.k_cache);
        transformer.layers[i].attention.v_cache->data = kv_cache_ptr;
        kv_cache_ptr += ggml_nbytes(transformer.layers[i].attention.v_cache);
    }
    CHATGLM_CHECK(kv_cache_ptr == kv_cache_buffer_.get() + kv_cache_size) << "corrupted kv cache";
}

void ChatGLMForConditionalGeneration::load(ModelLoader &loader) {
    loader.read_tensor("transformer.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "input_layernorm.bias", transformer.layers[i].input_layernorm.bias);
        loader.read_tensor(layer_prefix + "attention.query_key_value.weight",
                           transformer.layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "attention.query_key_value.bias",
                           transformer.layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "attention.dense.weight", transformer.layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "attention.dense.bias", transformer.layers[i].attention.dense.bias);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                           transformer.layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.bias",
                           transformer.layers[i].post_attention_layernorm.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.dense_h_to_4h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.bias", transformer.layers[i].mlp.dense_h_to_4h.bias);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.dense_4h_to_h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.bias", transformer.layers[i].mlp.dense_4h_to_h.bias);
    }
    loader.read_tensor("transformer.final_layernorm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("transformer.final_layernorm.bias", transformer.final_layernorm.bias);
    CHATGLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";

    for (int i = 0; i < config.num_hidden_layers; i++) {
        tensor_to_device(transformer.layers[i].attention.query_key_value.weight);
        tensor_to_device(transformer.layers[i].attention.dense.weight);
        tensor_to_device(transformer.layers[i].mlp.dense_h_to_4h.weight);
        tensor_to_device(transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
}

ggml_tensor *ChatGLMForConditionalGeneration::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past,
                                                      int n_ctx) const {
    ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, n_past, n_ctx);
    // NOTE: only compute next_token_logits for the last token
    if (input_ids->ne[0] > 1) {
        transformer_outputs =
            ggml_view_1d(ctx->gctx.get(), transformer_outputs, config.hidden_size,
                         (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
    }
    ggml_tensor *lm_head_weight = transformer.word_embeddings.weight; // tied weight
    ggml_tensor *lm_logits = ggml_mul_mat(ctx->gctx.get(), lm_head_weight, transformer_outputs);
    return lm_logits;
}

// ===== ChatGLM2-6B =====

ChatGLM2Tokenizer::ChatGLM2Tokenizer(std::string_view serialized_model_proto) {
    const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
    CHATGLM_CHECK(status.ok()) << status.ToString();

    int special_id = sp.GetPieceSize();
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;
}

std::vector<int> ChatGLM2Tokenizer::encode(const std::string &text) const {
    std::vector<int> ids;
    sp.Encode(text, &ids);
    ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    return ids;
}

std::string ChatGLM2Tokenizer::decode(const std::vector<int> &ids) const {
    // filter out special tokens
    std::vector<int> normal_ids(ids);
    normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                     normal_ids.end());

    std::string text;
    sp.Decode(normal_ids, &text);
    return text;
}

std::vector<int> ChatGLM2Tokenizer::encode_history(const std::vector<std::string> &history, int max_length) const {
    std::string prompt = build_prompt(history);
    std::vector<int> input_ids = encode(prompt);
    if ((int)input_ids.size() > max_length) {
        // sliding window: drop the least recent history while keeping the special prefix tokens
        int num_drop = (int)input_ids.size() - max_length;
        input_ids.erase(input_ids.begin() + 2, input_ids.begin() + 2 + num_drop);
    }
    return input_ids;
}

std::string ChatGLM2Tokenizer::build_prompt(const std::vector<std::string> &history) {
    CHATGLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;
    for (size_t i = 0; i < history.size(); i += 2) {
        oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << history[i] << "\n\n答：";
        if (i < history.size() - 1) {
            oss_prompt << history[i + 1] << "\n\n";
        }
    }
    return oss_prompt.str();
}

bool ChatGLM2Tokenizer::is_special_id(int id) const {
    return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
           id == eop_token_id;
}

ggml_tensor *GLM2SelfAttention::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) const {
    const int hidden_size = hidden_states->ne[0];
    const int qlen = hidden_states->ne[1];
    const int head_size = hidden_size / num_attention_heads;
    const int rope_dim = head_size / 2;
    const int mqa_scale = num_attention_heads / num_kv_heads;

    ggml_tensor *qkv = query_key_value.forward(ctx, hidden_states); // [qlen, hidden + 2 * kv_hidden]

    ggml_tensor *query_layer =
        ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv),
                     qkv->nb[1], 0); // [qlen, heads, head_size]
    query_layer = ggml_rope_inplace(ctx->gctx.get(), query_layer, n_past, rope_dim, 0, 0);
    query_layer = ggml_view_4d(ctx->gctx.get(), query_layer, head_size, mqa_scale, num_kv_heads, qlen,
                               query_layer->nb[1], query_layer->nb[1] * mqa_scale, query_layer->nb[2],
                               0);                                        // [qlen, kv_heads, mqa_scale, head_size]
    query_layer = ggml_permute(ctx->gctx.get(), query_layer, 0, 2, 3, 1); // [kv_heads, mqa_scale, qlen, head_size]

    ggml_tensor *key_layer =
        ggml_view_3d(ctx->gctx.get(), qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv),
                     qkv->nb[1], hidden_size * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]
    key_layer = ggml_rope_inplace(ctx->gctx.get(), key_layer, n_past, rope_dim, 0, 0);
    key_layer = ggml_permute(ctx->gctx.get(), key_layer, 0, 2, 1, 3); // [kv_heads, qlen, head_size]

    ggml_tensor *value_layer = ggml_view_3d(
        ctx->gctx.get(), qkv, head_size, num_kv_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
        (hidden_size + head_size * num_kv_heads) * ggml_element_size(qkv)); // [qlen, kv_heads, head_size]
    value_layer = ggml_permute(ctx->gctx.get(), value_layer, 1, 2, 0, 3);   // [kv_heads, head_size, qlen]

    // store key & value to cache
    ggml_tensor *k_cache_view =
        ggml_view_3d(ctx->gctx.get(), k_cache, head_size, qlen, num_kv_heads, k_cache->nb[1], k_cache->nb[2],
                     n_past * head_size * ggml_element_size(k_cache)); // [kv_heads, qlen, head_size]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx.get(), key_layer, k_cache_view));
    ggml_tensor *v_cache_view =
        ggml_view_3d(ctx->gctx.get(), v_cache, qlen, head_size, num_kv_heads, v_cache->nb[1], v_cache->nb[2],
                     n_past * ggml_element_size(v_cache)); // [kv_heads, head_size, qlen]
    ggml_build_forward_expand(&ctx->gf, ggml_cpy(ctx->gctx.get(), value_layer, v_cache_view));

    // concat key & value with past kv
    key_layer = ggml_view_4d(ctx->gctx.get(), k_cache, head_size, n_past + qlen, mqa_scale, num_kv_heads,
                             k_cache->nb[1], 0, k_cache->nb[2],
                             0); // [kv_heads, mqa_scale, klen, head_size]
    value_layer = ggml_view_4d(ctx->gctx.get(), v_cache, n_past + qlen, head_size, mqa_scale, num_kv_heads,
                               v_cache->nb[1], 0, v_cache->nb[2],
                               0); // [kv_heads, mqa_scale, head_size, klen]

    // flash attention
    ggml_tensor *context_layer = ggml_flash_attn(ctx->gctx.get(), query_layer, key_layer, value_layer,
                                                 true); // [mqa_scale, kv_heads, qlen, head_size]
    context_layer = ggml_reshape_2d(
        ctx->gctx.get(), ggml_cont(ctx->gctx.get(), ggml_permute(ctx->gctx.get(), context_layer, 0, 3, 1, 2)),
        hidden_size, qlen); // [qlen, hidden]

    ggml_tensor *attn_output = dense.forward(ctx, context_layer);
    return attn_output;
}

ggml_tensor *GLM2MLP::forward(ForwardContext *ctx, ggml_tensor *hidden_states) const {
    ggml_tensor *output = dense_h_to_4h.forward(ctx, hidden_states);

    // swiglu activation
    ggml_tensor *x0 = ggml_view_2d(ctx->gctx.get(), output, output->ne[0] / 2, output->ne[1], output->nb[1], 0);
    ggml_tensor *x1 = ggml_view_2d(ctx->gctx.get(), output, output->ne[0] / 2, output->ne[1], output->nb[1],
                                   output->ne[0] / 2 * ggml_element_size(output));
    output = ggml_mul_inplace(ctx->gctx.get(), ggml_silu_inplace(ctx->gctx.get(), ggml_cont(ctx->gctx.get(), x0)), x1);

    output = dense_4h_to_h.forward(ctx, output);
    return output;
}

ggml_tensor *GLM2Block::forward(ForwardContext *ctx, ggml_tensor *hidden_states, int n_past) const {
    ggml_tensor *residual = ggml_dup(ctx->gctx.get(), hidden_states);
    ggml_build_forward_expand(&ctx->gf, residual);

    hidden_states = input_layernorm.forward(ctx, hidden_states);
    hidden_states = attention.forward(ctx, hidden_states, n_past);
    hidden_states = ggml_add_inplace(ctx->gctx.get(), hidden_states, residual);

    residual = ggml_dup(ctx->gctx.get(), hidden_states);
    ggml_build_forward_expand(&ctx->gf, residual);

    hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
    hidden_states = mlp.forward(ctx, hidden_states);
    hidden_states = ggml_add_inplace(ctx->gctx.get(), hidden_states, residual);

    return hidden_states;
}

ChatGLM2Model::ChatGLM2Model(InitContext *ctx, const ChatGLM2Config &config)
    : word_embeddings(ctx, config.vocab_size, config.hidden_size), final_layernorm(ctx, config.hidden_size) {
    // TODO: reduce max length? 32k might be too large for cpu inference
    layers.reserve(config.num_hidden_layers);
    for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
        layers.emplace_back(ctx, config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                            config.intermediate_size, config.max_length);
    }
}

ggml_tensor *ChatGLM2Model::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past) const {
    ggml_tensor *hidden_states = word_embeddings.forward(ctx, input_ids);
    for (const auto &layer : layers) {
        ggml_set_scratch(ctx->gctx.get(), ctx->scratch);
        hidden_states = layer.forward(ctx, hidden_states, n_past);
    }
    ggml_scratch empty_scratch = {0, 0, nullptr};
    ggml_set_scratch(ctx->gctx.get(), empty_scratch);
    hidden_states = final_layernorm.forward(ctx, hidden_states);
    return hidden_states;
}

ChatGLM2ForConditionalGeneration::ChatGLM2ForConditionalGeneration(const ChatGLM2Config &config)
    : BaseModelForConditionalGeneration(MODEL_TYPE_CHATGLM2, config, MEM_SIZE, SCRATCH_SIZE), config(config) {
    constexpr size_t tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
    const size_t num_tensors = 3 + config.num_hidden_layers * 9;
    const size_t ctx_size = num_tensors * tensor_ovhd;
    w_ctx_.gctx = GGMLContext(ctx_size, nullptr, true);
    w_ctx_.dtype = config.dtype;

    transformer = ChatGLM2Model(&w_ctx_, config);
    lm_head = Linear(&w_ctx_, config.hidden_size, config.vocab_size, false);

    const size_t kv_cache_size = config.num_hidden_layers * 2ull * config.max_length * config.hidden_size /
                                 config.num_attention_heads * config.num_kv_heads * ggml_type_size(GGML_TYPE_F32);
    kv_cache_buffer_.reset(new char[kv_cache_size]);

    char *kv_cache_ptr = kv_cache_buffer_.get();
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        transformer.layers[i].attention.k_cache->data = kv_cache_ptr;
        kv_cache_ptr += ggml_nbytes(transformer.layers[i].attention.k_cache);
        transformer.layers[i].attention.v_cache->data = kv_cache_ptr;
        kv_cache_ptr += ggml_nbytes(transformer.layers[i].attention.v_cache);
    }
    CHATGLM_CHECK(kv_cache_ptr == kv_cache_buffer_.get() + kv_cache_size) << "corrupted kv cache";
}

void ChatGLM2ForConditionalGeneration::load(ModelLoader &loader) {
    loader.read_tensor("transformer.embedding.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.encoder.layers." + std::to_string(i) + '.';
        loader.read_tensor(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.weight",
                           transformer.layers[i].attention.query_key_value.weight);
        loader.read_tensor(layer_prefix + "self_attention.query_key_value.bias",
                           transformer.layers[i].attention.query_key_value.bias);
        loader.read_tensor(layer_prefix + "self_attention.dense.weight", transformer.layers[i].attention.dense.weight);
        loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",
                           transformer.layers[i].post_attention_layernorm.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.dense_h_to_4h.weight);
        loader.read_tensor(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
    loader.read_tensor("transformer.encoder.final_layernorm.weight", transformer.final_layernorm.weight);
    loader.read_tensor("transformer.output_layer.weight", lm_head.weight);
    CHATGLM_CHECK(ggml_used_mem(w_ctx_.gctx.get()) == ggml_get_mem_size(w_ctx_.gctx.get()))
        << "corrupted model weights";

    for (int i = 0; i < config.num_hidden_layers; i++) {
        tensor_to_device(transformer.layers[i].attention.query_key_value.weight);
        tensor_to_device(transformer.layers[i].attention.dense.weight);
        tensor_to_device(transformer.layers[i].mlp.dense_h_to_4h.weight);
        tensor_to_device(transformer.layers[i].mlp.dense_4h_to_h.weight);
    }
    tensor_to_device(lm_head.weight);
}

ggml_tensor *ChatGLM2ForConditionalGeneration::forward(ForwardContext *ctx, ggml_tensor *input_ids, int n_past,
                                                       int n_ctx) const {
    ggml_tensor *transformer_outputs = transformer.forward(ctx, input_ids, n_past);
    // NOTE: only compute next_token_logits for the last token
    if (input_ids->ne[0] > 1) {
        transformer_outputs =
            ggml_view_1d(ctx->gctx.get(), transformer_outputs, config.hidden_size,
                         (input_ids->ne[0] - 1) * config.hidden_size * ggml_element_size(transformer_outputs));
    }
    ggml_tensor *lm_logits = lm_head.forward(ctx, transformer_outputs);
    return lm_logits;
}

// ===== pipeline =====

Pipeline::Pipeline(const std::string &path) {
    mapped_file = std::make_unique<MappedFile>(path);
    ModelLoader loader(std::string_view((char *)mapped_file->data, mapped_file->size));

    // load magic
    std::string magic = loader.read_string(4);
    CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

    // load model type
    ModelType model_type = (ModelType)loader.read_basic<int>();
    if (model_type == MODEL_TYPE_CHATGLM) {
        // load version
        int version = loader.read_basic<int>();
        CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

        // load config
        ChatGLMConfig config = loader.read_basic<ChatGLMConfig>();

        // load tokenizer
        int proto_size = loader.read_basic<int>();
        std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
        loader.seek(proto_size, SEEK_CUR);
        tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto);

        // load model
        model = std::make_unique<ChatGLMForConditionalGeneration>(config);
        model->load(loader);
    } else if (model_type == MODEL_TYPE_CHATGLM2) {
        // load version
        int version = loader.read_basic<int>();
        CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

        // load config
        ChatGLM2Config config = loader.read_basic<ChatGLM2Config>();

        // load tokenizer
        int proto_size = loader.read_basic<int>();
        std::string_view serialized_model_proto((char *)mapped_file->data + loader.tell(), proto_size);
        loader.seek(proto_size, SEEK_CUR);
        tokenizer = std::make_unique<ChatGLM2Tokenizer>(serialized_model_proto);

        // load model
        model = std::make_unique<ChatGLM2ForConditionalGeneration>(config);
        model->load(loader);
    } else {
        CHATGLM_THROW << "invalid model type " << model_type;
    }
}

std::string Pipeline::chat(const std::vector<std::string> &history, const GenerationConfig &gen_config,
                           BaseStreamer *streamer) const {
    std::vector<int> input_ids = tokenizer->encode_history(history, gen_config.max_context_length);
    std::vector<int> output_ids = model->generate(input_ids, gen_config, streamer);

    std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
    std::string output = tokenizer->decode(new_output_ids);
    return output;
}

} // namespace chatglm
