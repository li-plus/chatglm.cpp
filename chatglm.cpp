#include "chatglm.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <ggml-quants.h>
#include <google/protobuf/stubs/strutil.h>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

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

#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

namespace chatglm {

static std::string shape_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; i--) {
        oss << tensor->ne[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

static std::string strides_to_string(ggml_tensor *tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; i--) {
        oss << tensor->nb[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

std::string to_string(ggml_tensor *tensor, bool with_data) {
    std::vector<char> buf(ggml_nbytes(tensor));
    if (tensor->buffer) {
        ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size());
    } else {
        memcpy(buf.data(), tensor->data, buf.size());
    }

    std::vector<float> float_buf(ggml_nelements(tensor));

    switch (tensor->type) {
    case GGML_TYPE_F32:
        memcpy(float_buf.data(), buf.data(), buf.size());
        break;
    case GGML_TYPE_F16:
        ggml_fp16_to_fp32_row((ggml_fp16_t *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    case GGML_TYPE_Q4_0:
        dequantize_row_q4_0((block_q4_0 *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    case GGML_TYPE_Q4_1:
        dequantize_row_q4_1((block_q4_1 *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    case GGML_TYPE_Q5_0:
        dequantize_row_q5_0((block_q5_0 *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    case GGML_TYPE_Q5_1:
        dequantize_row_q5_1((block_q5_1 *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    case GGML_TYPE_Q8_0:
        dequantize_row_q8_0((block_q8_0 *)buf.data(), float_buf.data(), ggml_nelements(tensor));
        break;
    default:
        CHATGLM_THROW << "Unsupported dtype " << tensor->type;
    }

    std::ostringstream oss;
    oss << "ggml_tensor(";

    if (with_data) {
        const int n_dims = ggml_n_dims(tensor);
        if (n_dims > 3)
            oss << "[";
        for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
            if (n_dims > 2)
                oss << (i3 > 0 ? ",\n\n[" : "[");
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                if (n_dims > 1)
                    oss << (i2 > 0 ? ",\n\n[" : "[");
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    oss << (i1 > 0 ? ",\n[" : "[");
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        oss << (i0 > 0 ? ", " : "");
                        const int i = ((i3 * tensor->ne[2] + i2) * tensor->ne[1] + i1) * tensor->ne[0] + i0;
                        oss << std::setw(7) << std::fixed << std::setprecision(4) << float_buf[i];
                    }
                    oss << "]";
                }
                if (n_dims > 1)
                    oss << "]";
            }
            if (n_dims > 2)
                oss << "]";
        }
        if (n_dims > 3)
            oss << "]";
        oss << ", ";
    }

    oss << "shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
    return oss.str();
}

const std::string ToolCallMessage::TYPE_FUNCTION = "function";
const std::string ToolCallMessage::TYPE_CODE = "code";

const std::string ChatMessage::ROLE_USER = "user";
const std::string ChatMessage::ROLE_ASSISTANT = "assistant";
const std::string ChatMessage::ROLE_SYSTEM = "system";
const std::string ChatMessage::ROLE_OBSERVATION = "observation";

void BaseTokenizer::check_chat_messages(const std::vector<ChatMessage> &messages) {
    std::string target_role = ChatMessage::ROLE_USER;
    for (size_t i = 0; i < messages.size(); i++) {
        if (messages[i].role != ChatMessage::ROLE_USER && messages[i].role != ChatMessage::ROLE_ASSISTANT) {
            continue;
        }
        CHATGLM_CHECK(messages[i].role == target_role)
            << "expect messages[" << i << "].role to be " << target_role << ", but got " << messages[i].role;
        target_role = (target_role == ChatMessage::ROLE_USER) ? ChatMessage::ROLE_ASSISTANT : ChatMessage::ROLE_USER;
    }
    CHATGLM_CHECK(target_role == ChatMessage::ROLE_ASSISTANT)
        << "expect last message role to be " << ChatMessage::ROLE_USER << ", but got " << ChatMessage::ROLE_ASSISTANT;
}

std::vector<ChatMessage> BaseTokenizer::filter_user_assistant_messages(const std::vector<ChatMessage> &messages) {
    std::vector<ChatMessage> user_assistant_messages;
    user_assistant_messages.reserve(messages.size());
    for (const auto &msg : messages) {
        if (msg.role == ChatMessage::ROLE_USER || msg.role == ChatMessage::ROLE_ASSISTANT) {
            user_assistant_messages.emplace_back(msg);
        }
    }
    return user_assistant_messages;
}

// for debugging purpose
[[maybe_unused]] static inline ggml_tensor *add_zero(ggml_context *ctx, ggml_tensor *tensor) {
    ggml_tensor *zeros = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(tensor), tensor->ne);
    ggml_set_f32(zeros, 0);
    ggml_tensor *out = ggml_add(ctx, tensor, zeros);
    return out;
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

// reference: https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
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
    if (is_first_line_) {
        ltrim(text);
    }
    if (text.empty()) {
        return;
    }

    std::string printable_text;
    if (text.back() == '\n') {
        // flush the cache after newline
        printable_text = text.substr(print_len_);
        is_first_line_ = false;
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
    if (is_first_line_) {
        ltrim(text);
    }
    os_ << text.substr(print_len_) << std::endl;
    is_prompt_ = true;
    is_first_line_ = true;
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

StateDict ModelLoader::read_state_dict() {
    StateDict sd;
    sd.ctx = make_unique_ggml_context(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead(), nullptr, true);
    sd.buf = unique_ggml_backend_buffer_t(ggml_backend_cpu_buffer_from_ptr(data, size));

    // assume state dict is stored at the back of file
    while (tell() < (int64_t)size) {
        // tensor name
        int name_size = read_basic<int>();
        std::string weight_name = read_string(name_size);

        // tensor shape
        int64_t ne[4]{1, 1, 1, 1};
        int ndim = read_basic<int>();
        CHATGLM_CHECK(0 < ndim && ndim <= 4);
        for (int i = ndim - 1; i >= 0; i--) {
            ne[i] = read_basic<int>();
        }

        // tensor dtype
        ggml_type dtype = (ggml_type)read_basic<int>();

        // tensor data
        ggml_tensor *tensor = ggml_new_tensor(sd.ctx.get(), dtype, ndim, ne);
        constexpr int64_t MEM_ALIGNED = 16;
        const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
        ggml_backend_tensor_alloc(sd.buf.get(), tensor, data + data_offset);
        // tensor->data = data + data_offset;
        seek(data_offset + ggml_nbytes(tensor), SEEK_SET);

        // add to state dict
        sd.kv.emplace(weight_name, tensor);
    }
    return sd;
}

Image Image::open(const std::string &path) {
    int width, height, channels;
    uint8_t *buffer = stbi_load(path.c_str(), &width, &height, &channels, 3);
    CHATGLM_CHECK(channels == 3);
    Image image(width, height, channels, buffer);
    stbi_image_free(buffer);
    return image;
}

Image Image::resize(size_t new_width, size_t new_height) const {
    Image output(new_width, new_height, channels);
    stbir_resize_uint8_srgb(pixels.data(), width, height, 0, output.pixels.data(), new_width, new_height, 0, STBIR_RGB);
    return output;
}

ModelContext::ModelContext()
    : compute_meta(ggml_tensor_overhead() * (GGML_DEFAULT_GRAPH_SIZE * 4) + ggml_graph_overhead()),
      ctx_w(make_unique_ggml_context(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE, nullptr, true)),
      ctx_kv(make_unique_ggml_context(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE, nullptr, true)),
      ctx_b(make_unique_ggml_context(compute_meta.size(), compute_meta.data(), true)),
      gf(ggml_new_graph_custom(ctx_b.get(), GGML_DEFAULT_GRAPH_SIZE * 4, false)) {

#if defined(GGML_USE_CUDA)
    backend = unique_ggml_backend_t(ggml_backend_cuda_init(0));
#elif defined(GGML_USE_METAL)
    backend = unique_ggml_backend_t(ggml_backend_metal_init());
#else
    backend = unique_ggml_backend_t(ggml_backend_cpu_init());
#endif
    CHATGLM_CHECK(backend) << "failed to initialize ggml backend";

    allocr = unique_ggml_gallocr_t(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
}

// ===== modules =====

ggml_tensor *Embedding::forward(ModelContext *mctx, ggml_tensor *input) const {
    ggml_tensor *output = ggml_get_rows(mctx->ctx_b.get(), weight, input);
    return output;
}

ggml_tensor *Linear::forward(ModelContext *mctx, ggml_tensor *input) const {
    // input: [seqlen, in_features]
    ggml_context *ctx = mctx->ctx_b.get();
    ggml_tensor *output = ggml_mul_mat(ctx, weight, input); // [seqlen, out_features]
    if (bias) {
        output = ggml_add_inplace(ctx, output, bias);
    }
    return output;
}

ggml_tensor *LayerNorm::forward(ModelContext *mctx, ggml_tensor *input) const {
    // input: [seqlen, normalized_shape]
    ggml_context *ctx = mctx->ctx_b.get();
    ggml_tensor *output = ggml_norm(ctx, input, eps);
    output = ggml_mul_inplace(ctx, output, weight);
    output = ggml_add_inplace(ctx, output, bias);
    return output;
}

ggml_tensor *RMSNorm::forward(ModelContext *mctx, ggml_tensor *input) const {
    ggml_context *ctx = mctx->ctx_b.get();
    ggml_tensor *output = ggml_rms_norm(ctx, input, eps);
    output = ggml_mul_inplace(ctx, output, weight);
    return output;
}

static ggml_tensor *apply_activation_inplace(ggml_context *ctx, ggml_tensor *hidden_states, ActivationType hidden_act) {
    switch (hidden_act) {
    case ActivationType::GELU:
        return ggml_gelu_inplace(ctx, hidden_states);
    case ActivationType::SILU:
        return ggml_silu_inplace(ctx, hidden_states);
    default:
        CHATGLM_THROW << "Unknown activation type " << (int)hidden_act;
    }
}

ggml_tensor *BasicMLP::forward(ModelContext *mctx, ggml_tensor *hidden_states) const {
    ggml_context *ctx = mctx->ctx_b.get();
    hidden_states = dense_h_to_4h.forward(mctx, hidden_states);
    hidden_states = apply_activation_inplace(ctx, hidden_states, hidden_act);
    hidden_states = dense_4h_to_h.forward(mctx, hidden_states);
    return hidden_states;
}

ggml_tensor *BasicGLU::forward(ModelContext *mctx, ggml_tensor *hidden_states) const {
    ggml_context *ctx = mctx->ctx_b.get();
    ggml_tensor *gate = gate_proj.forward(mctx, hidden_states);
    gate = apply_activation_inplace(ctx, gate, hidden_act);
    hidden_states = up_proj.forward(mctx, hidden_states);
    hidden_states = ggml_mul_inplace(ctx, hidden_states, gate);
    hidden_states = down_proj.forward(mctx, hidden_states);
    return hidden_states;
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/common/common.cpp
static int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

static void set_default_num_threads(ggml_backend_t backend, int num_tokens) {
    int n_threads = 1;
    if (ggml_backend_is_cpu(backend)) {
        if (num_tokens > 1) {
            // context
            n_threads = get_num_physical_cores();
        } else {
            // decode
            n_threads = std::min(get_num_physical_cores(), 16);
        }
    }
    if (num_tokens >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas()) {
        // BLAS is enabled
        n_threads = std::min(4, n_threads);
    }

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(backend)) {
        ggml_backend_metal_set_n_cb(backend, n_threads);
    }
#endif
}

std::string to_string(ModelType model_type) {
    static const std::unordered_map<ModelType, std::string> m{{ModelType::CHATGLM, "ChatGLM"},
                                                              {ModelType::CHATGLM2, "ChatGLM2"},
                                                              {ModelType::CHATGLM3, "ChatGLM3"},
                                                              {ModelType::CHATGLM4, "ChatGLM4"},
                                                              {ModelType::CHATGLM4V, "ChatGLM4V"}};
    return m.at(model_type);
}

static ggml_tensor *apply_rotary_emb_basic(ModelContext *mctx, ggml_tensor *layer, ggml_tensor *position_ids,
                                           RopeType rope_type, float rope_theta) {
    // tensor a (activation) is of shape [s, #h, d]
    // tensor b (position_ids) is of shape [s]
    ggml_context *ctx = mctx->ctx_b.get();
#ifdef GGML_USE_CUDA
    if (!ggml_is_contiguous(layer)) {
        layer = ggml_cont(ctx, layer);
    }
#endif
    const int head_size = layer->ne[0];
    layer = ggml_rope_ext_inplace(ctx, layer, position_ids, nullptr, head_size, (int)rope_type, 0, rope_theta, 1.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f); // [s, #h, d]
    return layer;
}

static ggml_tensor *apply_rotary_emb_glm(ModelContext *mctx, ggml_tensor *layer, ggml_tensor *position_ids) {
    // tensor a (activation) is of shape [s, #h, d]
    // tensor b (position_ids) is of shape [2 * s]
    ggml_context *ctx = mctx->ctx_b.get();

    const int head_size = layer->ne[0];
    const int num_heads = layer->ne[1];
    const int qlen = layer->ne[2];
    const int rope_dim = head_size / 2;

    ggml_tensor *b1 = ggml_view_1d(ctx, position_ids, qlen, 0);
    ggml_tensor *b2 = ggml_view_1d(ctx, position_ids, qlen, qlen * ggml_element_size(position_ids));

    ggml_tensor *a1 = ggml_view_3d(ctx, layer, head_size / 2, num_heads, qlen, layer->nb[1], layer->nb[2], 0);
    ggml_tensor *a2 = ggml_view_3d(ctx, layer, head_size / 2, num_heads, qlen, layer->nb[1], layer->nb[2],
                                   head_size / 2 * ggml_element_size(layer));

    ggml_tensor *a1_rope = a1;
    ggml_tensor *a2_rope = a2;
#ifdef GGML_USE_CUDA
    a1_rope = ggml_cont(ctx, a1_rope);
    a2_rope = ggml_cont(ctx, a2_rope);
#endif

    a1_rope = ggml_rope_inplace(ctx, a1_rope, b1, rope_dim, (int)RopeType::NEOX); // [s, #h, d/2]
    a2_rope = ggml_rope_inplace(ctx, a2_rope, b2, rope_dim, (int)RopeType::NEOX); // [s, #h, d/2]

#ifdef GGML_USE_CUDA
    a1_rope = ggml_cpy(ctx, a1_rope, a1);
    a2_rope = ggml_cpy(ctx, a2_rope, a2);
#endif
    ggml_build_forward_expand(mctx->gf, a1_rope);
    ggml_build_forward_expand(mctx->gf, a2_rope);

    return layer;
}

static ggml_tensor *apply_rotary_emb_glm2(ModelContext *mctx, ggml_tensor *layer, ggml_tensor *position_ids,
                                          float rope_theta) {
    // NOTE: ChatGLM2 applies RoPE only on half of the features. The remaining half is skipped.
    // layer: [s, #h, d], position_ids: [s]
    ggml_context *ctx = mctx->ctx_b.get();

    const int head_size = layer->ne[0];
    const int rope_dim = head_size / 2;

    ggml_tensor *half_layer_view =
        ggml_view_3d(ctx, layer, rope_dim, layer->ne[1], layer->ne[2], layer->nb[1], layer->nb[2], 0);

    ggml_tensor *half_layer = half_layer_view;
#ifdef GGML_USE_CUDA
    half_layer = ggml_cont(ctx, half_layer);
#endif
    ggml_tensor *roped_half_layer =
        ggml_rope_ext_inplace(ctx, half_layer, position_ids, nullptr, rope_dim, (int)RopeType::GPTJ, 0, rope_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f); // [s, #h, d]
#ifdef GGML_USE_CUDA
    roped_half_layer = ggml_cpy(ctx, roped_half_layer, half_layer_view);
#endif
    ggml_build_forward_expand(mctx->gf, roped_half_layer);

    return layer;
}

static ggml_tensor *apply_rotary_emb(ModelContext *mctx, ggml_tensor *layer, ggml_tensor *position_ids,
                                     RopeType rope_type, float rope_theta) {
    switch (rope_type) {
    case RopeType::GPTJ:
    case RopeType::NEOX:
        return apply_rotary_emb_basic(mctx, layer, position_ids, rope_type, rope_theta);
    case RopeType::CHATGLM:
        return apply_rotary_emb_glm(mctx, layer, position_ids);
    case RopeType::CHATGLM2:
        return apply_rotary_emb_glm2(mctx, layer, position_ids, rope_theta);
    case RopeType::DISABLED:
        return layer;
    default:
        CHATGLM_THROW << "Unknown rope type " << (int)rope_type;
    }
}

ggml_tensor *BasicAttention::forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                                     ggml_tensor *position_ids, int n_past) const {
    ggml_context *ctx = mctx->ctx_b.get();

    const int hidden_size = hidden_states->ne[0];
    const int qlen = hidden_states->ne[1];
    const int head_size = hidden_size / num_attention_heads;
    const int num_shared_q_heads = num_attention_heads / num_key_value_heads;

    ggml_tensor *qkv = query_key_value.forward(mctx, hidden_states); // [sq, (#h + 2 * #kvh) * d]

    // split mixed qkv into separate query, key and value
    ggml_tensor *query_layer; // [s, #h, d]
    ggml_tensor *key_layer;   // [s, #kvh, d]
    ggml_tensor *value_layer; // [s, #kvh, d]

    if (interleaved_qkv) {
        CHATGLM_CHECK(num_shared_q_heads == 1) << "interleaved qkv is not supported for GQA";
        query_layer = ggml_view_3d(ctx, qkv, head_size, num_attention_heads, qlen,
                                   3 * head_size * ggml_element_size(qkv), qkv->nb[1], 0);
        key_layer = ggml_view_3d(ctx, qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                                 qkv->nb[1], head_size * ggml_element_size(qkv));
        value_layer =
            ggml_view_3d(ctx, qkv, head_size, num_attention_heads, qlen, 3 * head_size * ggml_element_size(qkv),
                         qkv->nb[1], 2 * head_size * ggml_element_size(qkv));
    } else {
        query_layer = ggml_view_3d(ctx, qkv, head_size, num_attention_heads, qlen, head_size * ggml_element_size(qkv),
                                   qkv->nb[1], 0);
        key_layer = ggml_view_3d(ctx, qkv, head_size, num_key_value_heads, qlen, head_size * ggml_element_size(qkv),
                                 qkv->nb[1], hidden_size * ggml_element_size(qkv));
        value_layer =
            ggml_view_3d(ctx, qkv, head_size, num_key_value_heads, qlen, head_size * ggml_element_size(qkv), qkv->nb[1],
                         (hidden_size + head_size * num_key_value_heads) * ggml_element_size(qkv));
    }

    query_layer = apply_rotary_emb(mctx, query_layer, position_ids, rope_type, rope_theta);
    key_layer = apply_rotary_emb(mctx, key_layer, position_ids, rope_type, rope_theta);

    query_layer = ggml_cont(ctx, ggml_permute(ctx, query_layer, 0, 2, 1, 3)); // [#h, s, d]
    if (num_shared_q_heads > 1) {
        query_layer = ggml_reshape_3d(ctx, query_layer, head_size, num_shared_q_heads * qlen,
                                      num_key_value_heads); // [#kvh, (#h/#kvh) * s, d]
    }

    key_layer = ggml_permute(ctx, key_layer, 0, 2, 1, 3);     // [#kvh, s, d]
    value_layer = ggml_permute(ctx, value_layer, 1, 2, 0, 3); // [#kvh, d, s]

    if (k_cache && v_cache) {
        // store key & value to cache
        ggml_tensor *k_cache_view =
            ggml_view_3d(ctx, k_cache, head_size, qlen, num_key_value_heads, k_cache->nb[1], k_cache->nb[2],
                         (num_virtual_tokens + n_past) * k_cache->nb[1]); // [#kvh, s, d]
        ggml_tensor *v_cache_view =
            ggml_view_3d(ctx, v_cache, qlen, head_size, num_key_value_heads, v_cache->nb[1], v_cache->nb[2],
                         (num_virtual_tokens + n_past) * v_cache->nb[0]); // [#kvh, d, s]
        ggml_build_forward_expand(mctx->gf, ggml_cpy(ctx, key_layer, k_cache_view));
        ggml_build_forward_expand(mctx->gf, ggml_cpy(ctx, value_layer, v_cache_view));

        // concat key & value with past kv
        key_layer = ggml_view_3d(ctx, k_cache, head_size, num_virtual_tokens + n_past + qlen, num_key_value_heads,
                                 k_cache->nb[1], k_cache->nb[2],
                                 0); // [#kvh, kvs, d]
        value_layer = ggml_view_3d(ctx, v_cache, num_virtual_tokens + n_past + qlen, head_size, num_key_value_heads,
                                   v_cache->nb[1], v_cache->nb[2],
                                   0); // [#kvh, d, kvs]
    } else {
        key_layer = ggml_cont(ctx, key_layer);
        value_layer = ggml_cont(ctx, value_layer);
    }

    // attention
    query_layer = ggml_scale_inplace(ctx, query_layer, 1.f / std::sqrt(head_size));
    ggml_tensor *attn_scores = ggml_mul_mat(ctx, key_layer, query_layer); // [#kvh, (#h/#kvh) * s, kvs]

    if (n_past == 0) {
        // build attention mask for context input
        if (num_shared_q_heads > 1) {
            attn_scores = ggml_reshape_3d(ctx, attn_scores, num_virtual_tokens + n_past + qlen, qlen,
                                          num_attention_heads); // [#h, s, kvs]
        }

        if (attn_mask_type == AttentionMaskType::BIDIRECTIONAL) {
            // pass
        } else if (attn_mask_type == AttentionMaskType::CAUSAL) {
            attn_scores = ggml_diag_mask_inf_inplace(ctx, attn_scores, num_virtual_tokens + n_past);
        } else {
            attn_scores = ggml_add_inplace(ctx, attn_scores, attention_mask);
        }

        if (num_shared_q_heads > 1) {
            attn_scores =
                ggml_reshape_3d(ctx, attn_scores, num_virtual_tokens + n_past + qlen, num_shared_q_heads * qlen,
                                num_key_value_heads); // [#kvh, (#h/#kvh) * s, kvs]
        }
    }

    ggml_tensor *attn_probs = ggml_soft_max_inplace(ctx, attn_scores); // [#kvh, (#h/#kvh) * s, kvs]

    ggml_tensor *context_layer = ggml_mul_mat(ctx, value_layer, attn_probs); // [#kvh, (#h/#kvh) * s, d]
    if (num_shared_q_heads > 1) {
        context_layer = ggml_reshape_3d(ctx, context_layer, head_size, qlen,
                                        num_attention_heads); // [#h, s, d]
    }
    context_layer = ggml_cont(ctx, ggml_permute(ctx, context_layer, 0, 2, 1, 3)); // [s, #h, d]
    context_layer = ggml_reshape_2d(ctx, context_layer, hidden_size, qlen);       // [s, #h * d]

    ggml_tensor *attn_output = dense.forward(mctx, context_layer);
    return attn_output;
}

BaseModelForCausalLM::BaseModelForCausalLM(ModelConfig config)
    : config(config), mctx_(std::make_unique<ModelContext>()) {}

ggml_tensor *BaseModelForCausalLM::forward_graph_compute(const std::vector<int> &input_ids,
                                                         const std::optional<Image> &image, int n_past, int n_ctx,
                                                         bool is_decoding) {
    mctx_->ctx_b = make_unique_ggml_context(mctx_->compute_meta.size(), mctx_->compute_meta.data(), true);
    mctx_->gf = ggml_new_graph_custom(mctx_->ctx_b.get(), GGML_DEFAULT_GRAPH_SIZE * 4, false);

    const int qlen = (n_past == 0) ? input_ids.size() : 1;

    ggml_tensor *curr_input_ids = ggml_new_tensor_1d(mctx_->ctx_b.get(), GGML_TYPE_I32, qlen);
    ggml_set_name(curr_input_ids, "input_ids");
    ggml_set_input(curr_input_ids);

    ggml_tensor *curr_image = nullptr;
    if (n_past == 0 && image) {
        curr_image = ggml_new_tensor_3d(mctx_->ctx_b.get(), GGML_TYPE_F32, config.vision.image_size,
                                        config.vision.image_size, config.vision.in_channels);
        ggml_set_name(curr_image, "image");
        ggml_set_input(curr_image);
    }

    ggml_tensor *lm_logits = forward(mctx_.get(), curr_input_ids, curr_image, input_ids, n_past, is_decoding);
    ggml_set_output(lm_logits);

    ggml_build_forward_expand(mctx_->gf, lm_logits);
    CHATGLM_CHECK(ggml_gallocr_alloc_graph(mctx_->allocr.get(), mctx_->gf));

    // TODO: move into set_graph_inputs
    curr_input_ids = ggml_graph_get_tensor(mctx_->gf, "input_ids");
    if (curr_input_ids) {
        ggml_backend_tensor_set(curr_input_ids, input_ids.data() + input_ids.size() - qlen, 0, qlen * sizeof(int));
    }
    set_graph_inputs(input_ids, image, n_past, n_ctx);

    set_default_num_threads(mctx_->backend.get(), qlen);
    CHATGLM_CHECK(ggml_backend_graph_compute(mctx_->backend.get(), mctx_->gf) == GGML_STATUS_SUCCESS);

#ifdef GGML_PERF
    ggml_graph_print(mctx_->gf);
#endif

    return lm_logits;
}

int BaseModelForCausalLM::generate_next_token(const std::vector<int> &input_ids, const std::optional<Image> &image,
                                              const GenerationConfig &gen_config, int n_past, int n_ctx) {
    ggml_tensor *lm_logits = forward_graph_compute(input_ids, image, n_past, n_ctx, true);
    CHATGLM_CHECK(ggml_n_dims(lm_logits) == 1);

    int vocab_size = lm_logits->ne[0];
    std::vector<float> next_token_logits(vocab_size);
    ggml_backend_tensor_get(lm_logits, next_token_logits.data(), 0, vocab_size * sizeof(float));

    // check nan
    for (int i = 0; i < vocab_size; i++) {
        CHATGLM_CHECK(std::isfinite(next_token_logits[i])) << "nan/inf encountered at lm_logits[" << i << "]";
    }

    // logits pre-process
    if (gen_config.repetition_penalty != 1.f) {
        sampling_repetition_penalty(next_token_logits.data(), next_token_logits.data() + vocab_size, input_ids,
                                    gen_config.repetition_penalty);
    }

    int next_token_id;
    if (gen_config.do_sample) {
        // temperature sampling
        if (gen_config.temperature > 0) {
            sampling_temperature(next_token_logits.data(), next_token_logits.data() + vocab_size,
                                 gen_config.temperature);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, next_token_logits[i]);
        }

        // top_k sampling
        if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
                           token_scores.data() + token_scores.size());
            token_scores.resize(gen_config.top_k);
        }

        // top_p sampling
        if (0.f < gen_config.top_p && gen_config.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            next_token_logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(next_token_logits.data(), next_token_logits.data() + token_scores.size());
        next_token_id = token_scores[dist(gen)].id;
    } else {
        // greedy search
        next_token_id =
            std::max_element(next_token_logits.begin(), next_token_logits.end()) - next_token_logits.begin();
    }

    return next_token_id;
}

void BaseModelForCausalLM::sampling_repetition_penalty(float *first, float *last, const std::vector<int> &input_ids,
                                                       float penalty) {
    CHATGLM_CHECK(penalty > 0) << "penalty must be a positive float, but got " << penalty;
    const float inv_penalty = 1.f / penalty;
    const int vocab_size = last - first;
    std::vector<bool> occurrence(vocab_size, false);
    for (const int id : input_ids) {
        if (!occurrence[id]) {
            first[id] *= (first[id] > 0) ? inv_penalty : penalty;
        }
        occurrence[id] = true;
    }
}

void BaseModelForCausalLM::sampling_temperature(float *first, float *last, float temp) {
    const float inv_temp = 1.f / temp;
    for (float *it = first; it != last; it++) {
        *it *= inv_temp;
    }
}

void BaseModelForCausalLM::sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last) {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

TokenIdScore *BaseModelForCausalLM::sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p) {
    // fast top_p in expected O(n) time complexity
    sampling_softmax_inplace(first, last);

    while (first + 1 < last) {
        const float pivot_score = (last - 1)->score; // use mid score?
        TokenIdScore *mid =
            std::partition(first, last - 1, [pivot_score](const TokenIdScore &x) { return x.score > pivot_score; });
        std::swap(*mid, *(last - 1));

        const float prefix_sum =
            std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x) { return sum + x.score; });
        if (prefix_sum >= top_p) {
            last = mid;
        } else if (prefix_sum + mid->score < top_p) {
            first = mid + 1;
            top_p -= prefix_sum + mid->score;
        } else {
            return mid + 1;
        }
    }
    return last;
}

void BaseModelForCausalLM::sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last) {
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

std::vector<int> BaseModelForCausalLM::generate(const std::vector<int> &input_ids, const std::optional<Image> &image,
                                                const GenerationConfig &gen_config, BaseStreamer *streamer) {
    CHATGLM_CHECK(gen_config.max_length <= config.max_length)
        << "Requested max_length (" << gen_config.max_length << ") exceeds pre-configured model max_length ("
        << config.max_length << ")";

    std::vector<int> output_ids;
    output_ids.reserve(gen_config.max_length);
    output_ids = input_ids;
    if (streamer) {
        streamer->put(input_ids);
    }

    int n_past = 0;
    const int n_ctx = input_ids.size();
    const int max_new_tokens = (gen_config.max_new_tokens > 0) ? gen_config.max_new_tokens : gen_config.max_length;

    while ((int)output_ids.size() < std::min(gen_config.max_length, n_ctx + max_new_tokens)) {
        int next_token_id = generate_next_token(output_ids, image, gen_config, n_past, n_ctx);

        n_past = count_tokens(output_ids, image);
        output_ids.emplace_back(next_token_id);

        if (streamer) {
            streamer->put({next_token_id});
        }

        if (next_token_id == config.eos_token_id ||
            std::find(config.extra_eos_token_ids.begin(), config.extra_eos_token_ids.end(), next_token_id) !=
                config.extra_eos_token_ids.end()) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return output_ids;
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

std::vector<int> ChatGLMTokenizer::encode(const std::string &text, int max_length) const {
    std::string input = preprocess(text);
    std::vector<int> ids;
    sp.Encode(input, &ids);
    ids.insert(ids.end(), {gmask_token_id, bos_token_id});
    if ((int)ids.size() > max_length) {
        // sliding window: always take the last max_length tokens
        ids.erase(ids.begin(), ids.end() - max_length);
    }
    return ids;
}

std::vector<int> ChatGLMTokenizer::apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const {
    std::string prompt = apply_chat_template_text(messages);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

std::string ChatGLMTokenizer::apply_chat_template_text(const std::vector<ChatMessage> &messages) {
    check_chat_messages(messages);
    std::vector<ChatMessage> user_assistant_messages = filter_user_assistant_messages(messages);

    std::ostringstream oss_prompt;
    if (user_assistant_messages.size() == 1) {
        oss_prompt << user_assistant_messages.front().content;
    } else {
        for (size_t i = 0; i < user_assistant_messages.size(); i += 2) {
            oss_prompt << "[Round " << i / 2 << "]\n问：" << user_assistant_messages[i].content << "\n答：";
            if (i + 1 < user_assistant_messages.size()) {
                oss_prompt << user_assistant_messages[i + 1].content << "\n";
            }
        }
    }
    return oss_prompt.str();
}

std::string ChatGLMTokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) const {
    CHATGLM_CHECK(skip_special_tokens) << "unimplemented";
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

static inline std::string replace_punctuations(const std::string &text) {
    // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
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
    std::wstring w_output = converter.from_bytes(text);
    for (const auto &punct_pair : punct_map) {
        w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
    }
    std::string output = converter.to_bytes(w_output);
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
    // punctuations
    output = replace_punctuations(output);

    return output;
}

ggml_tensor *GLMBlock::forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                               ggml_tensor *position_ids, int n_past) const {
    ggml_context *ctx = mctx->ctx_b.get();

    ggml_tensor *attn_input = input_layernorm.forward(mctx, hidden_states);
    ggml_tensor *attn_output = attention.forward(mctx, attn_input, attention_mask, position_ids, n_past);
    ggml_build_forward_expand(mctx->gf, attn_output);
    attn_input = ggml_scale_inplace(ctx, attn_input, alpha);
    hidden_states = ggml_add_inplace(ctx, attn_input, attn_output);

    ggml_tensor *mlp_input = post_attention_layernorm.forward(mctx, hidden_states);
    ggml_tensor *mlp_output = mlp.forward(mctx, mlp_input);
    ggml_build_forward_expand(mctx->gf, mlp_output);
    mlp_input = ggml_scale_inplace(ctx, mlp_input, alpha);
    ggml_tensor *output = ggml_add_inplace(ctx, mlp_input, mlp_output);

    return output;
}

static void alloc_weight_context(ModelContext *mctx, const ggml_backend_buffer_t sd_buf) {
    void *sd_buf_base = ggml_backend_buffer_get_base(sd_buf);
    const size_t sd_buf_size = ggml_backend_buffer_get_size(sd_buf);
    if (ggml_backend_is_cpu(mctx->backend.get())) {
        mctx->buf_w = unique_ggml_backend_buffer_t(ggml_backend_cpu_buffer_from_ptr(sd_buf_base, sd_buf_size));
    }
#ifdef GGML_USE_METAL
    else if (ggml_backend_is_metal(mctx->backend.get())) {
        const size_t max_size = ggml_get_max_tensor_size(mctx->ctx_w.get());
        mctx->buf_w =
            unique_ggml_backend_buffer_t(ggml_backend_metal_buffer_from_ptr(sd_buf_base, sd_buf_size, max_size));
    }
#endif
    else {
        mctx->buf_w =
            unique_ggml_backend_buffer_t(ggml_backend_alloc_ctx_tensors(mctx->ctx_w.get(), mctx->backend.get()));
    }
}

void ChatGLMForCausalLM::load_state_dict(const StateDict &sd) {
    alloc_weight_context(mctx_.get(), sd.buf.get());

    StateDict self_sd = state_dict();
    for (auto &item : self_sd.kv) {
        const std::string &name = item.first;
        ggml_tensor *self_weight = item.second;
        ggml_tensor *ckpt_weight = sd.kv.at(name);
        CHATGLM_CHECK(ggml_nbytes(self_weight) == ggml_nbytes(ckpt_weight));
        if (ggml_backend_is_cpu(mctx_->backend.get()) || ggml_cpu_has_metal()) {
            ggml_backend_tensor_alloc(mctx_->buf_w.get(), self_weight, ckpt_weight->data);
        } else {
            ggml_backend_tensor_set(self_weight, ckpt_weight->data, 0, ggml_nbytes(self_weight));
        }
    }
}

void ChatGLMModel::set_graph_inputs(ggml_cgraph *gf, const std::vector<int> &input_ids,
                                    const std::optional<Image> &image, int n_past, int n_ctx) const {
    // attention_mask: [s, kvs] auto broadcast to [#h, s, kvs]
    // semantic: attn_scores[:, :-1, -1] = -inf
    const int qlen = input_ids.size() - n_past;
    if (n_past == 0) {
        ggml_tensor *attention_mask = ggml_graph_get_tensor(gf, "attention_mask");
        const int kvlen = attention_mask->ne[0];
        std::vector<float> attention_mask_buffer(qlen * kvlen, 0.f);
        CHATGLM_CHECK(ggml_nbytes(attention_mask) == attention_mask_buffer.size() * sizeof(float));
        for (int i = 0; i < qlen - 1; i++) {
            attention_mask_buffer[i * kvlen + (kvlen - 1)] = -INFINITY;
        }
        ggml_backend_tensor_set(attention_mask, attention_mask_buffer.data(), 0,
                                attention_mask_buffer.size() * sizeof(float));
    }

    // position_ids: [2 * qlen]
    ggml_tensor *position_ids = ggml_graph_get_tensor(gf, "position_ids");
    CHATGLM_CHECK(ggml_n_dims(position_ids) == 1 && position_ids->ne[0] == 2 * qlen)
        << "invalid position ids size " << position_ids->ne[0];

    std::vector<int> position_ids_buffer(position_ids->ne[0]);
    for (int i = 0; i < qlen; i++) {
        const int p = n_past + i;
        position_ids_buffer[i] = std::min(p, n_ctx - 2);
        position_ids_buffer[qlen + i] = std::max(p - (n_ctx - 2), 0);
    }
    ggml_backend_tensor_set(position_ids, position_ids_buffer.data(), 0, position_ids_buffer.size() * sizeof(int));
}

StateDict ChatGLMForCausalLM::state_dict() const {
    StateDict sd;
    sd.kv.emplace("transformer.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.layers." + std::to_string(i) + '.';
        sd.kv.emplace(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        sd.kv.emplace(layer_prefix + "input_layernorm.bias", transformer.layers[i].input_layernorm.bias);
        sd.kv.emplace(layer_prefix + "attention.query_key_value.weight",
                      transformer.layers[i].attention.query_key_value.weight);
        sd.kv.emplace(layer_prefix + "attention.query_key_value.bias",
                      transformer.layers[i].attention.query_key_value.bias);
        sd.kv.emplace(layer_prefix + "attention.dense.weight", transformer.layers[i].attention.dense.weight);
        sd.kv.emplace(layer_prefix + "attention.dense.bias", transformer.layers[i].attention.dense.bias);
        sd.kv.emplace(layer_prefix + "post_attention_layernorm.weight",
                      transformer.layers[i].post_attention_layernorm.weight);
        sd.kv.emplace(layer_prefix + "post_attention_layernorm.bias",
                      transformer.layers[i].post_attention_layernorm.bias);
        sd.kv.emplace(layer_prefix + "mlp.dense_h_to_4h.weight", transformer.layers[i].mlp.dense_h_to_4h.weight);
        sd.kv.emplace(layer_prefix + "mlp.dense_h_to_4h.bias", transformer.layers[i].mlp.dense_h_to_4h.bias);
        sd.kv.emplace(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.dense_4h_to_h.weight);
        sd.kv.emplace(layer_prefix + "mlp.dense_4h_to_h.bias", transformer.layers[i].mlp.dense_4h_to_h.bias);
    }
    sd.kv.emplace("transformer.final_layernorm.weight", transformer.final_layernorm.weight);
    sd.kv.emplace("transformer.final_layernorm.bias", transformer.final_layernorm.bias);
    return sd;
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

std::vector<int> ChatGLM2Tokenizer::encode(const std::string &text, int max_length) const {
    std::vector<int> ids;
    sp.Encode(text, &ids);
    ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    if ((int)ids.size() > max_length) {
        // sliding window: drop the least recent history while keeping the two special prefix tokens
        int num_drop = (int)ids.size() - max_length;
        ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
    }
    return ids;
}

std::string ChatGLM2Tokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) const {
    CHATGLM_CHECK(skip_special_tokens) << "unimplemented";
    // filter out special tokens
    std::vector<int> normal_ids(ids);
    normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                     normal_ids.end());

    std::string text;
    sp.Decode(normal_ids, &text);
    text = replace_punctuations(text);
    return text;
}

std::vector<int> ChatGLM2Tokenizer::apply_chat_template(const std::vector<ChatMessage> &messages,
                                                        int max_length) const {
    std::string prompt = apply_chat_template_text(messages);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

std::string ChatGLM2Tokenizer::apply_chat_template_text(const std::vector<ChatMessage> &messages) {
    check_chat_messages(messages);
    std::vector<ChatMessage> user_assistant_messages = filter_user_assistant_messages(messages);

    std::ostringstream oss_prompt;
    for (size_t i = 0; i < user_assistant_messages.size(); i += 2) {
        oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << user_assistant_messages[i].content << "\n\n答：";
        if (i < user_assistant_messages.size() - 1) {
            oss_prompt << user_assistant_messages[i + 1].content << "\n\n";
        }
    }
    return oss_prompt.str();
}

bool ChatGLM2Tokenizer::is_special_id(int id) const {
    return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
           id == eop_token_id;
}

void ChatGLM2ForCausalLM::load_state_dict(const StateDict &sd) {
    alloc_weight_context(mctx_.get(), sd.buf.get());

    if (config.num_virtual_tokens > 0) {
        ggml_tensor *past_key_values = sd.kv.at("past_key_values");
        load_prefix_cache(past_key_values);
    }

    auto self_sd = state_dict();
    load_state_dict(mctx_.get(), self_sd, sd);
}

void ChatGLM2ForCausalLM::load_state_dict(ModelContext *mctx, StateDict &dst, const StateDict &src) {
    for (auto it = src.kv.begin(); it != src.kv.end(); it++) {
        const std::string &name = it->first;
        ggml_tensor *ckpt_weight = it->second;

        if (name == "past_key_values") {
            continue;
        }

        size_t pos = name.rfind("mlp.dense_h_to_4h.weight");
        if (pos != std::string::npos) {
            // split dense_h_to_4h to gate & up
            std::string gate_name = name.substr(0, pos) + "mlp.gate_proj.weight";
            ggml_tensor *gate_proj = dst.kv.at(gate_name);

            std::string up_name = name.substr(0, pos) + "mlp.up_proj.weight";
            ggml_tensor *up_proj = dst.kv.at(up_name);

            CHATGLM_CHECK(ggml_nbytes(ckpt_weight) == ggml_nbytes(gate_proj) + ggml_nbytes(up_proj)) << name;

            if (ggml_backend_is_cpu(mctx->backend.get()) || ggml_cpu_has_metal()) {
                ggml_backend_tensor_alloc(mctx->buf_w.get(), gate_proj, ckpt_weight->data);
                ggml_backend_tensor_alloc(mctx->buf_w.get(), up_proj,
                                          (char *)ckpt_weight->data + ggml_nbytes(gate_proj));
            } else {
                ggml_backend_tensor_set(gate_proj, ckpt_weight->data, 0, ggml_nbytes(gate_proj));
                ggml_backend_tensor_set(up_proj, (char *)ckpt_weight->data + ggml_nbytes(gate_proj), 0,
                                        ggml_nbytes(up_proj));
            }
        } else {
            // normal weight
            ggml_tensor *self_weight = dst.kv.at(name);
            CHATGLM_CHECK(ggml_nbytes(self_weight) == ggml_nbytes(ckpt_weight)) << name;
            if (ggml_backend_is_cpu(mctx->backend.get()) || ggml_cpu_has_metal()) {
                ggml_backend_tensor_alloc(mctx->buf_w.get(), self_weight, ckpt_weight->data);
            } else {
                ggml_backend_tensor_set(self_weight, ckpt_weight->data, 0, ggml_nbytes(self_weight));
            }
        }
    }
}

void ChatGLM2Model::set_graph_inputs(ggml_cgraph *gf, const std::vector<int> &input_ids,
                                     const std::optional<Image> &image, int n_past, int n_ctx) const {
    ggml_tensor *position_ids = ggml_graph_get_tensor(gf, "position_ids");
    const int qlen = input_ids.size() - n_past;
    CHATGLM_CHECK(ggml_n_dims(position_ids) == 1 && position_ids->ne[0] == qlen)
        << "invalid position ids size " << position_ids->ne[0];

    std::vector<int> position_ids_buffer(position_ids->ne[0]);
    std::iota(position_ids_buffer.begin(), position_ids_buffer.end(), n_past);
    ggml_backend_tensor_set(position_ids, position_ids_buffer.data(), 0, position_ids_buffer.size() * sizeof(int));
}

StateDict ChatGLM2ForCausalLM::state_dict() const {
    StateDict sd;
    sd.kv.emplace("transformer.embedding.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string layer_prefix = "transformer.encoder.layers." + std::to_string(i) + '.';
        sd.kv.emplace(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        sd.kv.emplace(layer_prefix + "self_attention.query_key_value.weight",
                      transformer.layers[i].attention.query_key_value.weight);
        sd.kv.emplace(layer_prefix + "self_attention.query_key_value.bias",
                      transformer.layers[i].attention.query_key_value.bias);
        sd.kv.emplace(layer_prefix + "self_attention.dense.weight", transformer.layers[i].attention.dense.weight);
        sd.kv.emplace(layer_prefix + "post_attention_layernorm.weight",
                      transformer.layers[i].post_attention_layernorm.weight);
        sd.kv.emplace(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        sd.kv.emplace(layer_prefix + "mlp.up_proj.weight", transformer.layers[i].mlp.up_proj.weight);
        // for compatibility
        sd.kv.emplace(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.down_proj.weight);
    }
    sd.kv.emplace("transformer.encoder.final_layernorm.weight", transformer.final_layernorm.weight);
    sd.kv.emplace("transformer.output_layer.weight", lm_head.weight);
    return sd;
}

// ===== ChatGLM3-6B =====

ChatGLM3Tokenizer::ChatGLM3Tokenizer(std::string_view serialized_model_proto) {
    const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
    CHATGLM_CHECK(status.ok()) << status.ToString();

    int special_id = sp.GetPieceSize();
    mask_token_id = special_id++;
    gmask_token_id = special_id++;
    smask_token_id = special_id++;
    sop_token_id = special_id++;
    eop_token_id = special_id++;
    system_token_id = special_id++;
    user_token_id = special_id++;
    assistant_token_id = special_id++;
    observation_token_id = special_id++;

    special_tokens = {
        {"[MASK]", mask_token_id},
        {"[gMASK]", gmask_token_id},
        {"[sMASK]", smask_token_id},
        {"sop", sop_token_id},
        {"eop", eop_token_id},
        {"<|system|>", system_token_id},
        {"<|user|>", user_token_id},
        {"<|assistant|>", assistant_token_id},
        {"<|observation|>", observation_token_id},
    };

    for (const auto &item : special_tokens) {
        index_special_tokens[item.second] = item.first;
    }
}

std::vector<int> ChatGLM3Tokenizer::encode(const std::string &text, int max_length) const {
    std::vector<int> ids;
    sp.Encode(text, &ids);
    ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    truncate(ids, max_length);
    return ids;
}

std::string ChatGLM3Tokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) const {
    std::vector<std::string> pieces;
    for (int id : ids) {
        auto pos = index_special_tokens.find(id);
        if (pos != index_special_tokens.end()) {
            // special tokens
            pieces.emplace_back(pos->second);
        } else {
            // normal tokens
            pieces.emplace_back(sp.IdToPiece(id));
        }
    }

    std::string text = sp.DecodePieces(pieces);

    if (skip_special_tokens) {
        text = remove_special_tokens(text);
    }

    return text;
}

std::string ChatGLM3Tokenizer::remove_special_tokens(const std::string &text) {
    // R"(<\|assistant\|> interpreter)"
    // R"(<\|assistant\|> interpre)"
    static const std::regex re(R"(<\|assistant\|>|<\|user\|>|<\|observation\|>)");
    std::string output = std::regex_replace(text, re, "");
    return output;
}

std::vector<int> ChatGLM3Tokenizer::encode_single_message(const std::string &role, const std::string &content) const {
    std::vector<int> input_ids;
    input_ids.emplace_back(get_command("<|" + role + "|>"));
    // TODO: support metadata
    std::vector<int> newline_ids;
    sp.Encode("\n", &newline_ids);
    input_ids.insert(input_ids.end(), newline_ids.begin(), newline_ids.end());
    std::vector<int> content_ids;
    sp.Encode(content, &content_ids);
    input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
    return input_ids;
}

std::vector<int> ChatGLM3Tokenizer::apply_chat_template(const std::vector<ChatMessage> &messages,
                                                        int max_length) const {
    std::vector<int> input_ids{gmask_token_id, sop_token_id};
    for (const auto &msg : messages) {
        auto msg_ids = encode_single_message(msg.role, msg.content);
        input_ids.insert(input_ids.end(), msg_ids.begin(), msg_ids.end());

        // encode code block into a separate message
        if (!msg.tool_calls.empty() && msg.tool_calls.front().type == ToolCallMessage::TYPE_CODE) {
            auto code_ids = encode_single_message(msg.role, msg.tool_calls.front().code.input);
            input_ids.insert(input_ids.end(), code_ids.begin(), code_ids.end());
        }
    }
    input_ids.emplace_back(assistant_token_id);
    truncate(input_ids, max_length);
    return input_ids;
}

ChatMessage ChatGLM3Tokenizer::decode_message(const std::vector<int> &ids) const {
    ChatMessage message;
    if (!ids.empty() && ids.back() == observation_token_id) {
        // insert an <|assistant|> token before content to match possible interpreter delimiter
        std::vector<int> full_ids{assistant_token_id};
        full_ids.insert(full_ids.end(), ids.begin(), ids.end());

        std::string output = decode(full_ids, false);
        const std::string ci_delim = "<|assistant|> interpreter";
        size_t ci_pos = output.find(ci_delim);
        if (ci_pos != std::string::npos) {
            // code interpreter
            std::string chat_output = output.substr(0, ci_pos);
            chat_output = remove_special_tokens(chat_output);
            trim(chat_output);
            std::string code_output = output.substr(ci_pos + ci_delim.size());
            code_output = remove_special_tokens(code_output);
            trim(code_output);
            message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(chat_output), std::nullopt,
                                  {ToolCallMessage(CodeMessage(std::move(code_output)))});
        } else {
            // tool call
            output = remove_special_tokens(output);

            // parse tool name
            std::string tool_name = "PARSE_ERROR";
            size_t pos = output.find('\n');
            if (pos != std::string::npos) {
                // split tool name and args by 1st linebreak
                tool_name = output.substr(0, pos);
                trim(tool_name);
                output.erase(0, pos + 1);
            }

            // post process output
            trim(output);

            // extract args
            std::string tool_args = "PARSE_ERROR";
            static const std::regex args_regex(R"(```.*?\n(.*?)\n```)");
            std::smatch sm;
            if (std::regex_search(output, sm, args_regex)) {
                CHATGLM_CHECK(sm.size() == 2) << "unexpected regex match results";
                tool_args = sm[1];
            }

            message = ChatMessage(ChatMessage::ROLE_ASSISTANT, std::move(output), std::nullopt,
                                  {ToolCallMessage(FunctionMessage(std::move(tool_name), std::move(tool_args)))});
        }
    } else {
        // conversation
        message = BaseTokenizer::decode_message(ids);
        trim(message.content); // strip leading linebreak in conversation mode
    }
    return message;
}

int ChatGLM3Tokenizer::get_command(const std::string &token) const {
    auto pos = special_tokens.find(token);
    CHATGLM_CHECK(pos != special_tokens.end()) << token << " is not a special token";
    return pos->second;
}

bool ChatGLM3Tokenizer::is_special_id(int id) const { return index_special_tokens.count(id) > 0; }

void ChatGLM3Tokenizer::truncate(std::vector<int> &ids, int max_length) {
    if ((int)ids.size() > max_length) {
        // sliding window: drop the least recent history while keeping the two special prefix tokens
        int num_drop = (int)ids.size() - max_length;
        ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
    }
}

// ===== ChatGLM4-9B =====

TiktokenCoreBPE::TiktokenCoreBPE(std::unordered_map<std::string, int> encoder,
                                 std::unordered_map<std::string, int> special_tokens_encoder,
                                 const std::string &pattern)
    : regex(std::make_unique<RE2>("(" + pattern + ")")), encoder(std::move(encoder)),
      special_tokens_encoder(std::move(special_tokens_encoder)) {
    CHATGLM_CHECK(regex->ok()) << regex->error();
    CHATGLM_CHECK(regex->NumberOfCapturingGroups() <= 2) << "unimplemented";

    decoder.reserve(this->encoder.size());
    for (const auto &[token, rank] : this->encoder) {
        decoder.emplace(rank, token);
    }

    special_tokens_decoder.reserve(this->special_tokens_encoder.size());
    for (const auto &[token, rank] : this->special_tokens_encoder) {
        special_tokens_decoder.emplace(rank, token);
    }
}

std::vector<std::pair<size_t, int>> TiktokenCoreBPE::_byte_pair_merge(const std::unordered_map<std::string, int> &ranks,
                                                                      const std::string &piece) {
    using rank_t = int;

    std::vector<std::pair<size_t, rank_t>> parts; // (start, rank)
    parts.reserve(piece.length() + 1);

    auto min_rank = std::make_pair<rank_t, size_t>(std::numeric_limits<rank_t>::max(),
                                                   std::numeric_limits<size_t>::max()); // (rank, start)

    for (size_t i = 0; i < piece.length() - 1; i++) {
        rank_t rank = std::numeric_limits<rank_t>::max();
        if (auto it = ranks.find(piece.substr(i, 2)); it != ranks.end()) {
            rank = it->second;
        }
        if (rank < min_rank.first) {
            min_rank = std::make_pair(rank, i);
        }
        parts.emplace_back(std::make_pair(i, rank));
    }
    parts.emplace_back(std::make_pair(piece.length() - 1, std::numeric_limits<rank_t>::max()));
    parts.emplace_back(std::make_pair(piece.length(), std::numeric_limits<rank_t>::max()));

    auto get_rank = [&piece, &ranks](const std::vector<std::pair<size_t, rank_t>> &parts, size_t i) {
        if (i + 3 < parts.size()) {
            size_t start = parts[i].first;
            size_t end = parts[i + 3].first;
            if (auto it = ranks.find(piece.substr(start, end - start)); it != ranks.end()) {
                return it->second;
            }
        }
        return std::numeric_limits<rank_t>::max();
    };

    while (min_rank.first != std::numeric_limits<rank_t>::max()) {
        size_t i = min_rank.second;
        if (i > 0) {
            parts[i - 1].second = get_rank(parts, i - 1);
        }
        parts[i].second = get_rank(parts, i);
        parts.erase(parts.begin() + i + 1);

        min_rank = std::make_pair(std::numeric_limits<rank_t>::max(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < parts.size() - 1; i++) {
            rank_t rank = parts[i].second;
            if (rank < min_rank.first) {
                min_rank = std::make_pair(rank, i);
            }
        }
    }

    return parts;
}

std::vector<int> TiktokenCoreBPE::byte_pair_encode(const std::string &piece,
                                                   const std::unordered_map<std::string, int> &ranks) {
    CHATGLM_CHECK(piece.length() > 1);

    auto parts = _byte_pair_merge(ranks, piece);

    std::vector<int> tokens;
    tokens.reserve(parts.size() - 1);

    for (size_t i = 1; i < parts.size(); i++) {
        size_t start = parts[i - 1].first;
        size_t end = parts[i].first;
        int rank = ranks.at(piece.substr(start, end - start));
        tokens.emplace_back(rank);
    }

    return tokens;
}

std::vector<int> TiktokenCoreBPE::_encode_ordinary_native(const std::string &text) const {
    std::vector<int> ret;
    re2::StringPiece input(text);
    re2::StringPiece prev_input(input);
    std::string piece;
    std::string piece2;
    while (RE2::FindAndConsume(&input, *regex, &piece, &piece2)) {
        if (!piece2.empty()) {
            // workaround for lookahead: capture sub group and restore input
            auto pos = prev_input.find(piece2);
            input = prev_input.substr(pos + piece2.length());
            piece = std::move(piece2);
        }
        if (auto it = encoder.find(piece); it != encoder.end()) {
            ret.emplace_back(it->second);
        } else {
            std::vector<int> bpe_ids = byte_pair_encode(piece, encoder);
            ret.insert(ret.end(), bpe_ids.begin(), bpe_ids.end());
        }
        prev_input = input;
    }
    return ret;
}

std::string TiktokenCoreBPE::_decode_native(const std::vector<int> &tokens) const {
    std::string ret;
    ret.reserve(tokens.size() * 2);
    for (int token : tokens) {
        if (auto it = decoder.find(token); it != decoder.end()) {
            ret.append(it->second);
        } else if (auto it = special_tokens_decoder.find(token); it != special_tokens_decoder.end()) {
            ret.append(it->second);
        } else {
            CHATGLM_THROW << "Unknown token " << token;
        }
    }
    return ret;
}

ChatGLM4Tokenizer::ChatGLM4Tokenizer(const std::string &vocab_text) {
    std::istringstream in(vocab_text);
    std::string base64_token;
    int rank;
    std::unordered_map<std::string, int> mergeable_ranks;
    while (in >> base64_token >> rank) {
        std::string token;
        CHATGLM_CHECK(google::protobuf::Base64Unescape(base64_token, &token));
        mergeable_ranks.emplace(std::move(token), rank);
    }
    size_t vocab_size = mergeable_ranks.size();

    const std::vector<std::string> all_special_tokens = {"<|endoftext|>",
                                                         "[MASK]",
                                                         "[gMASK]",
                                                         "[sMASK]",
                                                         "<sop>",
                                                         "<eop>",
                                                         "<|system|>",
                                                         "<|user|>",
                                                         "<|assistant|>",
                                                         "<|observation|>",
                                                         "<|begin_of_image|>",
                                                         "<|end_of_image|>",
                                                         "<|begin_of_video|>",
                                                         "<|end_of_video|>"};

    std::unordered_map<std::string, int> special_tokens_encoder;
    special_tokens_encoder.reserve(all_special_tokens.size());
    for (const auto &token : all_special_tokens) {
        special_tokens_encoder.emplace(token, vocab_size++);
    }
    // common special token ids
    eos_token_id = special_tokens_encoder.at("<|endoftext|>");
    gmask_token_id = special_tokens_encoder.at("[gMASK]");
    sop_token_id = special_tokens_encoder.at("<sop>");
    user_token_id = special_tokens_encoder.at("<|user|>");
    assistant_token_id = special_tokens_encoder.at("<|assistant|>");
    observation_token_id = special_tokens_encoder.at("<|observation|>");
    boi_token_id = special_tokens_encoder.at("<|begin_of_image|>");
    eoi_token_id = special_tokens_encoder.at("<|end_of_image|>");

    const std::string pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|(\s+)(?:\s)|\s+)";
    core_bpe = TiktokenCoreBPE(std::move(mergeable_ranks), std::move(special_tokens_encoder), pattern);
}

std::vector<int> ChatGLM4Tokenizer::encode(const std::string &text, int max_length) const {
    std::vector<int> ids = core_bpe.encode_ordinary(text);
    ids.insert(ids.begin(), {gmask_token_id, sop_token_id}); // special prefix
    truncate(ids, max_length);
    return ids;
}

std::string ChatGLM4Tokenizer::decode(const std::vector<int> &ids, bool skip_special_tokens) const {
    std::vector<int> valid_ids = ids;
    if (skip_special_tokens) {
        valid_ids.erase(std::remove_if(valid_ids.begin(), valid_ids.end(),
                                       [this](int id) { return core_bpe.special_tokens_decoder.count(id) > 0; }),
                        valid_ids.end());
    }
    return core_bpe.decode(valid_ids);
}

ChatMessage ChatGLM4Tokenizer::decode_message(const std::vector<int> &ids) const {
    // TODO: support tool call
    ChatMessage message = BaseTokenizer::decode_message(ids);
    trim(message.content); // strip leading linebreak in conversation mode
    return message;
}

std::vector<int> ChatGLM4Tokenizer::apply_chat_template(const std::vector<ChatMessage> &messages,
                                                        int max_length) const {
    std::vector<int> input_ids{gmask_token_id, sop_token_id};
    std::vector<int> newline_ids = core_bpe.encode_ordinary("\n");
    for (const auto &msg : messages) {
        input_ids.emplace_back(core_bpe.special_tokens_encoder.at("<|" + msg.role + "|>"));
        input_ids.insert(input_ids.end(), newline_ids.begin(), newline_ids.end());
        if (msg.image.has_value()) {
            input_ids.insert(input_ids.end(), {boi_token_id, eos_token_id, eoi_token_id});
        }
        std::vector<int> content_ids = core_bpe.encode_ordinary(msg.content);
        input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
    }
    input_ids.emplace_back(assistant_token_id);
    truncate(input_ids, max_length);
    return input_ids;
}

void ChatGLM4Tokenizer::truncate(std::vector<int> &ids, int max_length) {
    if ((int)ids.size() > max_length) {
        // sliding window: drop the least recent history while keeping the two special prefix tokens
        int num_drop = (int)ids.size() - max_length;
        ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
    }
}

// ===== GLM4V-9B =====

ggml_tensor *Conv2d::forward(ModelContext *mctx, ggml_tensor *input) const {
    // input: [b, c, h, w]
    ggml_context *ctx = mctx->ctx_b.get();
    ggml_tensor *output = ggml_conv_2d(ctx, weight, input, stride, stride, 0, 0, 1, 1); // [b, oc, oh, ow]
    output = ggml_add_inplace(ctx, output, bias);
    return output;
}

ggml_tensor *PatchEmbedding::forward(ModelContext *mctx, ggml_tensor *input) const {
    // input: [c, h, w]
    ggml_context *ctx = mctx->ctx_b.get();
    input = proj.forward(mctx, input);                                              // [oc, oh, ow]
    input = ggml_reshape_2d(ctx, input, input->ne[0] * input->ne[1], input->ne[2]); // [oc, oh * ow]
    input = ggml_permute(ctx, input, 1, 0, 2, 3);                                   // [oh * ow, oc] view as [s, h]

    ggml_tensor *embeddings = ggml_new_tensor_2d(ctx, input->type, hidden_size(), num_positions());
    ggml_set_name(embeddings, "embeddings");
    ggml_set_input(embeddings);

    // concat (cls, x)
    ggml_tensor *cls_embedding_view =
        ggml_cpy(ctx, cls_embedding, ggml_view_1d(ctx, embeddings, cls_embedding->ne[0], 0));
    ggml_tensor *img_embedding_view = ggml_cpy(
        ctx, input, ggml_view_2d(ctx, embeddings, input->ne[0], input->ne[1], embeddings->nb[1], embeddings->nb[1]));
    ggml_build_forward_expand(mctx->gf, cls_embedding_view);
    ggml_build_forward_expand(mctx->gf, img_embedding_view);

    embeddings = ggml_add_inplace(ctx, embeddings, position_embedding.weight); // [s, h]
    return embeddings;
}

ggml_tensor *EVA2CLIPBlock::forward(ModelContext *mctx, ggml_tensor *hidden_states, ggml_tensor *attention_mask,
                                    ggml_tensor *position_ids, int n_past) const {
    ggml_context *ctx = mctx->ctx_b.get();

    ggml_tensor *residual = hidden_states;
    hidden_states = attention.forward(mctx, hidden_states, attention_mask, position_ids, n_past);
    hidden_states = input_layernorm.forward(mctx, hidden_states);
    hidden_states = ggml_add_inplace(ctx, hidden_states, residual);

    residual = hidden_states;
    hidden_states = mlp.forward(mctx, hidden_states);
    hidden_states = post_attention_layernorm.forward(mctx, hidden_states);
    hidden_states = ggml_add_inplace(ctx, hidden_states, residual);

    return hidden_states;
}

EVA2CLIPTransformer::EVA2CLIPTransformer(ModelContext *mctx, const VisionModelConfig &config) {
    layers.reserve(config.num_hidden_layers);
    for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
        layers.emplace_back(mctx, config.dtype, config.hidden_size, config.num_attention_heads,
                            config.num_attention_heads, config.intermediate_size, config.num_positions, config.norm_eps,
                            config.hidden_act, true, true, false, RopeType::DISABLED, -1,
                            AttentionMaskType::BIDIRECTIONAL, 0, false);
    }
}

ggml_tensor *EVA2CLIPTransformer::forward(ModelContext *mctx, ggml_tensor *hidden_states) const {
    for (const auto &layer : layers) {
        hidden_states = layer.forward(mctx, hidden_states, nullptr, nullptr, 0);
    }
    return hidden_states;
}

ggml_tensor *EVA2CLIPModel::forward(ModelContext *mctx, ggml_tensor *input) const {
    ggml_context *ctx = mctx->ctx_b.get();

    ggml_tensor *hidden_states = patch_embedding.forward(mctx, input);
    hidden_states = transformer.forward(mctx, hidden_states); // [s, hd]

    const int grid_size = std::round(std::sqrt(hidden_states->ne[1] - 1));
    hidden_states = ggml_view_3d(ctx, hidden_states, hidden_states->ne[0], grid_size, grid_size, hidden_states->nb[1],
                                 grid_size * hidden_states->nb[1], hidden_states->nb[1]); // [g, g, hd]
    // TODO: must use this cont?
    hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 2, 0, 1, 3)); // [hd, g, g]
    hidden_states = conv.forward(mctx, hidden_states);                            // [hd, g/2, g/2]
    hidden_states = ggml_reshape_2d(ctx, hidden_states, hidden_states->ne[0] * hidden_states->ne[1],
                                    hidden_states->ne[2]); // [hd, s]
    // TODO: this cont?
    hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 1, 0, 2, 3)); // [s, hd]

    hidden_states = linear_proj.forward(mctx, hidden_states);
    hidden_states = norm1.forward(mctx, hidden_states);
    hidden_states = ggml_gelu_inplace(ctx, hidden_states);
    hidden_states = glu.forward(mctx, hidden_states);

    ggml_tensor *output = ggml_new_tensor_2d(ctx, hidden_states->type, hidden_states->ne[0], hidden_states->ne[1] + 2);

    // concat (boi, x, eoi)
    ggml_tensor *boi_view = ggml_cpy(ctx, boi, ggml_view_1d(ctx, output, hidden_states->ne[0], 0));
    ggml_tensor *hidden_states_view =
        ggml_cpy(ctx, hidden_states,
                 ggml_view_2d(ctx, output, hidden_states->ne[0], hidden_states->ne[1], output->nb[1], output->nb[1]));
    ggml_tensor *eoi_view =
        ggml_cpy(ctx, eoi, ggml_view_1d(ctx, output, hidden_states->ne[0], (hidden_states->ne[1] + 1) * output->nb[1]));
    ggml_build_forward_expand(mctx->gf, boi_view);
    ggml_build_forward_expand(mctx->gf, hidden_states_view);
    ggml_build_forward_expand(mctx->gf, eoi_view);

    output = ggml_scale_inplace(ctx, output, 1.f / scaling_factor);

    return output;
}

ggml_tensor *ChatGLM4VModel::forward_embeddings(ModelContext *mctx, ggml_tensor *input_ids, ggml_tensor *images,
                                                const std::vector<int> &input_ids_vec, int n_past) const {
    ggml_context *ctx = mctx->ctx_b.get();

    ggml_tensor *text_emb = word_embeddings.forward(mctx, input_ids);

    if (!(n_past == 0 && images)) {
        return text_emb;
    }

    CHATGLM_CHECK(images->ne[0] == images->ne[1] && images->ne[2] == 3);

    const int vision_begin =
        std::find(input_ids_vec.begin(), input_ids_vec.end(), config.boi_token_id) - input_ids_vec.begin();
    const int vision_end =
        std::find(input_ids_vec.begin(), input_ids_vec.end(), config.eoi_token_id) - input_ids_vec.begin() + 1;
    CHATGLM_CHECK(vision_begin < (int)input_ids_vec.size() && vision_end <= (int)input_ids_vec.size() &&
                  vision_begin + 3 == vision_end);

    ggml_tensor *vision_emb = vision.forward(mctx, images);

    const int text_tokens = input_ids->ne[0];
    const int total_tokens = text_tokens - 1 + num_vision_tokens();

    ggml_tensor *embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, total_tokens);
    // before boi
    ggml_tensor *text_emb0_view =
        ggml_cpy(ctx, ggml_view_2d(ctx, text_emb, config.hidden_size, vision_begin, text_emb->nb[1], 0),
                 ggml_view_2d(ctx, embeddings, config.hidden_size, vision_begin, embeddings->nb[1], 0));
    // vision tokens
    ggml_tensor *vision_emb_view = ggml_cpy(ctx, vision_emb,
                                            ggml_view_2d(ctx, embeddings, config.hidden_size, vision_emb->ne[1],
                                                         embeddings->nb[1], vision_begin * embeddings->nb[1]));
    // after eoi
    ggml_tensor *text_emb1_view =
        ggml_cpy(ctx,
                 ggml_view_2d(ctx, text_emb, config.hidden_size, text_tokens - vision_end, text_emb->nb[1],
                              vision_end * text_emb->nb[1]),
                 ggml_view_2d(ctx, embeddings, config.hidden_size, text_tokens - vision_end, embeddings->nb[1],
                              (total_tokens - (text_tokens - vision_end)) * embeddings->nb[1]));

    ggml_build_forward_expand(mctx->gf, text_emb0_view);
    ggml_build_forward_expand(mctx->gf, vision_emb_view);
    ggml_build_forward_expand(mctx->gf, text_emb1_view);

    return embeddings;
}

void ChatGLM4VModel::set_graph_inputs(ggml_cgraph *gf, const std::vector<int> &input_ids,
                                      const std::optional<Image> &image, int n_past, int n_ctx) const {
    // set position_ids
    ggml_tensor *position_ids = ggml_graph_get_tensor(gf, "position_ids");
    if (position_ids) {
        std::vector<int> position_ids_buf(position_ids->ne[0]);
        const auto vision_idx =
            std::find(input_ids.begin(), input_ids.end(), config.boi_token_id) - input_ids.begin() + 1;
        if (n_past == 0 && vision_idx < (int)input_ids.size()) {
            std::iota(position_ids_buf.begin(), position_ids_buf.begin() + vision_idx, 0);
            std::fill(position_ids_buf.begin() + vision_idx,
                      position_ids_buf.begin() + vision_idx + num_vision_tokens(), vision_idx);
            std::iota(position_ids_buf.begin() + vision_idx + num_vision_tokens(), position_ids_buf.end(),
                      vision_idx + 1);
        } else {
            const int n_past_text = (n_past > 0) ? input_ids.size() - 1 : 0;
            std::iota(position_ids_buf.begin(), position_ids_buf.end(), n_past_text);
        }
        ggml_backend_tensor_set(position_ids, position_ids_buf.data(), 0, position_ids_buf.size() * sizeof(int));
    }

    // set image
    ggml_tensor *image_tensor = ggml_graph_get_tensor(gf, "image");
    if (image_tensor) {
        // resize
        Image src_resized = image->resize(config.vision.image_size, config.vision.image_size);
        // to float, normalize, channel first
        std::vector<float> pixels_f32(src_resized.pixels.size());
        const size_t num_pixels = src_resized.height * src_resized.width;
        for (size_t i = 0; i < num_pixels; i++) {
            pixels_f32[0 * num_pixels + i] =
                (src_resized.pixels[i * src_resized.channels + 0] / 255.f - 0.48145466f) / 0.26862954f;
            pixels_f32[1 * num_pixels + i] =
                (src_resized.pixels[i * src_resized.channels + 1] / 255.f - 0.45782750f) / 0.26130258f;
            pixels_f32[2 * num_pixels + i] =
                (src_resized.pixels[i * src_resized.channels + 2] / 255.f - 0.40821073f) / 0.27577711f;
        }
        // copy to tensor
        ggml_backend_tensor_set(image_tensor, pixels_f32.data(), 0, ggml_nbytes(image_tensor));
    }
}

int ChatGLM4VForCausalLM::count_tokens(const std::vector<int> &input_ids, const std::optional<Image> &image) const {
    int token_cnt = input_ids.size();
    if (image) {
        token_cnt += transformer.num_vision_tokens() - 1;
    }
    return token_cnt;
}

void ChatGLM4VForCausalLM::load_state_dict(const StateDict &sd) {
    alloc_weight_context(mctx_.get(), sd.buf.get());

    auto self_sd = state_dict();
    ChatGLM2ForCausalLM::load_state_dict(mctx_.get(), self_sd, sd);
}

StateDict ChatGLM4VForCausalLM::state_dict() const {
    StateDict sd;

    // vision
    sd.kv.emplace("transformer.vision.patch_embedding.cls_embedding", transformer.vision.patch_embedding.cls_embedding);
    sd.kv.emplace("transformer.vision.patch_embedding.proj.weight", transformer.vision.patch_embedding.proj.weight);
    sd.kv.emplace("transformer.vision.patch_embedding.proj.bias", transformer.vision.patch_embedding.proj.bias);
    sd.kv.emplace("transformer.vision.patch_embedding.position_embedding.weight",
                  transformer.vision.patch_embedding.position_embedding.weight);

    for (size_t i = 0; i < transformer.vision.transformer.layers.size(); i++) {
        const std::string prefix = "transformer.vision.transformer.layers." + std::to_string(i) + '.';
        sd.kv.emplace(prefix + "input_layernorm.weight",
                      transformer.vision.transformer.layers[i].input_layernorm.weight);
        sd.kv.emplace(prefix + "input_layernorm.bias", transformer.vision.transformer.layers[i].input_layernorm.bias);
        sd.kv.emplace(prefix + "attention.query_key_value.weight",
                      transformer.vision.transformer.layers[i].attention.query_key_value.weight);
        sd.kv.emplace(prefix + "attention.query_key_value.bias",
                      transformer.vision.transformer.layers[i].attention.query_key_value.bias);
        sd.kv.emplace(prefix + "attention.dense.weight",
                      transformer.vision.transformer.layers[i].attention.dense.weight);
        sd.kv.emplace(prefix + "attention.dense.bias", transformer.vision.transformer.layers[i].attention.dense.bias);
        sd.kv.emplace(prefix + "mlp.fc1.weight", transformer.vision.transformer.layers[i].mlp.dense_h_to_4h.weight);
        sd.kv.emplace(prefix + "mlp.fc1.bias", transformer.vision.transformer.layers[i].mlp.dense_h_to_4h.bias);
        sd.kv.emplace(prefix + "mlp.fc2.weight", transformer.vision.transformer.layers[i].mlp.dense_4h_to_h.weight);
        sd.kv.emplace(prefix + "mlp.fc2.bias", transformer.vision.transformer.layers[i].mlp.dense_4h_to_h.bias);
        sd.kv.emplace(prefix + "post_attention_layernorm.weight",
                      transformer.vision.transformer.layers[i].post_attention_layernorm.weight);
        sd.kv.emplace(prefix + "post_attention_layernorm.bias",
                      transformer.vision.transformer.layers[i].post_attention_layernorm.bias);
    }
    sd.kv.emplace("transformer.vision.conv.weight", transformer.vision.conv.weight);
    sd.kv.emplace("transformer.vision.conv.bias", transformer.vision.conv.bias);
    sd.kv.emplace("transformer.vision.linear_proj.linear_proj.weight", transformer.vision.linear_proj.weight);
    sd.kv.emplace("transformer.vision.linear_proj.norm1.weight", transformer.vision.norm1.weight);
    sd.kv.emplace("transformer.vision.linear_proj.norm1.bias", transformer.vision.norm1.bias);
    sd.kv.emplace("transformer.vision.linear_proj.gate_proj.weight", transformer.vision.glu.gate_proj.weight);
    sd.kv.emplace("transformer.vision.linear_proj.dense_h_to_4h.weight", transformer.vision.glu.up_proj.weight);
    sd.kv.emplace("transformer.vision.linear_proj.dense_4h_to_h.weight", transformer.vision.glu.down_proj.weight);
    sd.kv.emplace("transformer.vision.boi", transformer.vision.boi);
    sd.kv.emplace("transformer.vision.eoi", transformer.vision.eoi);

    // text
    sd.kv.emplace("transformer.embedding.word_embeddings.weight", transformer.word_embeddings.weight);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        const std::string layer_prefix = "transformer.encoder.layers." + std::to_string(i) + '.';
        sd.kv.emplace(layer_prefix + "input_layernorm.weight", transformer.layers[i].input_layernorm.weight);
        sd.kv.emplace(layer_prefix + "self_attention.query_key_value.weight",
                      transformer.layers[i].attention.query_key_value.weight);
        sd.kv.emplace(layer_prefix + "self_attention.query_key_value.bias",
                      transformer.layers[i].attention.query_key_value.bias);
        sd.kv.emplace(layer_prefix + "self_attention.dense.weight", transformer.layers[i].attention.dense.weight);
        sd.kv.emplace(layer_prefix + "post_attention_layernorm.weight",
                      transformer.layers[i].post_attention_layernorm.weight);
        sd.kv.emplace(layer_prefix + "mlp.gate_proj.weight", transformer.layers[i].mlp.gate_proj.weight);
        sd.kv.emplace(layer_prefix + "mlp.up_proj.weight", transformer.layers[i].mlp.up_proj.weight);
        // for compatibility
        sd.kv.emplace(layer_prefix + "mlp.dense_4h_to_h.weight", transformer.layers[i].mlp.down_proj.weight);
    }
    sd.kv.emplace("transformer.encoder.final_layernorm.weight", transformer.final_layernorm.weight);
    sd.kv.emplace("transformer.output_layer.weight", lm_head.weight);
    return sd;
}

// ===== pipeline =====

Pipeline::Pipeline(const std::string &path, int max_length) {
    auto _update_config_max_length = [](ModelConfig &config, int max_length) {
        if (max_length > 0) {
            CHATGLM_CHECK(max_length <= config.max_length)
                << "Requested max_length (" << max_length << ") exceeds the max possible model sequence length ("
                << config.max_length << ")";
            config.max_length = max_length;
        }
    };

    mapped_file_ = std::make_unique<MappedFile>(path);
    ModelLoader loader(mapped_file_->data, mapped_file_->size);

    // load magic
    std::string magic = loader.read_string(4);
    CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

    // load model type
    ModelType model_type = (ModelType)loader.read_basic<int>();
    // load version
    int version = loader.read_basic<int>();
    if (model_type == ModelType::CHATGLM) {
        // load config
        ModelConfig config;
        if (version == 1) {
            config = ModelConfig(model_type, loader.read_basic<ModelConfigRecordV1>(), 1e-5f, 10000.f, 0);
        } else if (version == 2) {
            config = ModelConfig(model_type, loader.read_basic<ModelConfigRecordV2>());
        } else {
            CHATGLM_THROW << "only support version 1 or 2 for now but got " << version;
        }
        _update_config_max_length(config, max_length);

        // load tokenizer
        int proto_size = loader.read_basic<int>();
        std::string_view serialized_model_proto((char *)mapped_file_->data + loader.tell(), proto_size);
        loader.seek(proto_size, SEEK_CUR);
        tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto);

        // load model
        model = std::make_unique<ChatGLMForCausalLM>(config);
        StateDict sd = loader.read_state_dict();
        model->load_state_dict(sd);
    } else if (model_type == ModelType::CHATGLM2 || model_type == ModelType::CHATGLM3) {
        // load config
        ModelConfig config;
        if (version == 1) {
            config = ModelConfig(model_type, loader.read_basic<ModelConfigRecordV1GQA>(), 1e-5f, 10000.f, 0);
        } else if (version == 2) {
            config = ModelConfig(model_type, loader.read_basic<ModelConfigRecordV2>());
        } else {
            CHATGLM_THROW << "only support version 1 or 2 for now but got " << version;
        }
        _update_config_max_length(config, max_length);

        // load tokenizer
        int proto_size = loader.read_basic<int>();
        std::string_view serialized_model_proto((char *)mapped_file_->data + loader.tell(), proto_size);
        loader.seek(proto_size, SEEK_CUR);

        if (model_type == ModelType::CHATGLM2) {
            tokenizer = std::make_unique<ChatGLM2Tokenizer>(serialized_model_proto);
            model = std::make_unique<ChatGLM2ForCausalLM>(config);
        } else {
            auto chatglm3_tokenizer = std::make_unique<ChatGLM3Tokenizer>(serialized_model_proto);
            // TODO: read from checkpoint file
            config.extra_eos_token_ids = {chatglm3_tokenizer->observation_token_id, chatglm3_tokenizer->user_token_id};
            tokenizer = std::move(chatglm3_tokenizer);
            model = std::make_unique<ChatGLM3ForCausalLM>(config);
        }

        // load model
        StateDict sd = loader.read_state_dict();
        model->load_state_dict(sd);
    } else if (model_type == ModelType::CHATGLM4 || model_type == ModelType::CHATGLM4V) {
        // load config
        CHATGLM_CHECK(version == 2) << "only support version 2 for now but got " << version;
        ModelConfig config(model_type, loader.read_basic<ModelConfigRecordV2>());
        _update_config_max_length(config, max_length);

        if (model_type == ModelType::CHATGLM4V) {
            config.vision = VisionModelConfig(loader.read_basic<VisionModelConfigRecord>());
        }

        // load tokenizer
        int vocab_text_size = loader.read_basic<int>();
        std::string vocab_text = loader.read_string(vocab_text_size);
        auto chatglm4_tokenizer = std::make_unique<ChatGLM4Tokenizer>(vocab_text);
        config.extra_eos_token_ids = {chatglm4_tokenizer->observation_token_id, chatglm4_tokenizer->user_token_id};
        config.boi_token_id = chatglm4_tokenizer->boi_token_id;
        config.eoi_token_id = chatglm4_tokenizer->eoi_token_id;
        tokenizer = std::move(chatglm4_tokenizer);

        // load model
        if (model_type == ModelType::CHATGLM4V) {
            model = std::make_unique<ChatGLM4VForCausalLM>(config);
        } else {
            model = std::make_unique<ChatGLM4ForCausalLM>(config);
        }
        StateDict sd = loader.read_state_dict();
        model->load_state_dict(sd);
    } else {
        CHATGLM_THROW << "invalid model type " << (int)model_type;
    }
}

std::vector<int> Pipeline::generate(const std::vector<int> &input_ids, const std::optional<Image> &image,
                                    const GenerationConfig &gen_config, BaseStreamer *streamer) const {
    std::vector<int> output_ids = model->generate(input_ids, image, gen_config, streamer);
    std::vector<int> new_output_ids(output_ids.begin() + input_ids.size(), output_ids.end());
    return new_output_ids;
}

std::string Pipeline::generate(const std::string &prompt, const GenerationConfig &gen_config,
                               BaseStreamer *streamer) const {
    std::vector<int> input_ids = tokenizer->encode(prompt, gen_config.max_context_length);
    std::vector<int> new_output_ids = generate(input_ids, std::nullopt, gen_config, streamer);
    std::string output = tokenizer->decode(new_output_ids);
    return output;
}

ChatMessage Pipeline::chat(const std::vector<ChatMessage> &messages, const GenerationConfig &gen_config,
                           BaseStreamer *streamer) const {
    auto it =
        std::find_if(messages.begin(), messages.end(), [](const ChatMessage &msg) { return msg.image.has_value(); });
    const std::optional<Image> image = (it != messages.end()) ? it->image : std::nullopt;
    std::vector<int> input_ids = tokenizer->apply_chat_template(messages, gen_config.max_context_length);
    std::vector<int> new_output_ids = generate(input_ids, image, gen_config, streamer);
    ChatMessage output = tokenizer->decode_message(new_output_ids);
    return output;
}

} // namespace chatglm
