#include "chatglm.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace chatglm {

namespace py = pybind11;
using namespace pybind11::literals;

class PyBaseTokenizer : public BaseTokenizer {
  public:
    using BaseTokenizer::BaseTokenizer;

    std::vector<int> encode(const std::string &text, int max_length) const override {
        PYBIND11_OVERRIDE_PURE(std::vector<int>, BaseTokenizer, encode, text, max_length);
    }
    std::string decode(const std::vector<int> &ids, bool skip_special_tokens) const override {
        PYBIND11_OVERLOAD_PURE(std::string, BaseTokenizer, decode, ids, skip_special_tokens);
    }
    std::vector<int> apply_chat_template(const std::vector<ChatMessage> &messages, int max_length) const override {
        PYBIND11_OVERLOAD_PURE(std::vector<int>, BaseTokenizer, apply_chat_template, messages, max_length);
    }
};

class PyBaseModelForCausalLM : public BaseModelForCausalLM {
  public:
    using BaseModelForCausalLM::BaseModelForCausalLM;

    void load_state_dict(const StateDict &sd) override {
        PYBIND11_OVERLOAD_PURE(void, PyBaseModelForCausalLM, load_state_dict, sd);
    }

    ggml_tensor *forward(ModelContext *mctx, ggml_tensor *input_ids, ggml_tensor *images,
                         const std::vector<int> &input_ids_vec, int n_past, bool is_decoding) const override {
        PYBIND11_OVERLOAD_PURE(ggml_tensor *, PyBaseModelForCausalLM, forward, mctx, input_ids, images, input_ids_vec,
                               n_past, is_decoding);
    }

    void set_graph_inputs(const std::vector<int> &input_ids, const std::optional<Image> &image, int n_past,
                          int n_ctx) const override {
        PYBIND11_OVERLOAD_PURE(void, PyBaseModelForCausalLM, set_graph_inputs, input_ids, image, n_past, n_ctx);
    }

    int count_tokens(const std::vector<int> &input_ids, const std::optional<Image> &image) const override {
        PYBIND11_OVERLOAD_PURE(int, PyBaseModelForCausalLM, count_tokens, input_ids, image);
    }
};

template <typename T>
static inline std::string to_string(const T &obj) {
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "ChatGLM.cpp python binding";

    py::enum_<ModelType>(m, "ModelType")
        .value("CHATGLM", ModelType::CHATGLM)
        .value("CHATGLM2", ModelType::CHATGLM2)
        .value("CHATGLM3", ModelType::CHATGLM3)
        .value("CHATGLM4", ModelType::CHATGLM4);

    py::class_<VisionModelConfig>(m, "VisionModelConfig")
        // .def_readonly("dtype", &VisionModelConfig::dtype)
        // .def_readonly("hidden_act", &VisionModelConfig::hidden_act)
        .def_readonly("hidden_size", &VisionModelConfig::hidden_size)
        .def_readonly("image_size", &VisionModelConfig::image_size)
        .def_readonly("in_channels", &VisionModelConfig::in_channels)
        .def_readonly("intermediate_size", &VisionModelConfig::intermediate_size)
        .def_readonly("norm_eps", &VisionModelConfig::norm_eps)
        .def_readonly("num_attention_heads", &VisionModelConfig::num_attention_heads)
        .def_readonly("num_hidden_layers", &VisionModelConfig::num_hidden_layers)
        .def_readonly("num_positions", &VisionModelConfig::num_positions)
        .def_readonly("patch_size", &VisionModelConfig::patch_size)
        .def_readonly("scaling_factor", &VisionModelConfig::scaling_factor);

    py::class_<ModelConfig>(m, "ModelConfig")
        .def_readonly("model_type", &ModelConfig::model_type)
        // .def_readonly("dtype", &ModelConfig::dtype)
        .def_readonly("vocab_size", &ModelConfig::vocab_size)
        .def_readonly("hidden_size", &ModelConfig::hidden_size)
        .def_readonly("num_attention_heads", &ModelConfig::num_attention_heads)
        .def_readonly("num_key_value_heads", &ModelConfig::num_key_value_heads)
        .def_readonly("num_hidden_layers", &ModelConfig::num_hidden_layers)
        .def_readonly("intermediate_size", &ModelConfig::intermediate_size)
        .def_readonly("norm_eps", &ModelConfig::norm_eps)
        .def_readonly("max_length", &ModelConfig::max_length)
        .def_readonly("bos_token_id", &ModelConfig::bos_token_id)
        .def_readonly("eos_token_id", &ModelConfig::eos_token_id)
        .def_readonly("pad_token_id", &ModelConfig::pad_token_id)
        .def_readonly("sep_token_id", &ModelConfig::sep_token_id)
        .def_readonly("extra_eos_token_ids", &ModelConfig::extra_eos_token_ids)
        .def_readonly("vision", &ModelConfig::vision)
        .def_property_readonly("model_type_name", &ModelConfig::model_type_name);

    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<int, int, int, bool, int, float, float, float>(), "max_length"_a = 2048, "max_new_tokens"_a = -1,
             "max_context_length"_a = 512, "do_sample"_a = true, "top_k"_a = 0, "top_p"_a = 0.7, "temperature"_a = 0.95,
             "repetition_penalty"_a = 1.0)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_context_length", &GenerationConfig::max_context_length)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty);

    py::class_<FunctionMessage>(m, "FunctionMessage")
        .def("__repr__", &to_string<FunctionMessage>)
        .def("__str__", &to_string<FunctionMessage>)
        .def_readwrite("name", &FunctionMessage::name)
        .def_readwrite("arguments", &FunctionMessage::arguments);

    py::class_<CodeMessage>(m, "CodeMessage")
        .def("__repr__", &to_string<CodeMessage>)
        .def("__str__", &to_string<CodeMessage>)
        .def_readwrite("input", &CodeMessage::input);

    py::class_<ToolCallMessage>(m, "ToolCallMessage")
        .def("__repr__", &to_string<ToolCallMessage>)
        .def("__str__", &to_string<ToolCallMessage>)
        .def_readwrite("type", &ToolCallMessage::type)
        .def_readwrite("function", &ToolCallMessage::function)
        .def_readwrite("code", &ToolCallMessage::code);

    py::class_<Image>(m, "Image", py::buffer_protocol())
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            CHATGLM_CHECK(info.format == py::format_descriptor<uint8_t>::format())
                << "Incompatible format: expect a byte array!";
            CHATGLM_CHECK(info.ndim == 3 && info.shape[2] == 3) << "Only support RGB image for now";

            for (int i = 1; i < info.ndim; i++) {
                CHATGLM_CHECK(info.strides[i] * info.shape[i] == info.strides[i - 1])
                    << "Only support contiguous array";
            }

            return Image(info.shape[1], info.shape[0], info.shape[2], (uint8_t *)info.ptr);
        }))
        .def_buffer([](Image &self) {
            return py::buffer_info(
                self.pixels.data(), sizeof(uint8_t), py::format_descriptor<uint8_t>::format(), 3,
                {self.height, self.width, self.channels},
                {self.width * self.channels * sizeof(uint8_t), self.channels * sizeof(uint8_t), sizeof(uint8_t)});
        })
        .def("__repr__", &to_string<Image>)
        .def("__str__", &to_string<Image>)
        .def_readonly("width", &Image::width)
        .def_readonly("height", &Image::height)
        .def_readonly("channels", &Image::channels)
        .def_readonly("pixels", &Image::pixels);

    py::class_<ChatMessage>(m, "ChatMessage")
        .def(py::init<std::string, std::string, std::optional<Image>, std::vector<ToolCallMessage>>(), "role"_a,
             "content"_a, "image"_a = std::nullopt, "tool_calls"_a = std::vector<ToolCallMessage>{})
        .def("__repr__", &to_string<ChatMessage>)
        .def("__str__", &to_string<ChatMessage>)
        .def_readonly_static("ROLE_SYSTEM", &ChatMessage::ROLE_SYSTEM)
        .def_readonly_static("ROLE_USER", &ChatMessage::ROLE_USER)
        .def_readonly_static("ROLE_ASSISTANT", &ChatMessage::ROLE_ASSISTANT)
        .def_readonly_static("ROLE_OBSERVATION", &ChatMessage::ROLE_OBSERVATION)
        .def_readwrite("role", &ChatMessage::role)
        .def_readwrite("content", &ChatMessage::content)
        .def_readwrite("image", &ChatMessage::image)
        .def_readwrite("tool_calls", &ChatMessage::tool_calls);

    py::class_<BaseTokenizer, PyBaseTokenizer>(m, "BaseTokenizer")
        .def("encode", &BaseTokenizer::encode, "text"_a, "max_length"_a)
        .def("decode", &BaseTokenizer::decode, "ids"_a, "skip_special_tokens"_a = true)
        .def("apply_chat_template", &BaseTokenizer::apply_chat_template, "messages"_a, "max_length"_a)
        .def("decode_message", &BaseTokenizer::decode_message, "ids"_a);

    py::class_<BaseModelForCausalLM, PyBaseModelForCausalLM>(m, "BaseModelForCausalLM")
        .def("generate_next_token", &BaseModelForCausalLM::generate_next_token, "input_ids"_a, "image"_a,
             "gen_config"_a, "n_past"_a, "n_ctx"_a)
        .def("count_tokens", &BaseModelForCausalLM::count_tokens, "input_ids"_a, "image"_a)
        .def_readonly("config", &BaseModelForCausalLM::config);

    // ===== ChatGLM =====

    py::class_<ChatGLMTokenizer, BaseTokenizer>(m, "ChatGLMTokenizer");

    py::class_<ChatGLMForCausalLM, BaseModelForCausalLM>(m, "ChatGLMForCausalLM");

    // ===== ChatGLM2 =====

    py::class_<ChatGLM2Tokenizer, BaseTokenizer>(m, "ChatGLM2Tokenizer");

    py::class_<ChatGLM2ForCausalLM, BaseModelForCausalLM>(m, "ChatGLM2ForCausalLM");

    // ===== ChatGLM3 =====

    py::class_<ChatGLM3Tokenizer, BaseTokenizer>(m, "ChatGLM3Tokenizer");

    // ===== ChatGLM4 =====

    py::class_<ChatGLM4Tokenizer, BaseTokenizer>(m, "ChatGLM4Tokenizer");

    // ===== Pipeline ====

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<const std::string &, int>(), "path"_a, "max_length"_a = -1)
        .def_property_readonly("model", [](const Pipeline &self) { return self.model.get(); })
        .def_property_readonly("tokenizer", [](const Pipeline &self) { return self.tokenizer.get(); });
}

} // namespace chatglm
