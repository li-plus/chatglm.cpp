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
    std::string decode(const std::vector<int> &ids) const override {
        PYBIND11_OVERLOAD_PURE(std::string, BaseTokenizer, decode, ids);
    }
    std::vector<int> encode_history(const std::vector<std::string> &history, int max_length) const override {
        PYBIND11_OVERLOAD_PURE(std::vector<int>, BaseTokenizer, encode_history, history, max_length);
    }
    std::string build_prompt(const std::vector<std::string> &history) const override {
        PYBIND11_OVERLOAD_PURE(std::string, BaseTokenizer, build_prompt, history);
    }
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "ChatGLM.cpp python binding";

    py::class_<BaseConfig>(m, "BaseConfig")
        .def_readonly("dtype", &BaseConfig::dtype)
        .def_readonly("vocab_size", &BaseConfig::vocab_size)
        .def_readonly("hidden_size", &BaseConfig::hidden_size)
        .def_readonly("num_attention_heads", &BaseConfig::num_attention_heads)
        .def_readonly("num_hidden_layers", &BaseConfig::num_hidden_layers)
        .def_readonly("intermediate_size", &BaseConfig::intermediate_size)
        .def_readonly("max_length", &BaseConfig::max_length)
        .def_readonly("bos_token_id", &BaseConfig::bos_token_id)
        .def_readonly("eos_token_id", &BaseConfig::eos_token_id)
        .def_readonly("pad_token_id", &BaseConfig::pad_token_id)
        .def_readonly("sep_token_id", &BaseConfig::sep_token_id);

    py::class_<BaseTokenizer, PyBaseTokenizer>(m, "BaseTokenizer")
        .def("encode", &BaseTokenizer::encode)
        .def("decode", &BaseTokenizer::decode)
        .def("encode_history", &BaseTokenizer::encode_history)
        .def("build_prompt", &BaseTokenizer::build_prompt);

    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<int, int, bool, int, float, float, int>(), "max_length"_a = 2048, "max_context_length"_a = 512,
             "do_sample"_a = true, "top_k"_a = 0, "top_p"_a = 0.7, "temperature"_a = 0.95, "num_threads"_a = 0)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("max_context_length", &GenerationConfig::max_context_length)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("num_threads", &GenerationConfig::num_threads);

    py::class_<ChatGLMConfig, BaseConfig>(m, "ChatGLMConfig");

    py::class_<ChatGLMTokenizer, BaseTokenizer>(m, "ChatGLMTokenizer");

    py::class_<ChatGLMForConditionalGeneration>(m, "ChatGLMForConditionalGeneration")
        .def_readonly("config", &ChatGLMForConditionalGeneration::config)
        .def_property_readonly("type_name", &ChatGLMForConditionalGeneration::type_name)
        .def("generate_next_token", &ChatGLMForConditionalGeneration::generate_next_token);

    py::class_<ChatGLM2Config, BaseConfig>(m, "ChatGLM2Config")
        .def_readonly("num_kv_heads", &ChatGLM2Config::num_kv_heads);

    py::class_<ChatGLM2Tokenizer, BaseTokenizer>(m, "ChatGLM2Tokenizer");

    py::class_<ChatGLM2ForConditionalGeneration>(m, "ChatGLM2ForConditionalGeneration")
        .def_readonly("config", &ChatGLM2ForConditionalGeneration::config)
        .def_property_readonly("type_name", &ChatGLM2ForConditionalGeneration::type_name)
        .def("generate_next_token", &ChatGLM2ForConditionalGeneration::generate_next_token);

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<const std::string &>())
        .def_property_readonly("model", [](const Pipeline &self) { return self.model.get(); })
        .def_property_readonly("tokenizer", [](const Pipeline &self) { return self.tokenizer.get(); });
}

} // namespace chatglm
