#include "chatglm.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace chatglm {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
    m.doc() = "ChatGLM.cpp python binding";

    py::class_<ChatGLMTokenizer>(m, "ChatGLMTokenizer")
        .def("encode", &ChatGLMTokenizer::encode)
        .def("decode", &ChatGLMTokenizer::decode);

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

    py::class_<ChatGLMConfig>(m, "ChatGLMConfig")
        .def_readonly("vocab_size", &ChatGLMConfig::vocab_size)
        .def_readonly("hidden_size", &ChatGLMConfig::hidden_size)
        .def_readonly("num_attention_heads", &ChatGLMConfig::num_attention_heads)
        .def_readonly("num_layers", &ChatGLMConfig::num_layers)
        .def_readonly("max_sequence_length", &ChatGLMConfig::max_sequence_length)
        .def_readonly("bos_token_id", &ChatGLMConfig::bos_token_id)
        .def_readonly("eos_token_id", &ChatGLMConfig::eos_token_id)
        .def_readonly("gmask_token_id", &ChatGLMConfig::gmask_token_id)
        .def_readonly("mask_token_id", &ChatGLMConfig::mask_token_id)
        .def_readonly("pad_token_id", &ChatGLMConfig::pad_token_id)
        .def_readonly("dtype", &ChatGLMConfig::dtype);

    py::class_<ChatGLMForConditionalGeneration>(m, "ChatGLMForConditionalGeneration")
        .def_readonly("config", &ChatGLMForConditionalGeneration::config)
        .def("generate_next_token", &ChatGLMForConditionalGeneration::generate_next_token);

    py::class_<ChatGLMPipeline>(m, "ChatGLMPipeline")
        .def(py::init<const std::string &>())
        .def_property_readonly("model", [](const ChatGLMPipeline &self) { return self.model.get(); })
        .def_property_readonly("tokenizer", [](const ChatGLMPipeline &self) { return self.tokenizer.get(); })
        .def_static("build_prompt", &ChatGLMPipeline::build_prompt);
}

} // namespace chatglm
