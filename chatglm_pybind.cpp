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
};

class PyBaseModelForCausalLM : public BaseModelForCausalLM {
  public:
    using BaseModelForCausalLM::BaseModelForCausalLM;

    void load(ModelLoader &loader) override { PYBIND11_OVERLOAD_PURE(void, PyBaseModelForCausalLM, load, loader); }
    ggml_tensor *forward(ModelContext *ctx, ggml_tensor *input_ids, int n_past, int n_ctx) const override {
        PYBIND11_OVERLOAD_PURE(ggml_tensor *, PyBaseModelForCausalLM, forward, ctx, input_ids, n_past, n_ctx)
    }
};

PYBIND11_MODULE(_C, m) {
    m.doc() = "ChatGLM.cpp python binding";

    py::class_<ModelConfig>(m, "ModelConfig")
        .def_readonly("model_type", &ModelConfig::model_type)
        .def_readonly("dtype", &ModelConfig::dtype)
        .def_readonly("vocab_size", &ModelConfig::vocab_size)
        .def_readonly("hidden_size", &ModelConfig::hidden_size)
        .def_readonly("num_attention_heads", &ModelConfig::num_attention_heads)
        .def_readonly("num_kv_heads", &ModelConfig::num_kv_heads)
        .def_readonly("num_hidden_layers", &ModelConfig::num_hidden_layers)
        .def_readonly("intermediate_size", &ModelConfig::intermediate_size)
        .def_readonly("norm_eps", &ModelConfig::norm_eps)
        .def_readonly("max_length", &ModelConfig::max_length)
        .def_readonly("bos_token_id", &ModelConfig::bos_token_id)
        .def_readonly("eos_token_id", &ModelConfig::eos_token_id)
        .def_readonly("pad_token_id", &ModelConfig::pad_token_id)
        .def_readonly("sep_token_id", &ModelConfig::sep_token_id)
        .def_property_readonly("model_type_name", &ModelConfig::model_type_name);

    py::class_<BaseTokenizer, PyBaseTokenizer>(m, "BaseTokenizer")
        .def("encode", &BaseTokenizer::encode)
        .def("decode", &BaseTokenizer::decode)
        .def("encode_history", &BaseTokenizer::encode_history);

    py::class_<BaseModelForCausalLM, PyBaseModelForCausalLM>(m, "BaseModelForCausalLM")
        .def("generate_next_token", &BaseModelForCausalLM::generate_next_token)
        .def_readonly("config", &BaseModelForCausalLM::config);

    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<int, int, bool, int, float, float, float, int>(), "max_length"_a = 2048,
             "max_context_length"_a = 512, "do_sample"_a = true, "top_k"_a = 0, "top_p"_a = 0.7, "temperature"_a = 0.95,
             "repetition_penalty"_a = 1.0, "num_threads"_a = 0)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("max_context_length", &GenerationConfig::max_context_length)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("num_threads", &GenerationConfig::num_threads);

    // ===== ChatGLM =====

    py::class_<ChatGLMTokenizer, BaseTokenizer>(m, "ChatGLMTokenizer");

    py::class_<ChatGLMForCausalLM, BaseModelForCausalLM>(m, "ChatGLMForCausalLM");

    // ===== ChatGLM2 =====

    py::class_<ChatGLM2Tokenizer, BaseTokenizer>(m, "ChatGLM2Tokenizer");

    py::class_<ChatGLM2ForCausalLM, BaseModelForCausalLM>(m, "ChatGLM2ForCausalLM");

    // ===== ChatGLM3 =====

    py::class_<ChatGLM3Tokenizer, BaseTokenizer>(m, "ChatGLM3Tokenizer");

    // ===== Baichuan7B/13B =====

    py::class_<BaichuanTokenizer, BaseTokenizer>(m, "BaichuanTokenizer");

    py::class_<Baichuan7BForCausalLM, BaseModelForCausalLM>(m, "Baichuan7BForCausalLM");

    py::class_<Baichuan13BForCausalLM, BaseModelForCausalLM>(m, "Baichuan13BForCausalLM");

    // ===== InternLM =====

    py::class_<InternLMTokenizer, BaseTokenizer>(m, "InternLMTokenizer");

    py::class_<InternLM7BForCausalLM, BaseModelForCausalLM>(m, "InternLM7BForCausalLM");

    py::class_<InternLM20BForCausalLM, BaseModelForCausalLM>(m, "InternLM20BForCausalLM");

    // ===== Pipeline ====

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<const std::string &>())
        .def_property_readonly("model", [](const Pipeline &self) { return self.model.get(); })
        .def_property_readonly("tokenizer", [](const Pipeline &self) { return self.tokenizer.get(); });
}

} // namespace chatglm
