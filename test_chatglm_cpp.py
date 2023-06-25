from pathlib import Path

import chatglm_cpp
import pytest

MODEL_PATH = Path(__file__).resolve().parent / "chatglm-ggml.bin"


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="model file not found")
def test_chatglm_pipeline():
    pipeline = chatglm_cpp.ChatGLMPipeline(MODEL_PATH)
    history = ["你好"]
    output = pipeline.chat(history, do_sample=False)
    assert output == "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"

    stream_output = pipeline.stream_chat(history, do_sample=False)
    stream_output = "".join(stream_output)
    assert stream_output == "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"
