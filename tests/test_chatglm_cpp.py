from pathlib import Path

import chatglm_cpp
import pytest

CHATGLM_MODEL_PATH = Path(__file__).resolve().parent / "chatglm-ggml.bin"
CHATGLM2_MODEL_PATH = Path(__file__).resolve().parent / "chatglm2-ggml.bin"


def test_chatglm_version():
    print(chatglm_cpp.__version__)


@pytest.mark.skipif(not CHATGLM_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm_pipeline():
    history = ["你好"]
    target = "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"

    pipeline = chatglm_cpp.Pipeline(CHATGLM_MODEL_PATH)
    output = pipeline.chat(history, do_sample=False)
    assert output == target

    stream_output = pipeline.stream_chat(history, do_sample=False)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not CHATGLM2_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm2_pipeline():
    history = ["你好"]
    target = "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"

    pipeline = chatglm_cpp.Pipeline(CHATGLM2_MODEL_PATH)
    output = pipeline.chat(history, do_sample=False)
    assert output == target

    stream_output = pipeline.stream_chat(history, do_sample=False)
    stream_output = "".join(stream_output)
    assert stream_output == target
