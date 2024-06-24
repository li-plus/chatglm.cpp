from pathlib import Path

import chatglm_cpp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHATGLM_MODEL_PATH = PROJECT_ROOT / "models/chatglm-ggml.bin"
CHATGLM2_MODEL_PATH = PROJECT_ROOT / "models/chatglm2-ggml.bin"
CHATGLM3_MODEL_PATH = PROJECT_ROOT / "models/chatglm3-ggml.bin"
CHATGLM4_MODEL_PATH = PROJECT_ROOT / "models/chatglm4-ggml.bin"
CODEGEEX2_MODEL_PATH = PROJECT_ROOT / "models/codegeex2-ggml.bin"


def test_chatglm_version():
    print(chatglm_cpp.__version__)


def check_pipeline(model_path, prompt, target, gen_kwargs={}):
    messages = [chatglm_cpp.ChatMessage(role="user", content=prompt)]

    pipeline = chatglm_cpp.Pipeline(model_path)
    output = pipeline.chat(messages, do_sample=False, **gen_kwargs).content
    assert output == target

    stream_output = pipeline.chat(messages, do_sample=False, stream=True, **gen_kwargs)
    stream_output = "".join([msg.content for msg in stream_output])
    if model_path in (CHATGLM3_MODEL_PATH, CHATGLM4_MODEL_PATH):
        # hack for ChatGLM3/4
        stream_output = stream_output.strip()
    assert stream_output == target


@pytest.mark.skipif(not CHATGLM_MODEL_PATH.exists(), reason="model file not found")
def test_pipeline_options():
    # check max_length option
    pipeline = chatglm_cpp.Pipeline(CHATGLM_MODEL_PATH)
    assert pipeline.model.config.max_length == 2048
    pipeline = chatglm_cpp.Pipeline(CHATGLM_MODEL_PATH, max_length=234)
    assert pipeline.model.config.max_length == 234

    # check if resources are properly released
    # for _ in range(100):
    #     chatglm_cpp.Pipeline(CHATGLM_MODEL_PATH)


@pytest.mark.skipif(not CHATGLM_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm_pipeline():
    check_pipeline(
        model_path=CHATGLM_MODEL_PATH,
        prompt="你好",
        target="你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
    )


@pytest.mark.skipif(not CHATGLM2_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm2_pipeline():
    check_pipeline(
        model_path=CHATGLM2_MODEL_PATH,
        prompt="你好",
        target="你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。",
    )


@pytest.mark.skipif(not CHATGLM3_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm3_pipeline():
    check_pipeline(
        model_path=CHATGLM3_MODEL_PATH,
        prompt="你好",
        target="你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。",
    )


@pytest.mark.skipif(not CHATGLM4_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm4_pipeline():
    check_pipeline(
        model_path=CHATGLM4_MODEL_PATH,
        prompt="你好",
        target="你好👋！我是人工智能助手，很高兴见到你，有什么可以帮助你的吗？",
    )


@pytest.mark.skipif(not CODEGEEX2_MODEL_PATH.exists(), reason="model file not found")
def test_codegeex2_pipeline():
    prompt = "# language: Python\n# write a bubble sort function\n"
    target = """

def bubble_sort(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


print(bubble_sort([5, 4, 3, 2, 1]))"""

    pipeline = chatglm_cpp.Pipeline(CODEGEEX2_MODEL_PATH)
    output = pipeline.generate(prompt, do_sample=False)
    assert output == target

    stream_output = pipeline.generate(prompt, do_sample=False, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not CHATGLM4_MODEL_PATH.exists(), reason="model file not found")
def test_langchain_api():
    import os
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    with patch.dict(os.environ, {"MODEL": str(CHATGLM4_MODEL_PATH)}):
        from chatglm_cpp.langchain_api import app

    client = TestClient(app)
    response = client.post("/", json={"prompt": "你好", "temperature": 0})
    assert response.status_code == 200
    assert response.json()["response"] == "你好👋！我是人工智能助手，很高兴见到你，有什么可以帮助你的吗？"


@pytest.mark.skipif(not CHATGLM4_MODEL_PATH.exists(), reason="model file not found")
def test_openai_api():
    import os
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    with patch.dict(os.environ, {"MODEL": str(CHATGLM4_MODEL_PATH)}):
        from chatglm_cpp.openai_api import app

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "你好"}], "temperature": 0}
    )
    assert response.status_code == 200
    response_message = response.json()["choices"][0]["message"]
    assert response_message["role"] == "assistant"
    assert response_message["content"] == "你好👋！我是人工智能助手，很高兴见到你，有什么可以帮助你的吗？"
