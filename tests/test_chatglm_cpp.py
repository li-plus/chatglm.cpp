from pathlib import Path

import chatglm_cpp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHATGLM_MODEL_PATH = PROJECT_ROOT / "chatglm-ggml.bin"
CHATGLM2_MODEL_PATH = PROJECT_ROOT / "chatglm2-ggml.bin"
CHATGLM3_MODEL_PATH = PROJECT_ROOT / "chatglm3-ggml.bin"
CODEGEEX2_MODEL_PATH = PROJECT_ROOT / "codegeex2-ggml.bin"
BAICHUAN13B_MODEL_PATH = PROJECT_ROOT / "baichuan-13b-chat-ggml.bin"
BAICHUAN2_7B_MODEL_PATH = PROJECT_ROOT / "baichuan2-7b-chat-ggml.bin"
BAICHUAN2_13B_MODEL_PATH = PROJECT_ROOT / "baichuan2-13b-chat-ggml.bin"
INTERNLM7B_MODEL_PATH = PROJECT_ROOT / "internlm-chat-7b-ggml.bin"
INTERNLM20B_MODEL_PATH = PROJECT_ROOT / "internlm-chat-20b-ggml.bin"


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

    stream_output = pipeline.chat(history, do_sample=False, stream=True)
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

    stream_output = pipeline.chat(history, do_sample=False, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not CHATGLM3_MODEL_PATH.exists(), reason="model file not found")
def test_chatglm3_pipeline():
    history = ["你好"]
    target = "你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。"

    pipeline = chatglm_cpp.Pipeline(CHATGLM3_MODEL_PATH)
    output = pipeline.chat(history, do_sample=False)
    assert output == target

    stream_output = pipeline.stream_chat(history, do_sample=False)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, do_sample=False, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not CODEGEEX2_MODEL_PATH.exists(), reason="model file not found")
def test_codegeex2_pipeline():
    prompt = "# language: Python\n# write a bubble sort function\n"
    target = """

def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list


print(bubble_sort([5, 4, 3, 2, 1]))"""

    pipeline = chatglm_cpp.Pipeline(CODEGEEX2_MODEL_PATH)
    output = pipeline.generate(prompt, do_sample=False)
    assert output == target

    stream_output = pipeline.generate(prompt, do_sample=False, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not BAICHUAN13B_MODEL_PATH.exists(), reason="model file not found")
def test_baichuan13b_pipeline():
    history = ["你好呀"]
    target = "你好！很高兴见到你。请问有什么我可以帮助你的吗？"

    gen_kwargs = dict(do_sample=False, repetition_penalty=1.1)

    pipeline = chatglm_cpp.Pipeline(BAICHUAN13B_MODEL_PATH)
    output = pipeline.chat(history, **gen_kwargs)
    assert output == target

    stream_output = pipeline.stream_chat(history, **gen_kwargs)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, **gen_kwargs, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not BAICHUAN2_7B_MODEL_PATH.exists(), reason="model file not found")
def test_baichuan2_7b_pipeline():
    history = ["你好呀"]
    target = "你好！很高兴为你服务。请问有什么问题我可以帮助你解决？"

    gen_kwargs = dict(do_sample=False, repetition_penalty=1.05)

    pipeline = chatglm_cpp.Pipeline(BAICHUAN2_7B_MODEL_PATH)
    output = pipeline.chat(history, **gen_kwargs)
    assert output == target

    stream_output = pipeline.stream_chat(history, **gen_kwargs)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, **gen_kwargs, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not BAICHUAN2_13B_MODEL_PATH.exists(), reason="model file not found")
def test_baichuan2_13b_pipeline():
    history = ["你好呀"]
    target = "你好！很高兴见到你。请问有什么我可以帮助你的吗？"

    gen_kwargs = dict(do_sample=False, repetition_penalty=1.05)

    pipeline = chatglm_cpp.Pipeline(BAICHUAN2_13B_MODEL_PATH)
    output = pipeline.chat(history, **gen_kwargs)
    assert output == target

    stream_output = pipeline.stream_chat(history, **gen_kwargs)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, **gen_kwargs, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not INTERNLM7B_MODEL_PATH.exists(), reason="model file not found")
def test_internlm7b_pipeline():
    history = ["你好"]
    target = "你好，有什么我可以帮助你的吗？"

    gen_kwargs = dict(do_sample=False)

    pipeline = chatglm_cpp.Pipeline(INTERNLM7B_MODEL_PATH)
    output = pipeline.chat(history, **gen_kwargs)
    assert output == target

    stream_output = pipeline.stream_chat(history, **gen_kwargs)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, **gen_kwargs, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target


@pytest.mark.skipif(not INTERNLM20B_MODEL_PATH.exists(), reason="model file not found")
def test_internlm20b_pipeline():
    history = ["你好"]
    target = "你好！有什么我可以帮助你的吗？"

    gen_kwargs = dict(do_sample=False)

    pipeline = chatglm_cpp.Pipeline(INTERNLM20B_MODEL_PATH)
    output = pipeline.chat(history, **gen_kwargs)
    assert output == target

    stream_output = pipeline.stream_chat(history, **gen_kwargs)
    stream_output = "".join(stream_output)
    assert stream_output == target

    stream_output = pipeline.chat(history, **gen_kwargs, stream=True)
    stream_output = "".join(stream_output)
    assert stream_output == target
