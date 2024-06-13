from __future__ import annotations

import base64
import functools
import io
import json
import queue
import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Callable

import chatglm_cpp
import jupyter_client
import streamlit as st
from PIL import Image

IPYKERNEL = "chatglm3-demo"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm3-ggml.bin"

CHAT_SYSTEM_PROMPT = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."

TOOLS = [
    {
        "name": "random_number_generator",
        "description": "Generates a random number x, s.t. range[0] <= x < range[1]",
        "parameters": {
            "type": "object",
            "properties": {
                "seed": {"description": "The random seed used by the generator", "type": "integer"},
                "range": {
                    "description": "The range of the generated numbers",
                    "type": "array",
                    "items": [{"type": "integer"}, {"type": "integer"}],
                },
            },
            "required": ["seed", "range"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for `city_name`",
        "parameters": {
            "type": "object",
            "properties": {"city_name": {"description": "The name of the city to be queried", "type": "string"}},
            "required": ["city_name"],
        },
    },
]

TOOL_SYSTEM_PROMPT = (
    "Answer the following questions as best as you can. You have access to the following tools:\n"
    + json.dumps(TOOLS, indent=4)
)

CI_SYSTEM_PROMPT = "你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。"


class Mode(str, Enum):
    CHAT = "💬 Chat"
    TOOL = "🛠️ Tool"
    CI = "🧑‍💻 Code Interpreter"


@st.cache_resource
def get_model(model_path: str) -> chatglm_cpp.Pipeline:
    return chatglm_cpp.Pipeline(model_path)


class Message(chatglm_cpp.ChatMessage):
    def __init__(
        self, role: str, content: str, tool_calls: list | None = None, image: Image.Image | None = None
    ) -> None:
        if tool_calls is None:
            tool_calls = []
        super().__init__(role, content, tool_calls)
        self.image = image

    @staticmethod
    def from_cpp(cpp_message: chatglm_cpp.ChatMessage) -> Message:
        return Message(
            role=cpp_message.role, content=cpp_message.content, tool_calls=cpp_message.tool_calls, image=None
        )


def show_message(message: Message) -> None:
    role_avatars = {"user": "user", "observation": "user", "assistant": "assistant"}
    avatar = role_avatars.get(message.role)
    if avatar is None:
        st.error(f"Unexpected message role {message.role}")
        return

    display_content = message.content
    if message.tool_calls:
        (tool_call,) = message.tool_calls
        if tool_call.type == "function":
            display_content = f"{tool_call.function.name}\n{display_content}"
        elif tool_call.type == "code":
            display_content += "\n" + tool_call.code.input

    if message.role == "observation":
        display_content = f"```\n{display_content.strip()}\n```"

    with st.chat_message(name=message.role, avatar=avatar):
        if message.image:
            st.image(message.image)
        else:
            st.markdown(display_content)


# ----- begin function call -----

_FUNCTION_REGISTRY = {}


def register_function(func: Callable) -> Callable:
    _FUNCTION_REGISTRY[func.__name__] = func

    @functools.wraps(func)
    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    return wrap


@register_function
def random_number_generator(seed: int, range: tuple[int, int]) -> int:
    import random

    return random.Random(seed).randint(*range)


@register_function
def get_weather(city_name: str) -> str:
    import requests

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
    resp.raise_for_status()
    resp = resp.json()

    ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    return json.dumps(ret)


def run_function(name: str, arguments: str) -> str:
    def tool_call(**kwargs):
        return kwargs

    func = _FUNCTION_REGISTRY.get(name)
    if func is None:
        return f"Function `{name}` is not defined"

    try:
        kwargs = eval(arguments, dict(tool_call=tool_call))
    except Exception:
        return f"Invalid arguments {arguments}"

    try:
        return str(func(**kwargs))
    except Exception:
        return traceback.format_exc()


# ----- end function call -----

# ----- begin code interpreter -----


@st.cache_resource
def get_kernel_client(kernel_name) -> jupyter_client.BlockingKernelClient:
    km = jupyter_client.KernelManager(kernel_name=kernel_name)
    km.start_kernel()

    kc: jupyter_client.BlockingKernelClient = km.blocking_client()
    kc.start_channels()

    return kc


def clean_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r"(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def extract_code(text: str) -> str:
    return re.search(r"```.*?\n(.*?)```", text, re.DOTALL)[1]


def run_code(kc: jupyter_client.BlockingKernelClient, code: str) -> str | Image.Image:
    kc.execute(code)

    try:
        shell_msg = kc.get_shell_msg(timeout=30)
        io_msg_content = None
        while True:
            try:
                next_io_msg_content = kc.get_iopub_msg(timeout=30)["content"]
            except queue.Empty:
                break
            if next_io_msg_content.get("execution_state") == "idle":
                break
            io_msg_content = next_io_msg_content

        if shell_msg["metadata"]["status"] == "timeout":
            return "Execution Timeout Expired"

        if shell_msg["metadata"]["status"] == "error":
            try:
                traceback_content = clean_ansi_codes(io_msg_content["traceback"][-1])
            except Exception:
                traceback_content = "Traceback Error"
            return traceback_content

        if "text" in io_msg_content:
            return io_msg_content["text"]

        data_content = io_msg_content.get("data")
        if data_content is not None:
            image_content = data_content.get("image/png")
            if image_content is not None:
                return Image.open(io.BytesIO(base64.b64decode(image_content)))

            text_content = data_content.get("text/plain")
            if text_content is not None:
                return text_content

        return ""

    except Exception:
        return traceback.format_exc()


# ----- end code interpreter -----


def main():
    st.set_page_config(page_title="ChatGLM3 Demo", page_icon="🚀", layout="centered", initial_sidebar_state="auto")

    pipeline = get_model(MODEL_PATH)

    st.session_state.setdefault("messages", [])

    st.title("ChatGLM3 Demo")

    prompt = st.chat_input("Chat with ChatGLM3!", key="chat_input")

    mode = st.radio("Mode", [x.value for x in Mode], horizontal=True, label_visibility="hidden")

    DEFAULT_SYSTEM_PROMPT_MAP = {
        Mode.CHAT: CHAT_SYSTEM_PROMPT,
        Mode.TOOL: TOOL_SYSTEM_PROMPT,
        Mode.CI: CI_SYSTEM_PROMPT,
    }
    default_system_prompt = DEFAULT_SYSTEM_PROMPT_MAP.get(mode)
    if default_system_prompt is None:
        st.error(f"Unexpected mode {mode}")

    with st.sidebar:
        top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.5, value=0.8, step=0.01)
        max_length = st.slider(label="Max Length", min_value=128, max_value=2048, value=2048, step=16)
        max_context_length = st.slider(label="Max Context Length", min_value=128, max_value=2048, value=1536, step=16)
        system_prompt = st.text_area(label="System Prompt", value=default_system_prompt, height=300)
        if st.button(label="Clear Context", type="primary"):
            st.session_state.messages = []

    messages: list[Message] = st.session_state.messages

    for msg in messages:
        show_message(msg)

    if not prompt:
        return

    prompt = prompt.strip()
    messages.append(Message(role="user", content=prompt))
    show_message(messages[-1])

    TOOL_CALL_MAX_RETRY = 5
    for _ in range(TOOL_CALL_MAX_RETRY):
        messages_with_system = []
        if system_prompt:
            messages_with_system.append(Message(role="system", content=system_prompt))
        messages_with_system += messages

        chunks = []
        response = ""

        with st.chat_message(name="assistant", avatar="assistant"):
            message_placeholder = st.empty()

            for chunk in pipeline.chat(
                messages_with_system,
                max_length=max_length,
                max_context_length=max_context_length,
                do_sample=temperature > 0,
                top_k=0,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.0,
                stream=True,
            ):
                response += chunk.content
                chunks.append(chunk)
                message_placeholder.markdown(response + "▌")

            message_placeholder.markdown(response)

        reply_message = Message.from_cpp(pipeline.merge_streaming_messages(chunks))
        messages.append(reply_message)
        if not reply_message.tool_calls:
            break

        (tool_call,) = reply_message.tool_calls
        if tool_call.type == "function":
            with st.spinner(f"Calling function `{tool_call.function.name}` ..."):
                observation = run_function(tool_call.function.name, tool_call.function.arguments)
        elif tool_call.type == "code":
            kc = get_kernel_client(IPYKERNEL)
            code = extract_code(tool_call.code.input)
            with st.spinner(f"Executing code ..."):
                observation = run_code(kc, code)
        else:
            st.error(f"Unexpected tool call type {tool_call.type}")
            return

        OBSERVATION_MAX_LENGTH = 1024
        if isinstance(observation, str) and len(observation) > OBSERVATION_MAX_LENGTH:
            observation = observation[:OBSERVATION_MAX_LENGTH] + " [TRUNCATED]"

        if isinstance(observation, str):
            messages.append(Message(role="observation", content=observation))
        else:
            messages.append(Message(role="observation", content="[IMAGE]", image=observation))
        show_message(messages[-1])


if __name__ == "__main__":
    main()
