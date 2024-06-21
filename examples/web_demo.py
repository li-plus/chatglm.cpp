# Adapted from https://github.com/THUDM/ChatGLM-6B/blob/main/web_demo.py

import argparse
from pathlib import Path

import chatglm_cpp
import gradio as gr

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm-ggml.bin"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=Path, help="model path")
parser.add_argument("--mode", default="chat", type=str, choices=["chat", "generate"], help="inference mode")
parser.add_argument("-l", "--max_length", default=2048, type=int, help="max total length including prompt and output")
parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
parser.add_argument("--temp", default=0.95, type=float, help="temperature")
parser.add_argument("--repeat_penalty", default=1.0, type=float, help="penalize repeat sequence of tokens")
parser.add_argument("--plain", action="store_true", help="display in plain text without markdown support")
args = parser.parse_args()

pipeline = chatglm_cpp.Pipeline(args.model, max_length=args.max_length)


def postprocess(text):
    if args.plain:
        return f"<pre>{text}</pre>"
    return text


def predict(input, chatbot, max_length, top_p, temperature, messages):
    chatbot.append((postprocess(input), ""))
    messages.append(chatglm_cpp.ChatMessage(role="user", content=input))

    generation_kwargs = dict(
        max_length=max_length,
        max_context_length=args.max_context_length,
        do_sample=temperature > 0,
        top_k=args.top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=args.repeat_penalty,
        stream=True,
    )

    response = ""
    if args.mode == "chat":
        chunks = []
        for chunk in pipeline.chat(messages, **generation_kwargs):
            response += chunk.content
            chunks.append(chunk)
            chatbot[-1] = (chatbot[-1][0], postprocess(response))
            yield chatbot, messages
        messages.append(pipeline.merge_streaming_messages(chunks))
    else:
        for chunk in pipeline.generate(input, **generation_kwargs):
            response += chunk
            chatbot[-1] = (chatbot[-1][0], postprocess(response))
            yield chatbot, messages

    yield chatbot, messages


def reset_user_input():
    return gr.update(value="")


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM.cpp</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=8)
            submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 2048, value=args.max_length, step=1.0, label="Maximum Length", interactive=True)
            top_p = gr.Slider(0, 1, value=args.top_p, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=args.temp, step=0.01, label="Temperature", interactive=True)
            emptyBtn = gr.Button("Clear History")

    messages = gr.State([])

    submitBtn.click(
        predict,
        [user_input, chatbot, max_length, top_p, temperature, messages],
        [chatbot, messages],
        show_progress=True,
    )
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, messages], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
