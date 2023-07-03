# Adapted from https://github.com/THUDM/ChatGLM-6B/blob/main/web_demo.py

import argparse
from pathlib import Path

import chatglm_cpp
import gradio as gr
import mdtex2html

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm-ggml.bin"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=Path)
parser.add_argument("-t", "--threads", default=0, type=int)
args = parser.parse_args()

pipeline = chatglm_cpp.Pipeline(args.model)

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    response = ""
    history.append(input)
    for response_piece in pipeline.stream_chat(
        history,
        max_length=max_length,
        do_sample=temperature > 0,
        top_p=top_p,
        temperature=temperature,
        num_threads=args.threads,
    ):
        response += response_piece
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history

    history.append(response)
    yield chatbot, history


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
            max_length = gr.Slider(0, 2048, value=2048, step=1.0, label="Maximum Length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            emptyBtn = gr.Button("Clear History")

    history = gr.State([])

    submitBtn.click(
        predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True
    )
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
