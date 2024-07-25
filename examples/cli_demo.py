import argparse
from pathlib import Path
from typing import List

import chatglm_cpp

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models/chatglm-ggml.bin"

BANNER = """
    ________          __  ________    __  ___                 
   / ____/ /_  ____ _/ /_/ ____/ /   /  |/  /_________  ____  
  / /   / __ \/ __ `/ __/ / __/ /   / /|_/ // ___/ __ \/ __ \ 
 / /___/ / / / /_/ / /_/ /_/ / /___/ /  / // /__/ /_/ / /_/ / 
 \____/_/ /_/\__,_/\__/\____/_____/_/  /_(_)___/ .___/ .___/  
                                              /_/   /_/       
""".strip(
    "\n"
)
WELCOME_MESSAGE = "Welcome to ChatGLM.cpp! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=str, help="model path")
    parser.add_argument("--mode", default="chat", type=str, choices=["chat", "generate"], help="inference mode")
    parser.add_argument("-p", "--prompt", default="你好", type=str, help="prompt to start generation with")
    parser.add_argument(
        "--pp", "--prompt_path", default=None, type=Path, help="path to the plain text file that stores the prompt"
    )
    parser.add_argument(
        "-s", "--system", default=None, type=str, help="system message to set the behavior of the assistant"
    )
    parser.add_argument(
        "--sp",
        "--system_path",
        default=None,
        type=Path,
        help="path to the plain text file that stores the system message",
    )
    parser.add_argument("--image", default=None, type=str, help="path to the input image for visual language models")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument(
        "-l", "--max_length", default=2048, type=int, help="max total length including prompt and output"
    )
    parser.add_argument(
        "--max_new_tokens",
        default=-1,
        type=int,
        help="max number of tokens to generate, ignoring the number of prompt tokens",
    )
    parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
    parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
    parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
    parser.add_argument("--temp", default=0.95, type=float, help="temperature")
    parser.add_argument("--repeat_penalty", default=1.0, type=float, help="penalize repeat sequence of tokens")
    args = parser.parse_args()

    prompt = args.prompt
    if args.pp:
        prompt = args.pp.read_text()

    system = args.system
    if args.sp:
        system = args.sp.read_text()

    pipeline = chatglm_cpp.Pipeline(args.model, max_length=args.max_length)

    if args.mode != "chat" and args.interactive:
        print("interactive demo is only supported for chat mode, falling back to non-interactive one")
        args.interactive = False

    generation_kwargs = dict(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        do_sample=args.temp > 0,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        repetition_penalty=args.repeat_penalty,
        stream=True,
    )

    system_messages: List[chatglm_cpp.ChatMessage] = []
    if system is not None:
        system_messages.append(chatglm_cpp.ChatMessage(role="system", content=system))

    messages = system_messages.copy()

    image = None
    if args.image is not None:
        import numpy as np
        from PIL import Image

        image = chatglm_cpp.Image(np.asarray(Image.open(args.image).convert("RGB")))

    if not args.interactive:
        if args.mode == "chat":
            messages.append(chatglm_cpp.ChatMessage(role="user", content=prompt, image=image))
            for chunk in pipeline.chat(messages, **generation_kwargs):
                print(chunk.content, sep="", end="", flush=True)
        else:
            for chunk in pipeline.generate(prompt, **generation_kwargs):
                print(chunk, sep="", end="", flush=True)
        print()
        return

    print(BANNER)
    print()
    print(WELCOME_MESSAGE)
    print()

    prompt_width = len(pipeline.model.config.model_type_name)

    if system:
        print(f"{'System':{prompt_width}} > {system}")

    prompt_image = image
    while True:
        if messages and messages[-1].tool_calls:
            (tool_call,) = messages[-1].tool_calls
            if tool_call.type == "function":
                print(
                    f"Function Call > Please manually call function `{tool_call.function.name}` and provide the results below."
                )
                input_prompt = "Observation   > "
            elif tool_call.type == "code":
                print(f"Code Interpreter > Please manually run the code and provide the results below.")
                input_prompt = "Observation      > "
            else:
                raise ValueError(f"unexpected tool call type {tool_call.type}")
            role = "observation"
        else:
            input_prompt = f"{'Prompt':{prompt_width}} > "
            role = "user"

        try:
            prompt = input(input_prompt)
        except EOFError:
            break

        if not prompt:
            continue
        if prompt == "stop":
            break
        if prompt == "clear":
            messages = system_messages.copy()
            prompt_image = image
            continue

        messages.append(chatglm_cpp.ChatMessage(role=role, content=prompt, image=prompt_image))
        prompt_image = None
        print(f"{pipeline.model.config.model_type_name} > ", sep="", end="")
        chunks = []
        for chunk in pipeline.chat(messages, **generation_kwargs):
            print(chunk.content, sep="", end="", flush=True)
            chunks.append(chunk)
        print()
        messages.append(pipeline.merge_streaming_messages(chunks))

    print("Bye")


if __name__ == "__main__":
    main()
