import argparse
from pathlib import Path

import chatglm_cpp

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm-ggml.bin"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("-p", "--prompt", default="你好", type=str)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-l", "--max_length", default=2048, type=int)
    parser.add_argument("-c", "--max_context_length", default=512, type=int)
    parser.add_argument("--top_k", default=0, type=int)
    parser.add_argument("--top_p", default=0.7, type=float)
    parser.add_argument("--temp", default=0.95, type=float)
    parser.add_argument("-t", "--threads", default=0, type=int)
    args = parser.parse_args()

    pipeline = chatglm_cpp.Pipeline(args.model)

    if not args.interactive:
        for piece in pipeline.stream_chat(
            [args.prompt],
            max_length=args.max_length,
            max_context_length=args.max_context_length,
            do_sample=args.temp > 0,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temp,
        ):
            print(piece, sep="", end="", flush=True)
        print()
        return

    print(BANNER)
    history = []
    while True:
        try:
            prompt = input(f"{'Prompt':{len(pipeline.model.type_name)}} > ")
        except EOFError:
            break
        if not prompt:
            continue
        history.append(prompt)
        print(f"{pipeline.model.type_name} > ", sep="", end="")
        output = ""
        for piece in pipeline.stream_chat(
            history,
            max_length=args.max_length,
            max_context_length=args.max_context_length,
            do_sample=args.temp > 0,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temp,
        ):
            print(piece, sep="", end="", flush=True)
            output += piece
        print()
        history.append(output)
    print("Bye")


if __name__ == "__main__":
    main()
