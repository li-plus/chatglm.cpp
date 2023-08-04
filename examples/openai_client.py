import argparse

import openai

parser = argparse.ArgumentParser()
parser.add_argument("--stream", action="store_true")
parser.add_argument("--prompt", default="你好", type=str)
args = parser.parse_args()

messages = [{"role": "user", "content": args.prompt}]
if args.stream:
    response = openai.ChatCompletion.create(model="default-model", messages=messages, stream=True)
    for chunk in response:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
    print()
else:
    response = openai.ChatCompletion.create(model="default-model", messages=messages)
    print(response["choices"][0]["message"]["content"])
