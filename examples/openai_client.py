import argparse

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--stream", action="store_true")
parser.add_argument("--prompt", default="你好", type=str)
parser.add_argument("--tool_call", action="store_true")
args = parser.parse_args()

client = OpenAI()

tools = None
if args.tool_call:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

messages = [{"role": "user", "content": args.prompt}]
if args.stream:
    response = client.chat.completions.create(model="default-model", messages=messages, stream=True, tools=tools)
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
    print()
else:
    response = client.chat.completions.create(model="default-model", messages=messages, tools=tools)
    print(response.choices[0].message.content)
