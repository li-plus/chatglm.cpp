import argparse
import base64
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default="Bearer chatglm-cpp-example", type=str)
parser.add_argument("--base_url", default=None, type=str)
parser.add_argument("--stream", action="store_true")
parser.add_argument("--prompt", default="你好", type=str)
parser.add_argument("--tool_call", action="store_true")
parser.add_argument("--image", default=None, type=str)
args = parser.parse_args()

client = OpenAI(api_key=args.api_key, base_url=args.base_url)

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

if args.image is not None:
    image_url = args.image
    if not image_url.startswith(("http://", "https://")):
        base64_image = base64.b64encode(Path(image_url).read_bytes()).decode()
        image_url = f"data:image/jpeg;base64,{base64_image}"
    user_content = [{"type": "text", "text": args.prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
else:
    user_content = args.prompt

messages = [{"role": "user", "content": user_content}]
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
