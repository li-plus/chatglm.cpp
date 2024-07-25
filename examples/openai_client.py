import argparse
import base64
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default="Bearer chatglm-cpp-example", type=str, help="API key of OpenAI api server")
parser.add_argument("--base_url", default=None, type=str, help="base url of OpenAI api server")
parser.add_argument("--stream", action="store_true", help="enable stream generation")
parser.add_argument("-p", "--prompt", default="你好", type=str, help="prompt to start generation with")
parser.add_argument("--tool_call", action="store_true", help="enable function call")
parser.add_argument("--image", default=None, type=str, help="path to the input image for visual language models")
parser.add_argument("--temp", default=0.95, type=float, help="temperature")
parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
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
response = client.chat.completions.create(
    model="default-model", messages=messages, stream=args.stream, temperature=args.temp, top_p=args.top_p, tools=tools
)
if args.stream:
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
    print()
else:
    print(response.choices[0].message.content)
