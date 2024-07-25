import asyncio
import base64
import io
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Union

import chatglm_cpp
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")


class Settings(BaseSettings):
    model: str = "models/chatglm3-ggml.bin"
    max_length: int = 4096


class ToolCallFunction(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    function: Optional[ToolCallFunction] = None
    type: Literal["function"]


class ContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentImageUrlData(BaseModel):
    url: str
    detail: str = "high"


class ContentImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ContentImageUrlData


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[ContentText, ContentImageUrl]]]
    tool_calls: Optional[List[ToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionToolFunction(BaseModel):
    description: Optional[str] = None
    name: str
    parameters: Dict


class ChatCompletionTool(BaseModel):
    type: Literal["function"] = "function"
    function: ChatCompletionToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)
    tools: Optional[List[ChatCompletionTool]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]
        }
    }


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "default-model"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "default-model",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


settings = Settings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
pipeline = chatglm_cpp.Pipeline(settings.model, max_length=settings.max_length)
lock = asyncio.Lock()


def stream_chat(messages, body):
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(role="assistant"))],
    )

    for chunk in pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        stream=True,
    ):
        yield ChatCompletionResponse(
            object="chat.completion.chunk",
            choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=chunk.content))],
        )

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
            for chunk in stream_chat(history, body):
                await asyncio.sleep(0)  # yield control back to event loop for cancellation check
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        logging.info(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        logging.info(f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e


@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    def to_json_arguments(arguments):
        def tool_call(**kwargs):
            return kwargs

        try:
            return json.dumps(eval(arguments, dict(tool_call=tool_call)))
        except Exception:
            return arguments

    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = []
    for msg in body.messages:
        if isinstance(msg.content, str):
            messages.append(chatglm_cpp.ChatMessage(role=msg.role, content=msg.content))
        else:
            if not (len(msg.content) == 2 and msg.content[0].type == "text" and msg.content[1].type == "image_url"):
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "multimodal content must have a text item followed by an image_url item",
                )

            import numpy as np
            from PIL import Image

            text = msg.content[0].text
            image_url = msg.content[1].image_url.url
            if image_url.startswith("data:"):
                image_bytes = base64.b64decode(image_url.split(",")[1])
            else:
                import requests

                image_bytes = requests.get(image_url).content
            image = chatglm_cpp.Image(np.asarray(Image.open(io.BytesIO(image_bytes))))

            messages.append(chatglm_cpp.ChatMessage(role=msg.role, content=text, image=image))

    if body.tools:
        system_content = (
            "Answer the following questions as best as you can. You have access to the following tools:\n"
            + json.dumps([tool.model_dump() for tool in body.tools], indent=4)
        )
        messages.insert(0, chatglm_cpp.ChatMessage(role="system", content=system_content))

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)

    max_context_length = 512
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )
    logging.info(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    prompt_tokens = len(pipeline.tokenizer.apply_chat_template(messages, max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

    finish_reason = "stop"
    tool_calls = None
    if output.tool_calls:
        tool_calls = [
            ToolCall(
                type=tool_call.type,
                function=ToolCallFunction(
                    name=tool_call.function.name, arguments=to_json_arguments(tool_call.function.arguments)
                ),
            )
            for tool_call in output.tool_calls
        ]
        finish_reason = "function_call"

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[
            ChatCompletionResponseChoice(
                message=ChatMessage(role="assistant", content=output.content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "owner"
    permission: List = []


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "owner", "permission": []}],
                }
            ]
        }
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id="gpt-3.5-turbo")])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
