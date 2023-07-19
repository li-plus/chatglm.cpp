# Adapted from https://github.com/lloydzhou/rwkv.cpp/blob/master/rwkv/api.py
import json
import logging
import uvicorn
import chatglm_cpp
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, BaseSettings
from sse_starlette.sse import EventSourceResponse


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm-ggml.bin"


class Settings(BaseSettings):
    server_name: str = "ChatGLM CPP API Server"
    model: str = str(DEFAULT_MODEL_PATH)  # Path to chatglm model in ggml format
    host: str = '0.0.0.0'
    port: int = 8000


settings = Settings()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
pipeline = None
completion_lock = Lock()
requests_num = 0

async def run_with_lock(func, request):
    global requests_num
    requests_num = requests_num + 1
    logging.debug("Start Waiting. RequestsNum: %r", requests_num)
    while completion_lock.locked():
        if await request.is_disconnected():
            logging.debug("Stop Waiting (Lock). RequestsNum: %r", requests_num)
            return
        # 等待
        logging.debug("Waiting. RequestsNum: %r", requests_num)
        time.sleep(0.1)
    else:
        with completion_lock:
            if await request.is_disconnected():
                logging.debug("Stop Waiting (Lock). RequestsNum: %r", requests_num)
                return
            return func()


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = chatglm_cpp.Pipeline(settings.model)
    logging.info('End Loading chatglm model')


async def process_generate(history, chat_model, body, request):
    # TODO calc tokens
    usage = {}

    if len(history) % 2 == 0:
        history = ['hi'] + history
    def func():
        for piece in pipeline.stream_chat(
            history,
            max_length=body.max_tokens,
            max_context_length=body.max_context_length,
            do_sample=body.temperature > 0,
            top_k=body.top_k,
            top_p=body.top_p,
            temperature=body.temperature,
        ):
            # debug log
            print(piece, end='', flush=True)
            yield piece

    async def generate():
        response = ''
        for delta in await run_with_lock(func, request):
            response += delta
            if body.stream:
                chunk = format_message('', delta, chunk=True, chat_model=chat_model)
                yield json.dumps(chunk)
        if body.stream:
            result = format_message(response, '', chunk=True, chat_model=chat_model, finish_reason='stop')
            result.update(usage=usage)
            yield json.dumps(result)
        else:
            result = format_message(response, response, chunk=False, chat_model=chat_model, finish_reason='stop')
            result.update(usage=usage)
            yield result

    if body.stream:
        return EventSourceResponse(generate())
    return await generate().__anext__()


def format_message(response, delta, chunk=False, chat_model=False, model_name='chatglm2-6b', finish_reason=None):
    if not chat_model:
        object = 'text_completion'
    else:
        if chunk:
            object = 'chat.completion.chunk'
        else:
            object = 'chat.completion'

    return {
        'object': object,
        'response': response,
        'model': model_name,
        'choices': [{
            'delta': {'content': delta},
            'index': 0,
            'finish_reason': finish_reason,
        } if chat_model else {
            'text': delta,
            'index': 0,
            'finish_reason': finish_reason,
        }]
    }


class ModelConfigBody(BaseModel):
    max_tokens: int = Field(default=2048, gt=0, le=102400)
    max_context_length: int = Field(default=512, gt=0, le=102400)
    temperature: float = Field(default=0.95, ge=0, le=2)
    top_p: float = Field(default=0.7, ge=0, le=1)
    top_k: float = Field(default=0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "max_tokens": 2048,
                "max_context_length": 512,
                "temperature": 0.95,
                "top_p": 0.7,
                "top_k": 0,
            }
        }


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionBody(ModelConfigBody):
    messages: List[Message]
    model: str = "chatglm2-6b"
    stream: bool = False

    class Config:
        schema_extra = {
            "example": {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "chatglm2-6b",
                "stream": False,
                "max_tokens": 2048,
                "max_context_length": 512,
                "temperature": 0.95,
                "top_p": 0.7,
                "top_k": 0,
            }
        }


class CompletionBody(ModelConfigBody):
    prompt: str or List[str]
    model: str = "chatglm2-6b"
    stream: bool = False

    class Config:
        schema_extra = {
            "example": {
                "prompt": "The following is an epic science fiction masterpiece that is immortalized, "
                + "with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
                "model": "chatglm2-6b",
                "stream": False,
                "max_tokens": 2048,
                "max_context_length": 512,
                "temperature": 0.95,
                "top_p": 0.7,
                "top_k": 0,
            }
        }


@app.post('/v1/completions')
@app.post('/completions')
async def completions(body: CompletionBody, request: Request):
    return await process_generate([body.prompt], False, body, request)


@app.post('/v1/chat/completions')
@app.post('/chat/completions')
async def chat_completions(body: ChatCompletionBody, request: Request):
    if len(body.messages) == 0 or body.messages[-1].role != 'user':
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no question found")

    # history = [f'{message.role}: {message.content}' for message in body.messages]
    history = [message.content for message in body.messages]
    return await process_generate(history, True, body, request)


if __name__ == "__main__":
    uvicorn.run("api_demo:app", host=settings.host, port=settings.port)

