import logging
from datetime import datetime
from typing import List, Tuple

import chatglm_cpp
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")


class Settings(BaseSettings):
    model: str = "chatglm-ggml.bin"


class ChatRequest(BaseModel):
    prompt: str
    history: List[Tuple[str, str]] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95


class ChatResponse(BaseModel):
    response: str
    history: List[Tuple[str, str]]
    status: int
    time: str


app = FastAPI()
settings = Settings()
logging.info(settings)

pipeline = chatglm_cpp.Pipeline(settings.model)


@app.post("/")
async def chat(body: ChatRequest) -> ChatResponse:
    chat_history = [msg for pair in body.history for msg in pair] + [body.prompt]
    response = pipeline.chat(
        chat_history,
        max_length=body.max_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )
    history = body.history + [(body.prompt, response)]
    answer = ChatResponse(
        response=response, history=history, status=200, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    logging.info(f'prompt: "{body.prompt}", response: "{response}"')
    return answer
