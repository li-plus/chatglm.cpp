import logging
from datetime import datetime
from typing import List, Tuple

import chatglm_cpp
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")


class Settings(BaseSettings):
    model: str = "chatglm-ggml.bin"


class ChatRequest(BaseModel):
    prompt: str
    history: List[Tuple[str, str]] = []
    max_length: int = Field(default=2048, ge=0)
    top_p: float = Field(default=0.7, ge=0, le=1)
    temperature: float = Field(default=0.95, ge=0, le=2)

    model_config = {"json_schema_extra": {"examples": [{"prompt": "你好"}]}}


class ChatResponse(BaseModel):
    response: str
    history: List[Tuple[str, str]]
    status: int
    time: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
                    "history": [["你好", "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"]],
                    "status": 200,
                    "time": "2023-08-05 23:01:35",
                }
            ]
        }
    }


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
        response=response, history=history, status=status.HTTP_200_OK, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    logging.info(f'prompt: "{body.prompt}", response: "{response}"')
    return answer
