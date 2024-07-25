import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import chatglm_cpp._C as _C
from chatglm_cpp._C import ChatMessage, Image

__version__ = "0.4.1"


@dataclass
class DeltaMessage:
    role: str
    content: str
    token_ids: List[int]


def _ensure_chat_message(message: Union[ChatMessage, Dict[str, Any]]) -> ChatMessage:
    if isinstance(message, ChatMessage):
        chat_message = message
    elif isinstance(message, dict):
        chat_message = ChatMessage(**message)
    else:
        raise TypeError(f"expect message type to be ChatMessage or dict, but got {type(message)}")
    return chat_message


class Pipeline(_C.Pipeline):
    def __init__(self, model_path: str, *, max_length: Optional[int] = None, dtype: Optional[str] = None) -> None:
        kwargs = {}
        if max_length is not None:
            kwargs.update(max_length=max_length)

        if Path(model_path).is_file():
            # load ggml model
            super().__init__(str(model_path), **kwargs)
        else:
            # convert hf model to ggml format
            from chatglm_cpp.convert import convert

            if dtype is None:
                dtype = "q4_0"  # default dtype

            with tempfile.NamedTemporaryFile("wb") as f:
                convert(f, model_path, dtype=dtype)
                super().__init__(f.name, **kwargs)

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        max_length: int = 2048,
        max_new_tokens: int = -1,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        repetition_penalty: float = 1.0,
        stream: bool = False,
    ) -> Union[Iterator[DeltaMessage], ChatMessage]:
        messages = [_ensure_chat_message(msg) for msg in messages]
        input_ids = self.tokenizer.apply_chat_template(messages, max_context_length)
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        image = None
        for msg in messages:
            if msg.image is not None:
                image = msg.image
                break

        if stream:
            return self._stream_chat(input_ids=input_ids, image=image, gen_config=gen_config)
        return self._sync_chat(input_ids=input_ids, image=image, gen_config=gen_config)

    def generate(
        self,
        prompt: str,
        *,
        max_length: int = 2048,
        max_new_tokens: int = -1,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        repetition_penalty: float = 1.0,
        stream: bool = False,
    ) -> Union[Iterator[str], str]:
        input_ids = self.tokenizer.encode(prompt, max_context_length)
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        if stream:
            return self._stream_generate(input_ids=input_ids, gen_config=gen_config)
        return self._sync_generate(input_ids=input_ids, gen_config=gen_config)

    def _stream_generate_ids(
        self, input_ids: List[int], image: Optional[Image], gen_config: _C.GenerationConfig
    ) -> Iterator[int]:
        input_ids = input_ids.copy()
        n_past = 0
        n_ctx = len(input_ids)
        max_new_tokens = gen_config.max_new_tokens if gen_config.max_new_tokens > 0 else gen_config.max_length

        while len(input_ids) < min(gen_config.max_length, n_ctx + max_new_tokens):
            next_token_id = self.model.generate_next_token(input_ids, image, gen_config, n_past, n_ctx)
            yield next_token_id
            n_past = self.model.count_tokens(input_ids, image)
            input_ids.append(next_token_id)

            if next_token_id in [self.model.config.eos_token_id, *self.model.config.extra_eos_token_ids]:
                break

    def _stream_chat(
        self, input_ids: List[int], image: Optional[Image], gen_config: _C.GenerationConfig
    ) -> Iterator[DeltaMessage]:
        token_cache = []
        print_len = 0
        print_token_len = 0
        for next_token_id in self._stream_generate_ids(input_ids=input_ids, image=image, gen_config=gen_config):
            token_cache.append(next_token_id)

            try:
                output = self.tokenizer.decode(token_cache)
            except UnicodeDecodeError:
                continue

            if output.endswith("\n"):
                yield DeltaMessage(
                    role=ChatMessage.ROLE_ASSISTANT, content=output[print_len:], token_ids=token_cache[print_token_len:]
                )
                token_cache = []
                print_len = 0
                print_token_len = 0
            elif output.endswith((",", "!", ":", ";", "?", "�")):
                pass
            else:
                yield DeltaMessage(
                    role=ChatMessage.ROLE_ASSISTANT, content=output[print_len:], token_ids=token_cache[print_token_len:]
                )
                print_len = len(output)
                print_token_len = len(token_cache)

        output = self.tokenizer.decode(token_cache)
        yield DeltaMessage(
            role=ChatMessage.ROLE_ASSISTANT, content=output[print_len:], token_ids=token_cache[print_token_len:]
        )

    def _stream_generate(self, input_ids: List[int], gen_config: _C.GenerationConfig) -> Iterator[str]:
        for msg in self._stream_chat(input_ids=input_ids, gen_config=gen_config):
            yield msg.content

    def _sync_generate_ids(
        self, input_ids: List[int], image: Optional[Image], gen_config: _C.GenerationConfig
    ) -> List[int]:
        return list(self._stream_generate_ids(input_ids=input_ids, image=image, gen_config=gen_config))

    def _sync_generate(self, input_ids: List[int], image: Optional[Image], gen_config: _C.GenerationConfig) -> str:
        output_ids = self._sync_generate_ids(input_ids=input_ids, image=image, gen_config=gen_config)
        return self.tokenizer.decode(output_ids)

    def _sync_chat(self, input_ids: List[int], image: Optional[Image], gen_config: _C.GenerationConfig) -> ChatMessage:
        output_ids = self._sync_generate_ids(input_ids=input_ids, image=image, gen_config=gen_config)
        return self.tokenizer.decode_message(output_ids)

    def merge_streaming_messages(self, chunks: List[DeltaMessage]) -> ChatMessage:
        output_ids = [x for chunk in chunks for x in chunk.token_ids]
        return self.tokenizer.decode_message(output_ids)
