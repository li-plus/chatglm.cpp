import tempfile
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Union

import chatglm_cpp._C as _C

__version__ = "0.2.4"


class Pipeline(_C.Pipeline):
    def __init__(self, model_path: str, *, dtype: Optional[str] = None) -> None:
        if Path(model_path).is_file():
            # load ggml model
            super().__init__(str(model_path))
        else:
            # convert hf model to ggml format
            from chatglm_cpp.convert import convert

            if dtype is None:
                dtype = "q4_0"  # default dtype

            with tempfile.NamedTemporaryFile("wb") as f:
                convert(f, model_path, dtype=dtype)
                super().__init__(f.name)

    def chat(
        self,
        history: List[str],
        *,
        max_length: int = 2048,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        num_threads: int = 0,
        stream: bool = False,
    ) -> Union[Iterator[str], str]:
        prompt = self.tokenizer.build_prompt(history)
        return self.generate(
            prompt=prompt,
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
            stream=stream,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_length: int = 2048,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        num_threads: int = 0,
        stream: bool = False,
    ) -> Union[Iterator[str], str]:
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
        )

        generate_fn = self._stream_generate if stream else self._generate
        return generate_fn(prompt=prompt, gen_config=gen_config)

    def _stream_generate(self, prompt: str, gen_config: _C.GenerationConfig) -> Iterator[str]:
        input_ids = self.tokenizer.encode(prompt, gen_config.max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)

        token_cache = []
        print_len = 0
        while len(output_ids) < gen_config.max_length:
            next_token_id = self.model.generate_next_token(input_ids, gen_config, n_past, n_ctx)
            n_past += len(input_ids)
            input_ids = [next_token_id]
            output_ids.append(next_token_id)

            token_cache.append(next_token_id)
            output = self.tokenizer.decode(token_cache)

            if output.endswith("\n"):
                yield output[print_len:]
                token_cache = []
                print_len = 0
            elif output.endswith((",", "!", ":", ";", "?", "�")):
                pass
            else:
                yield output[print_len:]
                print_len = len(output)

            if next_token_id == self.model.config.eos_token_id:
                break

        output = self.tokenizer.decode(token_cache)
        yield output[print_len:]

    def _generate(self, prompt: str, gen_config: _C.GenerationConfig) -> str:
        input_ids = self.tokenizer.encode(prompt, gen_config.max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)

        while len(output_ids) < gen_config.max_length:
            next_token_id = self.model.generate_next_token(input_ids, gen_config, n_past, n_ctx)
            n_past += len(input_ids)
            input_ids = [next_token_id]
            output_ids.append(next_token_id)
            if next_token_id == self.model.config.eos_token_id:
                break

        output = self.tokenizer.decode(output_ids[n_ctx:])
        return output

    def stream_chat(
        self,
        history: List[str],
        *,
        max_length: int = 2048,
        max_context_length: int = 512,
        do_sample: bool = True,
        top_k: int = 0,
        top_p: float = 0.7,
        temperature: float = 0.95,
        num_threads: int = 0,
    ) -> Iterator[str]:
        warnings.warn(
            "stream_chat is deprecated in favor of chat(..., stream=True), and will be removed in the next major version of chatglm-cpp",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat(
            history=history,
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
            stream=True,
        )
