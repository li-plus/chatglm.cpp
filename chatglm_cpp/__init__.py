import os
from pathlib import Path
from typing import Iterator, List

import chatglm_cpp._C as _C

__version__ = "0.2.1"


class Pipeline(_C.Pipeline):
    def __init__(self, model_path: os.PathLike) -> None:
        model_path = Path(model_path)
        super().__init__(str(model_path))

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
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
        )

        input_ids = self.tokenizer.encode_history(history, max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)

        token_cache = []
        print_len = 0
        while len(output_ids) < max_length:
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
    ) -> str:
        gen_config = _C.GenerationConfig(
            max_length=max_length,
            max_context_length=max_context_length,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_threads=num_threads,
        )

        input_ids = self.tokenizer.encode_history(history, max_context_length)

        output_ids = input_ids
        n_past = 0
        n_ctx = len(input_ids)

        while len(output_ids) < max_length:
            next_token_id = self.model.generate_next_token(input_ids, gen_config, n_past, n_ctx)
            n_past += len(input_ids)
            input_ids = [next_token_id]
            output_ids.append(next_token_id)
            if next_token_id == self.model.config.eos_token_id:
                break

        output = self.tokenizer.decode(output_ids[n_ctx:])
        return output
