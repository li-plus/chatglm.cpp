"""
ChatGLM.cpp python binding
"""
from __future__ import annotations
import typing
import typing_extensions
__all__ = ['BaseModelForCausalLM', 'BaseTokenizer', 'ChatGLM2ForCausalLM', 'ChatGLM2Tokenizer', 'ChatGLM3Tokenizer', 'ChatGLM4Tokenizer', 'ChatGLMForCausalLM', 'ChatGLMTokenizer', 'ChatMessage', 'CodeMessage', 'FunctionMessage', 'GenerationConfig', 'Image', 'ModelConfig', 'ModelType', 'Pipeline', 'ToolCallMessage', 'VisionModelConfig']
class BaseModelForCausalLM:
    def count_tokens(self, input_ids: list[int], image: Image | None) -> int:
        ...
    def generate_next_token(self, input_ids: list[int], image: Image | None, gen_config: GenerationConfig, n_past: int, n_ctx: int) -> int:
        ...
    @property
    def config(self) -> ModelConfig:
        ...
class BaseTokenizer:
    def apply_chat_template(self, messages: list[ChatMessage], max_length: int) -> list[int]:
        ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        ...
    def decode_message(self, ids: list[int]) -> ChatMessage:
        ...
    def encode(self, text: str, max_length: int) -> list[int]:
        ...
class ChatGLM2ForCausalLM(BaseModelForCausalLM):
    pass
class ChatGLM2Tokenizer(BaseTokenizer):
    pass
class ChatGLM3Tokenizer(BaseTokenizer):
    pass
class ChatGLM4Tokenizer(BaseTokenizer):
    pass
class ChatGLMForCausalLM(BaseModelForCausalLM):
    pass
class ChatGLMTokenizer(BaseTokenizer):
    pass
class ChatMessage:
    ROLE_ASSISTANT: typing.ClassVar[str] = 'assistant'
    ROLE_OBSERVATION: typing.ClassVar[str] = 'observation'
    ROLE_SYSTEM: typing.ClassVar[str] = 'system'
    ROLE_USER: typing.ClassVar[str] = 'user'
    content: str
    image: Image | None
    role: str
    tool_calls: list[ToolCallMessage]
    def __init__(self, role: str, content: str, image: Image | None = None, tool_calls: list[ToolCallMessage] = []) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class CodeMessage:
    input: str
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class FunctionMessage:
    arguments: str
    name: str
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class GenerationConfig:
    do_sample: bool
    max_context_length: int
    max_length: int
    max_new_tokens: int
    repetition_penalty: float
    temperature: float
    top_k: int
    top_p: float
    def __init__(self, max_length: int = 2048, max_new_tokens: int = -1, max_context_length: int = 512, do_sample: bool = True, top_k: int = 0, top_p: float = 0.7, temperature: float = 0.95, repetition_penalty: float = 1.0) -> None:
        ...
class Image:
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def channels(self) -> int:
        ...
    @property
    def height(self) -> int:
        ...
    @property
    def pixels(self) -> list[int]:
        ...
    @property
    def width(self) -> int:
        ...
class ModelConfig:
    @property
    def bos_token_id(self) -> int:
        ...
    @property
    def eos_token_id(self) -> int:
        ...
    @property
    def extra_eos_token_ids(self) -> list[int]:
        ...
    @property
    def hidden_size(self) -> int:
        ...
    @property
    def intermediate_size(self) -> int:
        ...
    @property
    def max_length(self) -> int:
        ...
    @property
    def model_type(self) -> ModelType:
        ...
    @property
    def model_type_name(self) -> str:
        ...
    @property
    def norm_eps(self) -> float:
        ...
    @property
    def num_attention_heads(self) -> int:
        ...
    @property
    def num_hidden_layers(self) -> int:
        ...
    @property
    def num_key_value_heads(self) -> int:
        ...
    @property
    def pad_token_id(self) -> int:
        ...
    @property
    def sep_token_id(self) -> int:
        ...
    @property
    def vision(self) -> VisionModelConfig:
        ...
    @property
    def vocab_size(self) -> int:
        ...
class ModelType:
    """
    Members:
    
      CHATGLM
    
      CHATGLM2
    
      CHATGLM3
    
      CHATGLM4
    """
    CHATGLM: typing.ClassVar[ModelType]  # value = <ModelType.CHATGLM: 1>
    CHATGLM2: typing.ClassVar[ModelType]  # value = <ModelType.CHATGLM2: 2>
    CHATGLM3: typing.ClassVar[ModelType]  # value = <ModelType.CHATGLM3: 3>
    CHATGLM4: typing.ClassVar[ModelType]  # value = <ModelType.CHATGLM4: 4>
    __members__: typing.ClassVar[dict[str, ModelType]]  # value = {'CHATGLM': <ModelType.CHATGLM: 1>, 'CHATGLM2': <ModelType.CHATGLM2: 2>, 'CHATGLM3': <ModelType.CHATGLM3: 3>, 'CHATGLM4': <ModelType.CHATGLM4: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Pipeline:
    def __init__(self, path: str, max_length: int = -1) -> None:
        ...
    @property
    def model(self) -> BaseModelForCausalLM:
        ...
    @property
    def tokenizer(self) -> BaseTokenizer:
        ...
class ToolCallMessage:
    code: CodeMessage
    function: FunctionMessage
    type: str
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class VisionModelConfig:
    @property
    def hidden_size(self) -> int:
        ...
    @property
    def image_size(self) -> int:
        ...
    @property
    def in_channels(self) -> int:
        ...
    @property
    def intermediate_size(self) -> int:
        ...
    @property
    def norm_eps(self) -> float:
        ...
    @property
    def num_attention_heads(self) -> int:
        ...
    @property
    def num_hidden_layers(self) -> int:
        ...
    @property
    def num_positions(self) -> int:
        ...
    @property
    def patch_size(self) -> int:
        ...
    @property
    def scaling_factor(self) -> float:
        ...
