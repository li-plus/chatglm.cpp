"""
Convert Hugging Face ChatGLM family models to GGML format
"""

import argparse
import platform
import struct
import sys
from enum import Enum
from pathlib import Path
from typing import BinaryIO, NamedTuple, Optional

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    import tiktoken
except ImportError:
    tiktoken = None

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

GGML_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()  # type: ignore


class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8


class ModelType(Enum):
    CHATGLM = 1
    CHATGLM2 = 2
    CHATGLM3 = 3
    CHATGLM4 = 4
    CHATGLM4V = 1004


class WeightMeta(NamedTuple):
    name: str
    dtype: ModelType


def quantize_q8_0(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_1(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[1] % GGML_QK4_1 == 0
    tensor = tensor.view(-1, GGML_QK4_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 4) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale & min into each block
    tensor = torch.cat((scale.half().view(torch.int8), min_vals.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q5_0(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to ggml_quantize_q5_0 in ggml.c
    assert tensor.shape[1] % GGML_QK5_0 == 0
    tensor = tensor.view(-1, GGML_QK5_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -16
    tensor = (tensor / scale + 16).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1)
    return tensor


def quantize_q5_1(tensor: torch.Tensor) -> torch.Tensor:
    # equivalent to ggml_quantize_q5_1 in ggml.c
    assert tensor.shape[1] % GGML_QK5_1 == 0
    tensor = tensor.view(-1, GGML_QK5_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 5) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale & min into each block
    tensor = torch.cat(
        (scale.half().view(torch.int8), min_vals.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1
    )
    return tensor


def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))

    # tensor data
    if ggml_type == GGMLType.F32:
        tensor = tensor.float()
    elif ggml_type == GGMLType.F16:
        tensor = tensor.half()
    elif ggml_type == GGMLType.Q8_0:
        tensor = quantize_q8_0(tensor)
    elif ggml_type == GGMLType.Q4_0:
        tensor = quantize_q4_0(tensor)
    elif ggml_type == GGMLType.Q4_1:
        tensor = quantize_q4_1(tensor)
    elif ggml_type == GGMLType.Q5_0:
        tensor = quantize_q5_0(tensor)
    elif ggml_type == GGMLType.Q5_1:
        tensor = quantize_q5_1(tensor)
    else:
        raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)


def dump_state_dict(f, weight_meta, state_dict, quantization_bit):
    tensor_info = []
    for meta in tqdm(weight_meta, desc="Processing model states"):
        tensor = state_dict[meta.name]
        if tensor.ndim == 2 and tensor.dtype == torch.int8:
            # de-quantize gemm weight back to float32
            assert quantization_bit in [4, 8]
            scale = state_dict[f"{meta.name}_scale"].float()  # channel-wise scale

            if quantization_bit == 4:
                # convert int4 weight to int8
                low_bits = ((tensor << 4) & 0xF0) >> 4
                high_bits = (tensor & 0xF0) >> 4
                tensor = torch.stack((high_bits, low_bits), dim=-1).view(tensor.shape[0], -1)
            tensor = tensor * scale[:, None]

        dump_tensor(f, meta.name, tensor, meta.dtype)
        tensor_info.append((meta.name, tuple(tensor.shape), meta.dtype.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


class BaseConverter:
    @classmethod
    def convert(cls, f, model, tokenizer, ggml_type, vision_type=None):
        f.write(b"ggml")  # magic
        f.write(struct.pack("i", cls.MODEL_TYPE.value))  # model type
        cls.dump_config(f, model.config, ggml_type, vision_type)
        cls.dump_tokenizer(f, tokenizer)
        cls.dump_model(f, model, ggml_type, vision_type)


def get_prefix_cache(prefix_encoder, pre_seq_len, num_layers, num_key_value_heads, head_size):
    prefix_tokens = torch.arange(pre_seq_len, dtype=torch.long)
    with torch.no_grad():
        past_key_values = prefix_encoder(prefix_tokens)
    past_key_values = (
        past_key_values.to(torch.half)
        .view(pre_seq_len, num_layers * 2, num_key_value_heads, head_size)
        .permute(1, 2, 0, 3)
        .contiguous()
    )
    return past_key_values


class ChatGLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM

    @staticmethod
    def dump_config(f, config, ggml_type, vision_type):
        assert config.position_encoding_2d, "unimplemented: position_encoding_2d should be True"
        assert (
            config.inner_hidden_size == 4 * config.hidden_size
        ), "unimplemented: inner_hidden_size should be 4 times hidden_size"

        config_version = 2
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_attention_heads,
            config.num_layers,
            config.inner_hidden_size,
            config.layernorm_epsilon,
            config.pre_seq_len if config.pre_seq_len is not None else 0,
            10000.0,  # rope_theta
            config.max_sequence_length,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
        ]
        f.write(struct.pack("iiiiiiiififiii", config_version, *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.sp_tokenizer.text_tokenizer.sp.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ggml_type, vision_type):
        assert torch.allclose(
            model.state_dict()["transformer.word_embeddings.weight"], model.state_dict()["lm_head.weight"]
        ), "unimplemented: lm_head weight must be tied to input embedding"

        weight_meta = [WeightMeta("transformer.word_embeddings.weight", ggml_type)]
        for i in range(model.config.num_layers):
            weight_meta += [
                WeightMeta(f"transformer.layers.{i}.input_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.input_layernorm.bias", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.attention.query_key_value.weight", ggml_type),
                WeightMeta(f"transformer.layers.{i}.attention.query_key_value.bias", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.attention.dense.weight", ggml_type),
                WeightMeta(f"transformer.layers.{i}.attention.dense.bias", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.post_attention_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.post_attention_layernorm.bias", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.mlp.dense_h_to_4h.weight", ggml_type),
                WeightMeta(f"transformer.layers.{i}.mlp.dense_h_to_4h.bias", GGMLType.F32),
                WeightMeta(f"transformer.layers.{i}.mlp.dense_4h_to_h.weight", ggml_type),
                WeightMeta(f"transformer.layers.{i}.mlp.dense_4h_to_h.bias", GGMLType.F32),
            ]
        weight_meta += [
            WeightMeta("transformer.final_layernorm.weight", GGMLType.F32),
            WeightMeta("transformer.final_layernorm.bias", GGMLType.F32),
        ]
        dump_state_dict(f, weight_meta, model.state_dict(), model.config.quantization_bit)


class ChatGLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM2

    @classmethod
    def dump_config(cls, f, config, ggml_type, vision_type):
        assert config.add_bias_linear is False, "unimplemented: add_bias_linear must be false"
        assert config.add_qkv_bias is True, "unimplemented: add_qkv_bias must be true"
        assert (
            config.apply_residual_connection_post_layernorm is False
        ), "unimplemented: apply_residual_connection_post_layernorm must be false"
        assert (
            config.kv_channels * config.num_attention_heads == config.hidden_size
        ), "unimplemented: invalid kv_channels"
        assert config.multi_query_attention is True, "unimplemented: multi_query_attention must be true"
        assert config.original_rope is True, "unimplemented: original_rope must be true"
        assert config.post_layer_norm is True, "unimplemented: post_layer_norm must be true"
        assert config.rmsnorm is True, "unimplemented: rmsnorm must be true"

        config_version = 2
        config_values = [
            ggml_type.value,
            config.padded_vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.multi_query_group_num,
            config.num_layers,
            config.ffn_hidden_size,
            config.layernorm_epsilon,
            config.pre_seq_len if getattr(config, "pre_seq_len", None) is not None else 0,
            10000.0 * getattr(config, "rope_ratio", 1),  # rope_theta
            config.seq_length,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
        ]

        f.write(struct.pack("iiiiiiiififiii", config_version, *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.tokenizer.sp_model.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ggml_type, vision_type):
        config = model.config

        state_dict = model.state_dict()

        weight_meta = []
        if getattr(config, "pre_seq_len", None) is not None and config.pre_seq_len > 0:
            past_key_values = get_prefix_cache(
                model.transformer.prefix_encoder,
                config.pre_seq_len,
                config.num_layers,
                config.multi_query_group_num,
                config.kv_channels,
            )
            state_dict["past_key_values"] = past_key_values
            weight_meta.append(WeightMeta("past_key_values", GGMLType.F16))

        weight_meta.append(WeightMeta("transformer.embedding.word_embeddings.weight", ggml_type))
        for i in range(config.num_layers):
            weight_meta += [
                WeightMeta(f"transformer.encoder.layers.{i}.input_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.dense.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.post_attention_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight", ggml_type),
            ]
        weight_meta += [
            WeightMeta("transformer.encoder.final_layernorm.weight", GGMLType.F32),
            WeightMeta("transformer.output_layer.weight", ggml_type),
        ]
        dump_state_dict(
            f,
            weight_meta=weight_meta,
            state_dict=state_dict,
            quantization_bit=getattr(config, "quantization_bit", None),
        )


class ChatGLM3Converter(ChatGLM2Converter):
    MODEL_TYPE = ModelType.CHATGLM3


class ChatGLM4Converter(ChatGLM2Converter):
    MODEL_TYPE = ModelType.CHATGLM4

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        vocab_text = Path(tokenizer.vocab_file).read_bytes()
        f.write(struct.pack("i", len(vocab_text)))
        f.write(vocab_text)


class ChatGLM4VConverter(ChatGLM4Converter):
    MODEL_TYPE = ModelType.CHATGLM4V

    @classmethod
    def dump_config(cls, f, config, ggml_type, vision_type):
        ChatGLM4Converter.dump_config(f, config, ggml_type, vision_type)

        config_values = [
            vision_type.value,
            config.vision_config["hidden_size"],
            config.vision_config["image_size"],
            config.vision_config["in_channels"],
            config.vision_config["intermediate_size"],
            config.vision_config["layer_norm_eps"],
            config.vision_config["num_heads"],
            config.vision_config["num_hidden_layers"],
            config.vision_config["num_positions"],
            config.vision_config["patch_size"],
            config.vision_config["scaling_factor"],
        ]

        f.write(struct.pack("iiiiifiiiif", *config_values))

    @staticmethod
    def dump_model(f, model, ggml_type, vision_type):
        config = model.config

        state_dict = model.state_dict()

        # vision
        weight_meta = [
            WeightMeta("transformer.vision.patch_embedding.cls_embedding", GGMLType.F16),
            WeightMeta("transformer.vision.patch_embedding.proj.weight", GGMLType.F16),
            WeightMeta("transformer.vision.patch_embedding.proj.bias", GGMLType.F32),
            WeightMeta("transformer.vision.patch_embedding.position_embedding.weight", GGMLType.F32),
        ]
        for i in range(config.vision_config["num_hidden_layers"]):
            weight_meta += [
                WeightMeta(f"transformer.vision.transformer.layers.{i}.input_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.input_layernorm.bias", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.attention.query_key_value.weight", vision_type),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.attention.query_key_value.bias", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.attention.dense.weight", vision_type),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.attention.dense.bias", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.mlp.fc1.weight", vision_type),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.mlp.fc1.bias", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.mlp.fc2.weight", vision_type),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.mlp.fc2.bias", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.post_attention_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.vision.transformer.layers.{i}.post_attention_layernorm.bias", GGMLType.F32),
            ]
        weight_meta += [
            WeightMeta("transformer.vision.conv.weight", GGMLType.F16),
            WeightMeta("transformer.vision.conv.bias", GGMLType.F32),
            WeightMeta("transformer.vision.linear_proj.linear_proj.weight", vision_type),
            WeightMeta("transformer.vision.linear_proj.norm1.weight", GGMLType.F32),
            WeightMeta("transformer.vision.linear_proj.norm1.bias", GGMLType.F32),
            WeightMeta("transformer.vision.linear_proj.gate_proj.weight", vision_type),
            WeightMeta("transformer.vision.linear_proj.dense_h_to_4h.weight", vision_type),
            WeightMeta("transformer.vision.linear_proj.dense_4h_to_h.weight", vision_type),
            WeightMeta("transformer.vision.boi", GGMLType.F16),
            WeightMeta("transformer.vision.eoi", GGMLType.F16),
        ]

        # text
        weight_meta.append(WeightMeta("transformer.embedding.word_embeddings.weight", ggml_type))
        for i in range(config.num_layers):
            weight_meta += [
                WeightMeta(f"transformer.encoder.layers.{i}.input_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.self_attention.dense.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.post_attention_layernorm.weight", GGMLType.F32),
                WeightMeta(f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight", ggml_type),
                WeightMeta(f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight", ggml_type),
            ]
        weight_meta += [
            WeightMeta("transformer.encoder.final_layernorm.weight", GGMLType.F32),
            WeightMeta("transformer.output_layer.weight", ggml_type),
        ]
        dump_state_dict(
            f,
            weight_meta=weight_meta,
            state_dict=state_dict,
            quantization_bit=getattr(config, "quantization_bit", None),
        )


def convert(
    f: BinaryIO,
    model_name_or_path: str,
    lora_model_name_or_path: Optional[str] = None,
    dtype: str = "q4_0",
    vision_dtype: str = "f16",
):
    ggml_type = GGMLType[dtype.upper()]
    vision_type = GGMLType[vision_dtype.upper()]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "AutoModel" in config.auto_map:
        auto_model_class = AutoModel
    elif "AutoModelForCausalLM" in config.auto_map:
        auto_model_class = AutoModelForCausalLM
    else:
        raise RuntimeError(f"Cannot find auto model class to load {model_name_or_path}")

    model = auto_model_class.from_pretrained(model_name_or_path, trust_remote_code=True, low_cpu_mem_usage=True)

    if lora_model_name_or_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
        model = model.merge_and_unload()

    model = model.eval()

    if model.config.model_type == "chatglm":
        if hasattr(model.config, "multi_query_attention"):
            # ChatGLM 2,3,4 share the same architecture and model config but their tokenizers are different.
            # ChatGLM4 uses tiktoken tokenizer, while ChatGLM 2,3 uses sentencepiece.
            # ChatGLM3 has system token to support system prompt, while ChatGLM2 does not.
            if tiktoken is not None and isinstance(tokenizer.tokenizer, tiktoken.Encoding):
                # TODO: store all eos token ids
                model.config.eos_token_id = tokenizer.eos_token_id
                if getattr(model.config, "vision_config", None) is not None:
                    ChatGLM4VConverter.convert(f, model, tokenizer, ggml_type, vision_type)
                else:
                    ChatGLM4Converter.convert(f, model, tokenizer, ggml_type)
            elif "<|system|>" in tokenizer.tokenizer.special_tokens:
                ChatGLM3Converter.convert(f, model, tokenizer, ggml_type)
            else:
                ChatGLM2Converter.convert(f, model, tokenizer, ggml_type)
        else:
            ChatGLMConverter.convert(f, model, tokenizer, ggml_type)
    else:
        raise RuntimeError(f"Unknown model type {model.config.model_type}")


def main():
    parser = argparse.ArgumentParser("chatglm-convert")
    parser.add_argument(
        "-i",
        "--model_name_or_path",
        default="THUDM/chatglm-6b",
        type=str,
        help="Model name or path used in AutoModel.from_pretrained",
    )
    parser.add_argument(
        "-l",
        "--lora_model_name_or_path",
        default=None,
        type=str,
        help="Lora model name or path used in PeftModel.from_pretrained",
    )
    parser.add_argument(
        "-o", "--save_path", default="models/chatglm-ggml.bin", type=Path, help="Path to save the generated GGML model"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="q4_0",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="GGML model quantization type",
    )
    parser.add_argument(
        "-vt",
        "--vision_type",
        default="f16",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="Vision model quantization type",
    )
    args = parser.parse_args()

    with open(args.save_path, "wb") as f:
        convert(
            f, args.model_name_or_path, args.lora_model_name_or_path, dtype=args.type, vision_dtype=args.vision_type
        )

    print(f"GGML model saved to {args.save_path}")


if __name__ == "__main__":
    main()
