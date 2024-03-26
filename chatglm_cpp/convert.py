"""
Convert Hugging Face ChatGLM family models to GGML format
"""

import argparse
import json
import platform
import struct
import sys
from enum import Enum, IntEnum
from pathlib import Path
from typing import BinaryIO, Optional

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

GGML_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()  # type: ignore


class GGMLType(IntEnum):
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
    BAICHUAN7B = 1024
    BAICHUAN13B = 1025
    INTERNLM = 1280


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


def quantize_tensor(tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32
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
        raise NotImplementedError(f"Cannot quantize tensor to dtype {ggml_type}")
    return tensor


def dump_tensor(f, tensor: torch.Tensor, ggml_type: GGMLType):
    tensor = quantize_tensor(tensor, ggml_type)
    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)


def dump_state_dict(f, tensor_meta, state_dict, quantization_bit):
    tensor_info = []
    for meta in tqdm(tensor_meta, desc="Processing model states"):
        name = meta["name"]
        target_dtype = meta["dtype"]
        target_shape = meta["shape"]

        tensor = state_dict[name]
        if tensor.ndim == 2:
            # 2d weight: should quantize it if needed, de-quantize it back to float32 first
            if tensor.dtype == torch.int8:
                assert quantization_bit in [4, 8]
                scale = state_dict[f"{name}_scale"].float()  # channel-wise scale

                if quantization_bit == 4:
                    # convert int4 weight to int8
                    low_bits = (tensor << 4) >> 4
                    high_bits = tensor >> 4
                    tensor = torch.stack((high_bits, low_bits), dim=-1).view(tensor.shape[0], -1)
                tensor = tensor * scale[:, None]
            else:
                tensor = tensor.float()
        else:
            # 1d weight: convert it to float32
            assert tensor.ndim == 1
            tensor = tensor.float()
            assert target_dtype == GGMLType.F32

        assert tensor.shape == target_shape

        dump_tensor(f, tensor, target_dtype)
        tensor_info.append((name, tensor.shape, target_dtype.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


class BaseConverter:
    @classmethod
    def convert(cls, f, model, tokenizer, ggml_type):
        f.write(b"ggml")  # magic
        f.write(struct.pack("I", 4))  # version 4
        cls.dump(f, tokenizer, model, ggml_type)


class ChatGLMConverter(BaseConverter):
    @classmethod
    def dump(cls, f, tokenizer, model, ggml_type):
        config = model.config
        assert config.position_encoding_2d, "unimplemented: position_encoding_2d should be True"
        assert (
            config.inner_hidden_size == 4 * config.hidden_size
        ), "unimplemented: inner_hidden_size should be 4 times hidden_size"

        assert torch.allclose(
            model.state_dict()["transformer.word_embeddings.weight"], model.state_dict()["lm_head.weight"]
        ), "unimplemented: lm_head weight must be tied to input embedding"

        tensor_meta = cls.get_tensor_meta(model, ggml_type)

        header_config = dict(
            tokenizer=dict(
                type="spm", model_size=len(tokenizer.sp_tokenizer.text_tokenizer.sp.serialized_model_proto())
            ),
            model=dict(
                model_type="chatglm",
                dtype=ggml_type,
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_attention_heads,
                num_hidden_layers=config.num_layers,
                intermediate_size=config.inner_hidden_size,
                norm_eps=config.layernorm_epsilon,
                max_length=config.max_sequence_length,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.pad_token_id,
            ),
            tensor_meta=tensor_meta,
        )

        # write header config
        header_config_bytes = json.dumps(header_config, separators=(",", ":")).encode()
        f.write(struct.pack("I", len(header_config_bytes)))
        f.write(header_config_bytes)

        # write tokenizer model
        f.write(tokenizer.sp_tokenizer.text_tokenizer.sp.serialized_model_proto())

        # write model weights
        dump_state_dict(f, tensor_meta, model.state_dict(), config.quantization_bit)

    @staticmethod
    def get_tensor_meta(model, ggml_type):
        config = model.config

        weight_names = ["transformer.word_embeddings.weight"]
        for i in range(config.num_layers):
            weight_names += [
                f"transformer.layers.{i}.input_layernorm.weight",
                f"transformer.layers.{i}.input_layernorm.bias",
                f"transformer.layers.{i}.attention.query_key_value.weight",
                f"transformer.layers.{i}.attention.query_key_value.bias",
                f"transformer.layers.{i}.attention.dense.weight",
                f"transformer.layers.{i}.attention.dense.bias",
                f"transformer.layers.{i}.post_attention_layernorm.weight",
                f"transformer.layers.{i}.post_attention_layernorm.bias",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.layers.{i}.mlp.dense_h_to_4h.bias",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.weight",
                f"transformer.layers.{i}.mlp.dense_4h_to_h.bias",
            ]
        weight_names += [
            "transformer.final_layernorm.weight",
            "transformer.final_layernorm.bias",
        ]

        state_dict = model.state_dict()
        tensor_meta = []
        for name in weight_names:
            weight = state_dict[name]
            shape = weight.shape
            if weight.ndim == 2:
                dtype = ggml_type
                if config.quantization_bit == 4:
                    shape[-1] *= 2
            else:
                dtype = GGMLType.F32
            tensor_meta.append(dict(name=name, dtype=dtype, shape=shape))

        return tensor_meta


class ChatGLM2Converter(BaseConverter):
    MODEL_TYPE = ModelType.CHATGLM2

    @staticmethod
    def dump_config(f, config, ggml_type):
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

        config_values = [
            ggml_type.value,
            config.padded_vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_layers,
            config.ffn_hidden_size,
            config.seq_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
            config.multi_query_group_num,
        ]

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.tokenizer.sp_model.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ggml_type):
        weight_names = ["transformer.embedding.word_embeddings.weight"]
        for i in range(model.config.num_layers):
            weight_names += [
                f"transformer.encoder.layers.{i}.input_layernorm.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight",
                f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias",
                f"transformer.encoder.layers.{i}.self_attention.dense.weight",
                f"transformer.encoder.layers.{i}.post_attention_layernorm.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight",
                f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight",
            ]
        weight_names += [
            "transformer.encoder.final_layernorm.weight",
            "transformer.output_layer.weight",
        ]
        dump_state_dict(f, weight_names, model.state_dict(), model.config.quantization_bit, ggml_type)


class ChatGLM3Converter(ChatGLM2Converter):
    MODEL_TYPE = ModelType.CHATGLM3


class BaichuanConverter(BaseConverter):
    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == "silu", "unimplemented: hidden_act must be silu"

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.model_max_length,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.sp_model.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ggml_type):
        weight_names = ["model.embed_tokens.weight"]
        for i in range(model.config.num_hidden_layers):
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.W_pack.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
            ]
        weight_names += [
            "model.norm.weight",
            "lm_head.weight",
        ]

        if model.config.vocab_size == 125696:
            # For Baichuan2, normalize lm_head weight
            model.lm_head.weight.data = F.normalize(model.lm_head.weight.data)

        dump_state_dict(f, weight_names, model.state_dict(), quantization_bit=None, ggml_type=ggml_type)


class Baichuan7BConverter(BaichuanConverter):
    MODEL_TYPE = ModelType.BAICHUAN7B


class Baichuan13BConverter(BaichuanConverter):
    MODEL_TYPE = ModelType.BAICHUAN13B


class InternLMConverter(BaseConverter):
    MODEL_TYPE = ModelType.INTERNLM

    @staticmethod
    def dump_config(f, config, ggml_type):
        assert config.hidden_act == "silu", "unimplemented: hidden_act must be silu"

        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.max_position_embeddings,
            config.bos_token_id if config.bos_token_id is not None else -1,
            config.eos_token_id if config.eos_token_id is not None else -1,
            config.pad_token_id if config.pad_token_id is not None else -1,
            config.sep_token_id if config.sep_token_id is not None else -1,
        ]

        f.write(struct.pack("i" * len(config_values), *config_values))

    @staticmethod
    def dump_tokenizer(f, tokenizer):
        serialized_model_proto = tokenizer.sp_model.serialized_model_proto()
        f.write(struct.pack("i", len(serialized_model_proto)))
        f.write(serialized_model_proto)

    @staticmethod
    def dump_model(f, model, ggml_type):
        state_dict = model.state_dict()
        for i in range(model.config.num_hidden_layers):
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
                (
                    state_dict[f"model.layers.{i}.self_attn.q_proj.weight"],
                    state_dict[f"model.layers.{i}.self_attn.k_proj.weight"],
                    state_dict[f"model.layers.{i}.self_attn.v_proj.weight"],
                ),
                dim=0,
            )
            if model.config.bias:
                state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                    (
                        state_dict[f"model.layers.{i}.self_attn.q_proj.bias"],
                        state_dict[f"model.layers.{i}.self_attn.k_proj.bias"],
                        state_dict[f"model.layers.{i}.self_attn.v_proj.bias"],
                    ),
                    dim=0,
                )

        weight_names = ["model.embed_tokens.weight"]
        for i in range(model.config.num_hidden_layers):
            optional_qkv_proj_bias = [f"model.layers.{i}.self_attn.qkv_proj.bias"] if model.config.bias else []
            optional_o_proj_bias = [f"model.layers.{i}.self_attn.o_proj.bias"] if model.config.bias else []
            weight_names += [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.qkv_proj.weight",
                *optional_qkv_proj_bias,
                f"model.layers.{i}.self_attn.o_proj.weight",
                *optional_o_proj_bias,
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
            ]
        weight_names += [
            "model.norm.weight",
            "lm_head.weight",
        ]

        dump_state_dict(f, weight_names, state_dict, quantization_bit=None, ggml_type=ggml_type)


def convert(f: BinaryIO, model_name_or_path: str, lora_model_name_or_path: Optional[str] = None, dtype: str = "q4_0"):
    ggml_type = GGMLType[dtype.upper()]

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

    if model.config.model_type == "chatglm":
        if hasattr(model.config, "multi_query_attention"):
            # ChatGLM3 shares the same architecture and model config with ChatGLM2, but its tokenizer further supports system prompts,
            # so we can check system token to discriminate ChatGLM3 from ChatGLM2.
            if "<|system|>" not in tokenizer.tokenizer.special_tokens:
                ChatGLM2Converter.convert(f, model, tokenizer, ggml_type)
            else:
                ChatGLM3Converter.convert(f, model, tokenizer, ggml_type)
        else:
            ChatGLMConverter.convert(f, model, tokenizer, ggml_type)
    elif model.config.model_type == "baichuan":
        if model.config.hidden_size == 5120:
            Baichuan13BConverter.convert(f, model, tokenizer, ggml_type)
        else:
            Baichuan7BConverter.convert(f, model, tokenizer, ggml_type)
    elif model.config.model_type == "internlm":
        InternLMConverter.convert(f, model, tokenizer, ggml_type)
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
        "-o", "--save_path", default="chatglm-ggml.bin", type=Path, help="Path to save the generated GGML model"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="q4_0",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="GGML model quantization type",
    )
    args = parser.parse_args()

    with open(args.save_path, "wb") as f:
        convert(f, args.model_name_or_path, args.lora_model_name_or_path, dtype=args.type)

    print(f"GGML model saved to {args.save_path}")


if __name__ == "__main__":
    import sys

    sys.argv[1:] = "-i THUDM/chatglm-6b -t q4_0 -o chatglm-ggml.bin".split()

    main()
