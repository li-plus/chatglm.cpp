"""
Convert Hugging Face ChatGLM models to GGML format
"""
import argparse
import platform
import struct
import sys
from enum import Enum
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

GGML_QK8_0 = 32
GGML_QK4_0 = 32

GGML_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()


class GgmlType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8


TORCH_TO_GGML_DTYPE = {
    torch.float32: GgmlType.F32,
    torch.float16: GgmlType.F16,
    torch.int8: GgmlType.Q8_0,
}


def dump_config(f, config, ggml_type):
    config_values = [
        config.vocab_size,
        config.hidden_size,
        config.num_attention_heads,
        config.num_layers,
        config.max_sequence_length,
        config.bos_token_id,
        config.eos_token_id,
        config.gmask_token_id,
        config.mask_token_id,
        config.pad_token_id,
        ggml_type.value,
    ]
    f.write(struct.pack("i" * len(config_values), *config_values))


def dump_tokenizer(f, tokenizer):
    serialized_model_proto = tokenizer.sp_tokenizer.text_tokenizer.sp.serialized_model_proto()
    f.write(struct.pack("i", len(serialized_model_proto)))
    f.write(serialized_model_proto)


def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GgmlType):
    assert tensor.dtype == torch.float32

    # header
    f.write(
        struct.pack(
            "i" * (3 + tensor.ndim),
            tensor.ndim,
            len(name.encode()),
            ggml_type.value,
            *tensor.shape,
        )
    )
    f.write(name.encode())

    # data
    if ggml_type == GgmlType.F32:
        tensor = tensor.float()
    elif ggml_type == GgmlType.F16:
        tensor = tensor.half()
    elif ggml_type == GgmlType.Q8_0:
        tensor = quantize_q8_0(tensor)
    elif ggml_type == GgmlType.Q4_0:
        tensor = quantize_q4_0(tensor)
    else:
        raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)


def dump_model(f, model, ggml_type):
    weight_names = ["transformer.word_embeddings.weight"]
    for i in range(model.config.num_layers):
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

    tensor_info = []
    state_dict = model.state_dict()
    for name in tqdm(weight_names, desc="Dumping model state"):
        tensor = state_dict[name]
        if tensor.ndim == 2:
            # 2d weight: should quantize it if needed

            # step 1: de-quantize it back to float32
            if tensor.dtype == torch.int8:
                assert model.config.quantization_bit in [4, 8]
                scale = state_dict[f"{name}_scale"].float()  # channel-wise scale

                if model.config.quantization_bit == 4:
                    # convert int4 weight to int8
                    low_bits = ((tensor << 4) & 0xF0) >> 4
                    high_bits = (tensor & 0xF0) >> 4
                    tensor = torch.stack((high_bits, low_bits), dim=-1).view(tensor.shape[0], -1)
                tensor = tensor * scale[:, None]
            else:
                tensor = tensor.float()

            # step 2: quantize it into ggml format
            tensor_ggml_type = ggml_type
        else:
            # 1d weight: convert it to float32
            assert tensor.ndim == 1
            tensor = tensor.float()
            tensor_ggml_type = GgmlType.F32

        dump_tensor(f, name, tensor, tensor_ggml_type)
        tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


def main():
    parser = argparse.ArgumentParser("chatglm-convert")
    parser.add_argument("-i", "--model_name_or_path", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("-l", "--lora_model_name_or_path", type=str, default=None)
    parser.add_argument("-o", "--save_path", type=Path, default="chatglm-ggml.bin")
    parser.add_argument("-t", "--type", type=str, default="q4_0", choices=["f32", "f16", "q8_0", "q4_0"])
    args = parser.parse_args()

    ggml_type = GgmlType[args.type.upper()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.lora_model_name_or_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_model_name_or_path)
        model = model.merge_and_unload()

    # check config
    assert model.config.position_encoding_2d, "unimplemented: position_encoding_2d should be True"
    assert (
        model.config.inner_hidden_size == 4 * model.config.hidden_size
    ), "unimplemented: inner_hidden_size should be 4 times hidden_size"
    assert torch.allclose(
        model.state_dict()["transformer.word_embeddings.weight"], model.state_dict()["lm_head.weight"]
    ), "unimplemented: lm_head weight must be tied to input embedding"

    # convert all weights to fp16
    with open(args.save_path, "wb") as f:
        f.write(b"ggml")  # magic
        dump_config(f, model.config, ggml_type)
        dump_tokenizer(f, tokenizer)
        dump_model(f, model, ggml_type)
    print(f"GGML model saved to {args.save_path}")


if __name__ == "__main__":
    main()
