#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# ChatGLM3-6B-Base
hf_model=THUDM/chatglm3-6b-base
ggml_model=chatglm3-base-ggml.bin

# Baichuan2-7B-Base
# hf_model=baichuan-inc/Baichuan2-7B-Base
# ggml_model=baichuan2-7b-base-ggml.bin

# InternLM
# hf_model=internlm/internlm-7b
# ggml_model=internlm-7b-base-ggml.bin

for dtype in f16; do
    python3 chatglm_cpp/convert.py -i $hf_model -o $ggml_model -t $dtype
    echo "[perplexity] dtype=$dtype"
    ./build/bin/perplexity -m $ggml_model -f data/wikitext-2-raw/wiki.test.raw -s 512 -l 2048
done
