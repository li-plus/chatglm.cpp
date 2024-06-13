#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# ChatGLM3-6B-Base
hf_model=THUDM/chatglm3-6b-base
ggml_model=models/chatglm3-base-ggml.bin

# ChatGLM4-9B-Base
hf_model=THUDM/glm-4-9b
ggml_model=models/chatglm4-base-ggml.bin

for dtype in f16 q8_0 q5_1 q5_0 q4_1 q4_0; do
    python3 chatglm_cpp/convert.py -i $hf_model -o $ggml_model -t $dtype
    echo "[perplexity] dtype=$dtype"
    ./build/bin/perplexity -m $ggml_model -f data/wikitext-2-raw/wiki.test.raw -s 512 -l 2048
done
