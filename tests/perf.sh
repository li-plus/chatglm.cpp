#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# InternLM-7B
hf_model=internlm/internlm-chat-7b
ggml_model=models/internlm-chat-7b-ggml.bin
benchmark=Benchmark.InternLM7B

# InternLM-20B
# hf_model=internlm/internlm-chat-20b
# ggml_model=models/internlm-chat-20b-ggml.bin
# benchmark=Benchmark.InternLM20B

# ChatGLM4-9B
hf_model=THUDM/glm-4-9b-chat
ggml_model=models/chatglm4-ggml.bin
benchmark=Benchmark.ChatGLM4

for dtype in f16 q8_0 q5_1 q5_0 q4_1 q4_0; do
    python3 chatglm_cpp/convert.py -i $hf_model -o $ggml_model -t $dtype
    for use_cublas in ON; do
        cmake -B build -DGGML_CUBLAS=$use_cublas && cmake --build build -j
        for i in $(seq 3); do
            echo "[benchmark] dtype=$dtype use_cublas=$use_cublas round=$i"
            ./build/bin/chatglm_test --gtest_filter="$benchmark"
        done
    done
done
