#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

hf_model=THUDM/chatglm3-6b
ggml_model=models/chatglm3-ggml.bin
benchmark=Benchmark.ChatGLM2

# ChatGLM4-9B
hf_model=THUDM/glm-4-9b-chat
ggml_model=models/chatglm4-ggml.bin
benchmark=Benchmark.ChatGLM4

for dtype in f16 q8_0 q5_1 q5_0 q4_1 q4_0; do
    python3 chatglm_cpp/convert.py -i $hf_model -o $ggml_model -t $dtype
    for use_cuda in OFF ON; do
        cmake -B build -DGGML_CUDA=$use_cuda && cmake --build build -j
        for i in $(seq 3); do
            echo "[benchmark] dtype=$dtype use_cuda=$use_cuda round=$i"
            ./build/bin/chatglm_test --gtest_filter="$benchmark"
        done
    done
done
