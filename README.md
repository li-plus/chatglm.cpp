# ChatGLM.cpp

[![CMake](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml/badge.svg)](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A C++ implementation of [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B). Run int4 model inference on your macBook in a minute.

![demo](docs/demo.gif)

## Features

* [x] Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp).
* [x] Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing.
* [x] Streaming generation with typewriter effect.
* [ ] TODO: GPU support, python binding, and web demo.

## Getting Started

1. Clone the ChatGLM.cpp repository into your local machine:
```sh
git clone --recursive https://github.com/li-plus/chatglm.cpp.git && cd chatglm.cpp
```

2. If you forgot the `--recursive` flag when cloning the repository, run the following command in the `chatglm.cpp` folder:
```sh
git submodule update --init --recursive
```

3. Convert ChatGLM-6B to GGML format using the `convert.py` script. For example, to convert the fp16 ChatGLM-6B model to q4_0 (quantized int4) GGML model, run:
```sh
python3 convert.py -i THUDM/chatglm-6b -t q4_0 -o chatglm-ggml.bin
```

4. Compile the project using CMake:
```sh
cmake -S . -B build
cmake --build build -j
```

5. Chat with the quantized ChatGLM model by running:
```sh
./build/bin/main -m chatglm-ggml.bin -q 你好
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```

6. To run ChatGLM in interactive mode, add the `-i` flag:
```sh
./build/bin/main -m chatglm-ggml.bin -i
```
In interactive mode, your chat history will serve as the context for the next-round conversation.

7. Run `./build/bin/main -h` to explore more options!

## Using CUDA

Use cuBLAS to accelerate matrix multiplication. Note that the current GGML CUDA implementation is really slow. The community is making efforts to optimize it.
```sh
cmake -S . -B build -DGGML_CUBLAS=ON
cmake --build build -j
```

## Performance

Measured on a Linux server with Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz using 16 threads.

|           | Q4_0  | Q8_0  | F16  | F32  |
|-----------|-------|-------|------|------|
| ms/token  | 92    | 130   | 217  | 399  |
| file size | 3.3GB | 6.2GB | 12GB | 23GB |
| mem usage | 4.2GB | 7.1GB | 13GB | 24GB |

## For Developers

* To perform unit tests, add the CMake flag `-DCHATGLM_ENABLE_TESTING=ON`, recompile, and run `./build/bin/chatglm_test`. For benchmark only, run `./build/bin/chatglm_test --gtest_filter=ChatGLM.benchmark`.
* To format the code, run `cmake --build build --target lint`. You should have `clang-format`, `black` and `isort` pre-installed.
* To check performance issue, add the CMake flag `-DGGML_PERF=ON`. You will see timing for each graph operation.

## Acknowledgements

* This project is greatly inspired by [ggerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp) and is based on his NN library [ggml](https://github.com/ggerganov/ggml).
* Thank [THUDM](https://github.com/THUDM) for the amazing [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and for releasing the model sources and checkpoints.
