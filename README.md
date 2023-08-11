# ChatGLM.cpp

[![CMake](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml/badge.svg)](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml)
[![Python package](https://github.com/li-plus/chatglm.cpp/actions/workflows/python-package.yml/badge.svg)](https://github.com/li-plus/chatglm.cpp/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/chatglm-cpp)](https://pypi.org/project/chatglm-cpp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

C++ implementation of [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) for real-time chatting on your MacBook.

![demo](docs/demo.gif)

## Features

Highlights:
* [x] Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp).
* [x] Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing.
* [x] Streaming generation with typewriter effect.
* [x] Python binding, web demo, and more possibilities.

Support Matrix:
* Hardwares: x86/arm CPU, NVIDIA GPU, Apple Silicon GPU
* Platforms: Linux, MacOS, Windows
* Models: ChatGLM, ChatGLM2, CodeGeeX2

## Getting Started

**Preparation**

Clone the ChatGLM.cpp repository into your local machine:
```sh
git clone --recursive https://github.com/li-plus/chatglm.cpp.git && cd chatglm.cpp
```

If you forgot the `--recursive` flag when cloning the repository, run the following command in the `chatglm.cpp` folder:
```sh
git submodule update --init --recursive
```

**Quantize Model**

Use `convert.py` to transform ChatGLM-6B or ChatGLM2-6B into quantized GGML format. For example, to convert the fp16 original model to q4_0 (quantized int4) GGML model, run:
```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm-6b -t q4_0 -o chatglm-ggml.bin
```

The original model (`-i <model_name_or_path>`) can be a HuggingFace model name or a local path to your pre-downloaded model. Currently supported models are:
* [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B): `THUDM/chatglm-6b`, `THUDM/chatglm-6b-int8`, `THUDM/chatglm-6b-int4`
* [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B): `THUDM/chatglm2-6b`, `THUDM/chatglm2-6b-int4`
* [CodeGeeX2](https://github.com/THUDM/CodeGeeX2): `THUDM/codegeex2-6b`, `THUDM/codegeex2-6b-int4`

You are free to try any of the below quantization types by specifying `-t <type>`:
* `q4_0`: 4-bit integer quantization with fp16 scales.
* `q4_1`: 4-bit integer quantization with fp16 scales and minimum values.
* `q5_0`: 5-bit integer quantization with fp16 scales.
* `q5_1`: 5-bit integer quantization with fp16 scales and minimum values.
* `q8_0`: 8-bit integer quantization with fp16 scales.
* `f16`: half precision floating point weights without quantization.
* `f32`: single precision floating point weights without quantization.

For LoRA model, add `-l <lora_model_name_or_path>` flag to merge your LoRA weights into the base model.

**Build & Run**

Compile the project using CMake:
```sh
cmake -B build
cmake --build build -j --config Release
```

Now you may chat with the quantized ChatGLM-6B model by running:
```sh
./build/bin/main -m chatglm-ggml.bin -p 你好
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```

To run the model in interactive mode, add the `-i` flag. For example:
```sh
./build/bin/main -m chatglm-ggml.bin -i
```
In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

**Try Other Models**

* ChatGLM2-6B
```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm2-6b -t q4_0 -o chatglm2-ggml.bin
./build/bin/main -m chatglm2-ggml.bin -p 你好 --top_p 0.8 --temp 0.8
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
```

* CodeGeeX2
```sh
$ python3 chatglm_cpp/convert.py -i THUDM/codegeex2-6b -t q4_0 -o codegeex2-ggml.bin
$ ./build/bin/main -m codegeex2-ggml.bin --temp 0 --mode generate -p "\
# language: Python
# write a bubble sort function
"


def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list


print(bubble_sort([5, 4, 3, 2, 1]))
```

## Using BLAS

BLAS library can be integrated to further accelerate matrix multiplication. However, in some cases, using BLAS may cause performance degradation. Whether to turn on BLAS should depend on the benchmarking result.

**Accelerate Framework**

Accelerate Framework is automatically enabled on macOS. To disable it, add the CMake flag `-DGGML_NO_ACCELERATE=ON`.

**OpenBLAS**

OpenBLAS provides acceleration on CPU. Add the CMake flag `-DGGML_OPENBLAS=ON` to enable it.
```sh
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

**cuBLAS**

cuBLAS uses NVIDIA GPU to accelerate BLAS. Add the CMake flag `-DGGML_CUBLAS=ON` to enable it.
```sh
cmake -B build -DGGML_CUBLAS=ON && cmake --build build -j
```

Note that the current GGML CUDA implementation is really slow. The community is making efforts to optimize it.

**Metal**

MPS (Metal Performance Shaders) allows computation to run on powerful Apple Silicon GPU. Add the CMake flag `-DGGML_METAL=ON` to enable it.
```sh
cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

## Python Binding

The Python binding provides high-level `chat` and `stream_chat` interface similar to the original Hugging Face ChatGLM(2)-6B.

**Installation**

Install from PyPI (recommended): will trigger compilation on your platform.
```sh
pip install -U chatglm-cpp
```

To enable cuBLAS acceleration on NVIDIA GPU:
```sh
CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install -U chatglm-cpp
```

To enable Metal on Apple silicon devices:
```sh
CMAKE_ARGS="-DGGML_METAL=ON" pip install -U chatglm-cpp
```

You may also install from source. Add the corresponding `CMAKE_ARGS` for acceleration.
```sh
# install from the latest source hosted on GitHub
pip install git+https://github.com/li-plus/chatglm.cpp.git@main
# or install from your local source after git cloning the repo
pip install .
```

**Using pre-converted ggml models**

Here is a simple demo that uses `chatglm_cpp.Pipeline` to load the GGML model and chat with it. First enter the examples folder (`cd examples`) and launch a Python interactive shell:
```python
>>> import chatglm_cpp
>>> 
>>> pipeline = chatglm_cpp.Pipeline("../chatglm-ggml.bin")
>>> pipeline.chat(["你好"])
'你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。'
```

To chat in stream, run the below Python example:
```sh
python3 cli_chat.py -m ../chatglm-ggml.bin -i
```

Launch a web demo to chat in your browser:
```sh
python3 web_demo.py -m ../chatglm-ggml.bin
```

![web_demo](docs/web_demo.jpg)

For other models:

* ChatGLM2
```sh
python3 cli_chat.py -m ../chatglm2-ggml.bin -p 你好 --temp 0.8 --top_p 0.8  # CLI demo
python3 web_demo.py -m ../chatglm2-ggml.bin --temp 0.8 --top_p 0.8  # web demo
```

* CodeGeeX2
```sh
# CLI demo
python3 cli_chat.py -m ../codegeex2-ggml.bin --temp 0 --mode generate -p "\
# language: Python
# write a bubble sort function
"
# web demo
python3 web_demo.py -m ../codegeex2-ggml.bin --temp 0 --max_length 512 --mode generate --plain
```

**Load and optimize Hugging Face LLMs in one line of code**

Sometimes it might be inconvenient to convert and save the intermediate GGML models beforehand. Here is an option to directly load from the original Hugging Face model, quantize it into GGML models in a minute, and start serving. All you need is to replace the GGML model path with the Hugging Face model name or path.
```python
>>> import chatglm_cpp
>>> 
>>> pipeline = chatglm_cpp.Pipeline("THUDM/chatglm-6b", dtype="q4_0")
Loading checkpoint shards: 100%|█████████████████████████████████████████████| 8/8 [00:10<00:00,  1.27s/it]
Processing model states: 100%|███████████████████████████████████████████| 339/339 [00:23<00:00, 14.73it/s]
...
>>> pipeline.chat(["你好"])
'你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。'
```

Likewise, replace the GGML model path with Hugging Face model in any example script, and it just works. For example:
```sh
python3 cli_chat.py -m THUDM/chatglm-6b -p 你好 -i
```

## API Server

We support various kinds of API servers to integrate with popular frontends. Extra dependencies can be installed by:
```sh
pip install 'chatglm-cpp[api]'
```
Remember to add the corresponding `CMAKE_ARGS` to enable acceleration.

**LangChain API**

Start the api server for LangChain:
```sh
MODEL=./chatglm2-ggml.bin uvicorn chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8000
```

Test the api endpoint with `curl`:
```sh
curl http://127.0.0.1:8000 -H 'Content-Type: application/json' -d '{"prompt": "你好"}'
```

Run with LangChain:
```python
>>> from langchain.llms import ChatGLM
>>> 
>>> llm = ChatGLM(endpoint_url="http://127.0.0.1:8000")
>>> llm.predict("你好")
'你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'
```

For more options, please refer to [examples/langchain_client.py](examples/langchain_client.py) and [LangChain ChatGLM Integration](https://python.langchain.com/docs/integrations/llms/chatglm).

**OpenAI API**

Start an API server compatible with OpenAI chat completions protocol:
```sh
MODEL=./chatglm2-ggml.bin uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000
```

Test your endpoint with `curl`:
```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

Use the OpenAI client to make streaming request:
```sh
OPENAI_API_BASE=http://127.0.0.1:8000/v1 python3 examples/openai_client.py --stream --prompt 你好
```

With this API server as backend, ChatGLM.cpp models can be seamlessly integrated into any frontend that uses OpenAI-style API, including [mckaywrigley/chatbot-ui](https://github.com/mckaywrigley/chatbot-ui), [fuergaosi233/wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt), [Yidadaa/ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web), and more.

## Using Docker

```sh
docker run -it --rm -v [model path]:/opt/ chulinx/chatglm /chatglm -m /opt/chatglm2-ggml.bin -p "你好啊"
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
```

## Performance

Environment:
* CPU backend performance is measured on a Linux server with Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz using 16 threads.
* CUDA backend is measured on a V100-SXM2-32GB GPU using 1 thread.
* MPS backend is measured on an Apple M2 Ultra device using 1 thread (currently only supports ChatGLM2).

ChatGLM-6B:

|                                | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0  | F16   | F32   |
|--------------------------------|-------|-------|-------|-------|-------|-------|-------|
| ms/token (CPU @ Platinum 8260) | 74    | 77    | 86    | 89    | 114   | 189   | 357   |
| ms/token (CUDA @ V100 SXM2)    | 10.0  | 9.8   | 10.7  | 10.6  | 14.6  | 19.8  | 34.2  |
| file size                      | 3.3GB | 3.7GB | 4.0GB | 4.4GB | 6.2GB | 12GB  | 23GB  |
| mem usage                      | 4.0GB | 4.4GB | 4.7GB | 5.1GB | 6.9GB | 13GB  | 24GB  |

ChatGLM2-6B:

|                                | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0  | F16   | F32   |
|--------------------------------|-------|-------|-------|-------|-------|-------|-------|
| ms/token (CPU @ Platinum 8260) | 64    | 71    | 79    | 83    | 106   | 189   | 372   |
| ms/token (CUDA @ V100 SXM2)    | 9.7   | 9.4   | 10.3  | 10.2  | 14.0  | 19.1  | 33.0  |
| ms/token (MPS @ M2 Ultra)      | 11.0  | 11.7  | N/A   | N/A   | N/A   | 32.1  | N/A   |
| file size                      | 3.3GB | 3.7GB | 4.0GB | 4.4GB | 6.2GB | 12GB  | 24GB  |
| mem usage                      | 3.4GB | 3.8GB | 4.1GB | 4.5GB | 6.2GB | 12GB  | 23GB  |

## Development

* To perform unit tests, add the CMake flag `-DCHATGLM_ENABLE_TESTING=ON`, recompile, and run `./build/bin/chatglm_test`. For benchmark only, run `./build/bin/chatglm_test --gtest_filter=ChatGLM.benchmark`.
* To format the code, run `cmake --build build --target lint`. You should have `clang-format`, `black` and `isort` pre-installed.
* To check performance issue, add the CMake flag `-DGGML_PERF=ON`. It will show timing for each graph operation when running the model.

## Acknowledgements

* This project is greatly inspired by [@ggerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp) and is based on his NN library [ggml](https://github.com/ggerganov/ggml).
* Thank [@THUDM](https://github.com/THUDM) for the amazing [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) and for releasing the model sources and checkpoints.
