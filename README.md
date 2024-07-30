# ChatGLM.cpp

[![CMake](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml/badge.svg)](https://github.com/li-plus/chatglm.cpp/actions/workflows/cmake.yml)
[![Python package](https://github.com/li-plus/chatglm.cpp/actions/workflows/python-package.yml/badge.svg)](https://github.com/li-plus/chatglm.cpp/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/chatglm-cpp)](https://pypi.org/project/chatglm-cpp/)
![Python](https://img.shields.io/pypi/pyversions/chatglm-cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

C++ implementation of [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B), [ChatGLM3](https://github.com/THUDM/ChatGLM3) and [GLM-4](https://github.com/THUDM/GLM-4)(V) for real-time chatting on your MacBook.

![demo](docs/demo.gif)

## Features

Highlights:
* Pure C++ implementation based on [ggml](https://github.com/ggerganov/ggml), working in the same way as [llama.cpp](https://github.com/ggerganov/llama.cpp).
* Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing.
* P-Tuning v2 and LoRA finetuned models support.
* Streaming generation with typewriter effect.
* Python binding, web demo, api servers and more possibilities.

Support Matrix:
* Hardwares: x86/arm CPU, NVIDIA GPU, Apple Silicon GPU
* Platforms: Linux, MacOS, Windows
* Models: [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B), [ChatGLM3](https://github.com/THUDM/ChatGLM3), [GLM-4](https://github.com/THUDM/GLM-4)(V), [CodeGeeX2](https://github.com/THUDM/CodeGeeX2)

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

Install necessary packages for loading and quantizing Hugging Face models:
```sh
python3 -m pip install -U pip
python3 -m pip install torch tabulate tqdm transformers accelerate sentencepiece
```

Use `convert.py` to transform ChatGLM-6B into quantized GGML format. For example, to convert the fp16 original model to q4_0 (quantized int4) GGML model, run:
```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm-6b -t q4_0 -o models/chatglm-ggml.bin
```

The original model (`-i <model_name_or_path>`) can be a Hugging Face model name or a local path to your pre-downloaded model. Currently supported models are:
* ChatGLM-6B: `THUDM/chatglm-6b`, `THUDM/chatglm-6b-int8`, `THUDM/chatglm-6b-int4`
* ChatGLM2-6B: `THUDM/chatglm2-6b`, `THUDM/chatglm2-6b-int4`, `THUDM/chatglm2-6b-32k`, `THUDM/chatglm2-6b-32k-int4`
* ChatGLM3-6B: `THUDM/chatglm3-6b`, `THUDM/chatglm3-6b-32k`, `THUDM/chatglm3-6b-128k`, `THUDM/chatglm3-6b-base`
* ChatGLM4(V)-9B: `THUDM/glm-4-9b-chat`, `THUDM/glm-4-9b-chat-1m`, `THUDM/glm-4-9b`, `THUDM/glm-4v-9b`
* CodeGeeX2: `THUDM/codegeex2-6b`, `THUDM/codegeex2-6b-int4`

You are free to try any of the below quantization types by specifying `-t <type>`:
| type   | precision | symmetric |
| ------ | --------- | --------- |
| `q4_0` | int4      | true      |
| `q4_1` | int4      | false     |
| `q5_0` | int5      | true      |
| `q5_1` | int5      | false     |
| `q8_0` | int8      | true      |
| `f16`  | half      |           |
| `f32`  | float     |           |

For LoRA models, add `-l <lora_model_name_or_path>` flag to merge your LoRA weights into the base model. For example, run `python3 chatglm_cpp/convert.py -i THUDM/chatglm3-6b -t q4_0 -o models/chatglm3-ggml-lora.bin -l shibing624/chatglm3-6b-csc-chinese-lora` to merge public LoRA weights from Hugging Face.

For P-Tuning v2 models using the [official finetuning script](https://github.com/THUDM/ChatGLM3/tree/main/finetune_demo), additional weights are automatically detected by `convert.py`. If `past_key_values` is on the output weight list, the P-Tuning checkpoint is successfully converted.

**Build & Run**

Compile the project using CMake:
```sh
cmake -B build
cmake --build build -j --config Release
```

Now you may chat with the quantized ChatGLM-6B model by running:
```sh
./build/bin/main -m models/chatglm-ggml.bin -p 你好
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
```

To run the model in interactive mode, add the `-i` flag. For example:
```sh
./build/bin/main -m models/chatglm-ggml.bin -i
```
In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

**Try Other Models**

<details open>
<summary>ChatGLM2-6B</summary>

```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm2-6b -t q4_0 -o models/chatglm2-ggml.bin
./build/bin/main -m models/chatglm2-ggml.bin -p 你好 --top_p 0.8 --temp 0.8
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
```
</details>

<details open>
<summary>ChatGLM3-6B</summary>

ChatGLM3-6B further supports function call and code interpreter in addition to chat mode.

Chat mode:
```sh
python3 chatglm_cpp/convert.py -i THUDM/chatglm3-6b -t q4_0 -o models/chatglm3-ggml.bin
./build/bin/main -m models/chatglm3-ggml.bin -p 你好 --top_p 0.8 --temp 0.8
# 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。
```

Setting system prompt:
```sh
./build/bin/main -m models/chatglm3-ggml.bin -p 你好 -s "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
# 你好👋！我是 ChatGLM3，有什么问题可以帮您解答吗？
```

Function call:
~~~
$ ./build/bin/main -m models/chatglm3-ggml.bin --top_p 0.8 --temp 0.8 --sp examples/system/function_call.txt -i
System   > Answer the following questions as best as you can. You have access to the following tools: ...
Prompt   > 生成一个随机数
ChatGLM3 > random_number_generator
```python
tool_call(seed=42, range=(0, 100))
```
Tool Call   > Please manually call function `random_number_generator` with args `tool_call(seed=42, range=(0, 100))` and provide the results below.
Observation > 23
ChatGLM3 > 根据您的要求，我使用随机数生成器API生成了一个随机数。根据API返回结果，生成的随机数为23。
~~~

Code interpreter:
~~~
$ ./build/bin/main -m models/chatglm3-ggml.bin --top_p 0.8 --temp 0.8 --sp examples/system/code_interpreter.txt -i
System   > 你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。
Prompt   > 列出100以内的所有质数
ChatGLM3 > 好的，我会为您列出100以内的所有质数。
```python
def is_prime(n):
   """Check if a number is prime."""
   if n <= 1:
       return False
   if n <= 3:
       return True
   if n % 2 == 0 or n % 3 == 0:
       return False
   i = 5
   while i * i <= n:
       if n % i == 0 or n % (i + 2) == 0:
           return False
       i += 6
   return True

primes_upto_100 = [i for i in range(2, 101) if is_prime(i)]
primes_upto_100
```

Code Interpreter > Please manually run the code and provide the results below.
Observation      > [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
ChatGLM3 > 100以内的所有质数为：

$$
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 
$$
~~~

</details>

<details open>
<summary>ChatGLM4-9B</summary>

Chat mode:
```sh
python3 chatglm_cpp/convert.py -i THUDM/glm-4-9b-chat -t q4_0 -o models/chatglm4-ggml.bin
./build/bin/main -m models/chatglm4-ggml.bin -p 你好 --top_p 0.8 --temp 0.8
# 你好👋！有什么可以帮助你的吗？
```

</details>

<details open>
<summary>ChatGLM4V-9B</summary>

[![03-Confusing-Pictures](examples/03-Confusing-Pictures.jpg)](https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg)

You may use `-vt <vision_type>` to set quantization type for the vision encoder. It is recommended to run GLM4V on GPU since vision encoding runs too slow on CPU even with 4-bit quantization.
```sh
python3 chatglm_cpp/convert.py -i THUDM/glm-4v-9b -t q4_0 -vt q4_0 -o models/chatglm4v-ggml.bin
./build/bin/main -m models/chatglm4v-ggml.bin --image examples/03-Confusing-Pictures.jpg -p "这张图片有什么不寻常的地方" --temp 0
# 这张图片中不寻常的地方在于，男子正在一辆黄色出租车后面熨衣服。通常情况下，熨衣是在家中或洗衣店进行的，而不是在车辆上。此外，出租车在行驶中，男子却能够稳定地熨衣，这增加了场景的荒诞感。
```

</details>

<details>
<summary>CodeGeeX2</summary>

```sh
$ python3 chatglm_cpp/convert.py -i THUDM/codegeex2-6b -t q4_0 -o models/codegeex2-ggml.bin
$ ./build/bin/main -m models/codegeex2-ggml.bin --temp 0 --mode generate -p "\
# language: Python
# write a bubble sort function
"


def bubble_sort(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


print(bubble_sort([5, 4, 3, 2, 1]))
```
</details>

## Using BLAS

BLAS library can be integrated to further accelerate matrix multiplication. However, in some cases, using BLAS may cause performance degradation. Whether to turn on BLAS should depend on the benchmarking result.

**Accelerate Framework**

Accelerate Framework is automatically enabled on macOS. To disable it, add the CMake flag `-DGGML_NO_ACCELERATE=ON`.

**OpenBLAS**

OpenBLAS provides acceleration on CPU. Add the CMake flag `-DGGML_OPENBLAS=ON` to enable it.
```sh
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

**CUDA**

CUDA accelerates model inference on NVIDIA GPU. Add the CMake flag `-DGGML_CUDA=ON` to enable it.
```sh
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
```

By default, all kernels will be compiled for all possible CUDA architectures and it takes some time. To run on a specific type of device, you may specify `CMAKE_CUDA_ARCHITECTURES` to speed up the nvcc compilation. For example:
```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80"       # for A100
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75"    # compatible with both V100 and T4
```

To find out the CUDA architecture of your GPU device, see [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

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

To enable CUDA on NVIDIA GPU:
```sh
CMAKE_ARGS="-DGGML_CUDA=ON" pip install -U chatglm-cpp
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

Pre-built wheels for CPU backend on Linux / MacOS / Windows are published on [release](https://github.com/li-plus/chatglm.cpp/releases). For CUDA / Metal backends, please compile from source code or source distribution.

**Using Pre-converted GGML Models**

Here is a simple demo that uses `chatglm_cpp.Pipeline` to load the GGML model and chat with it. First enter the examples folder (`cd examples`) and launch a Python interactive shell:
```python
>>> import chatglm_cpp
>>> 
>>> pipeline = chatglm_cpp.Pipeline("../models/chatglm-ggml.bin")
>>> pipeline.chat([chatglm_cpp.ChatMessage(role="user", content="你好")])
ChatMessage(role="assistant", content="你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。", tool_calls=[])
```

To chat in stream, run the below Python example:
```sh
python3 cli_demo.py -m ../models/chatglm-ggml.bin -i
```

Launch a web demo to chat in your browser:
```sh
python3 web_demo.py -m ../models/chatglm-ggml.bin
```

![web_demo](docs/web_demo.jpg)

For other models:

<details open>
<summary>ChatGLM2-6B</summary>

```sh
python3 cli_demo.py -m ../models/chatglm2-ggml.bin -p 你好 --temp 0.8 --top_p 0.8  # CLI demo
python3 web_demo.py -m ../models/chatglm2-ggml.bin --temp 0.8 --top_p 0.8  # web demo
```
</details>

<details open>
<summary>ChatGLM3-6B</summary>

**CLI Demo**

Chat mode:
```sh
python3 cli_demo.py -m ../models/chatglm3-ggml.bin -p 你好 --temp 0.8 --top_p 0.8
```

Function call:
```sh
python3 cli_demo.py -m ../models/chatglm3-ggml.bin --temp 0.8 --top_p 0.8 --sp system/function_call.txt -i
```

Code interpreter:
```sh
python3 cli_demo.py -m ../models/chatglm3-ggml.bin --temp 0.8 --top_p 0.8 --sp system/code_interpreter.txt -i
```

**Web Demo**

Install Python dependencies and the IPython kernel for code interpreter.
```sh
pip install streamlit jupyter_client ipython ipykernel
ipython kernel install --name chatglm3-demo --user
```

Launch the web demo:
```sh
streamlit run chatglm3_demo.py
```

| Function Call               | Code Interpreter               |
|-----------------------------|--------------------------------|
| ![](docs/function_call.png) | ![](docs/code_interpreter.png) |

</details>

<details open>
<summary>ChatGLM4-9B</summary>

Chat mode:
```sh
python3 cli_demo.py -m ../models/chatglm4-ggml.bin -p 你好 --temp 0.8 --top_p 0.8
```
</details>

<details open>
<summary>ChatGLM4V-9B</summary>

Chat mode:
```sh
python3 cli_demo.py -m ../models/chatglm4v-ggml.bin --image 03-Confusing-Pictures.jpg -p "这张图片有什么不寻常之处" --temp 0
```
</details>

<details>
<summary>CodeGeeX2</summary>

```sh
# CLI demo
python3 cli_demo.py -m ../models/codegeex2-ggml.bin --temp 0 --mode generate -p "\
# language: Python
# write a bubble sort function
"
# web demo
python3 web_demo.py -m ../models/codegeex2-ggml.bin --temp 0 --max_length 512 --mode generate --plain
```
</details>

**Converting Hugging Face LLMs at Runtime**

Sometimes it might be inconvenient to convert and save the intermediate GGML models beforehand. Here is an option to directly load from the original Hugging Face model, quantize it into GGML models in a minute, and start serving. All you need is to replace the GGML model path with the Hugging Face model name or path.
```python
>>> import chatglm_cpp
>>> 
>>> pipeline = chatglm_cpp.Pipeline("THUDM/chatglm-6b", dtype="q4_0")
Loading checkpoint shards: 100%|██████████████████████████████████| 8/8 [00:10<00:00,  1.27s/it]
Processing model states: 100%|████████████████████████████████| 339/339 [00:23<00:00, 14.73it/s]
...
>>> pipeline.chat([chatglm_cpp.ChatMessage(role="user", content="你好")])
ChatMessage(role="assistant", content="你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。", tool_calls=[])
```

Likewise, replace the GGML model path with Hugging Face model in any example script, and it just works. For example:
```sh
python3 cli_demo.py -m THUDM/chatglm-6b -p 你好 -i
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
MODEL=./models/chatglm2-ggml.bin uvicorn chatglm_cpp.langchain_api:app --host 127.0.0.1 --port 8000
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

Start an API server compatible with [OpenAI chat completions protocol](https://platform.openai.com/docs/api-reference/chat):
```sh
MODEL=./models/chatglm3-ggml.bin uvicorn chatglm_cpp.openai_api:app --host 127.0.0.1 --port 8000
```

Test your endpoint with `curl`:
```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

Use the OpenAI client to chat with your model:
```python
>>> from openai import OpenAI
>>> 
>>> client = OpenAI(base_url="http://127.0.0.1:8000/v1")
>>> response = client.chat.completions.create(model="default-model", messages=[{"role": "user", "content": "你好"}])
>>> response.choices[0].message.content
'你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。'
```

For stream response, check out the example client script:
```sh
python3 examples/openai_client.py --base_url http://127.0.0.1:8000/v1 --stream --prompt 你好
```

Tool calling is also supported:
```sh
python3 examples/openai_client.py --base_url http://127.0.0.1:8000/v1 --tool_call --prompt 上海天气怎么样
```

Request GLM4V with image inputs:
```sh
# request with local image file
python3 examples/openai_client.py --base_url http://127.0.0.1:8000/v1 --prompt "描述这张图片" \
    --image examples/03-Confusing-Pictures.jpg --temp 0
# request with image url
python3 examples/openai_client.py --base_url http://127.0.0.1:8000/v1 --prompt "描述这张图片" \
    --image https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg --temp 0
```

With this API server as backend, ChatGLM.cpp models can be seamlessly integrated into any frontend that uses OpenAI-style API, including [mckaywrigley/chatbot-ui](https://github.com/mckaywrigley/chatbot-ui), [fuergaosi233/wechat-chatgpt](https://github.com/fuergaosi233/wechat-chatgpt), [Yidadaa/ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web), and more.

## Using Docker

**Option 1: Building Locally**

Building docker image locally and start a container to run inference on CPU:
```sh
docker build . --network=host -t chatglm.cpp
# cpp demo
docker run -it --rm -v $PWD/models:/chatglm.cpp/models chatglm.cpp ./build/bin/main -m models/chatglm-ggml.bin -p "你好"
# python demo
docker run -it --rm -v $PWD/models:/chatglm.cpp/models chatglm.cpp python3 examples/cli_demo.py -m models/chatglm-ggml.bin -p "你好"
# langchain api server
docker run -it --rm -v $PWD/models:/chatglm.cpp/models -p 8000:8000 -e MODEL=models/chatglm-ggml.bin chatglm.cpp \
    uvicorn chatglm_cpp.langchain_api:app --host 0.0.0.0 --port 8000
# openai api server
docker run -it --rm -v $PWD/models:/chatglm.cpp/models -p 8000:8000 -e MODEL=models/chatglm-ggml.bin chatglm.cpp \
    uvicorn chatglm_cpp.openai_api:app --host 0.0.0.0 --port 8000
```

For CUDA support, make sure [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is installed. Then run:
```sh
docker build . --network=host -t chatglm.cpp-cuda \
    --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu20.04 \
    --build-arg CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80"
docker run -it --rm --gpus all -v $PWD/models:/chatglm.cpp/models chatglm.cpp-cuda \
    ./build/bin/main -m models/chatglm-ggml.bin -p "你好"
```

**Option 2: Using Pre-built Image**

The pre-built image for CPU inference is published on both [Docker Hub](https://hub.docker.com/repository/docker/liplusx/chatglm.cpp) and [GitHub Container Registry (GHCR)](https://github.com/li-plus/chatglm.cpp/pkgs/container/chatglm.cpp).

To pull from Docker Hub and run demo:
```sh
docker run -it --rm -v $PWD/models:/chatglm.cpp/models liplusx/chatglm.cpp:main \
    ./build/bin/main -m models/chatglm-ggml.bin -p "你好"
```

To pull from GHCR and run demo:
```sh
docker run -it --rm -v $PWD/models:/chatglm.cpp/models ghcr.io/li-plus/chatglm.cpp:main \
    ./build/bin/main -m models/chatglm-ggml.bin -p "你好"
```

Python demo and API servers are also supported in pre-built image. Use it in the same way as **Option 1**.

## Performance

Environment:
* CPU backend performance is measured on a Linux server with Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz using 16 threads.
* CUDA backend is measured on a V100-SXM2-32GB GPU using 1 thread.
* MPS backend is measured on an Apple M2 Ultra device using 1 thread.

ChatGLM-6B:

|                                | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0  | F16   |
|--------------------------------|-------|-------|-------|-------|-------|-------|
| ms/token (CPU @ Platinum 8260) | 74    | 77    | 86    | 89    | 114   | 189   |
| ms/token (CUDA @ V100 SXM2)    | 8.1   | 8.7   | 9.4   | 9.5   | 12.0  | 19.1  |
| ms/token (MPS @ M2 Ultra)      | 11.5  | 12.3  | N/A   | N/A   | 16.1  | 24.4  |
| file size                      | 3.3G  | 3.7G  | 4.0G  | 4.4G  | 6.2G  | 12G   |
| mem usage                      | 4.0G  | 4.4G  | 4.7G  | 5.1G  | 6.9G  | 13G   |

ChatGLM2-6B / ChatGLM3-6B / CodeGeeX2:

|                                | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0  | F16   |
|--------------------------------|-------|-------|-------|-------|-------|-------|
| ms/token (CPU @ Platinum 8260) | 64    | 71    | 79    | 83    | 106   | 189   |
| ms/token (CUDA @ V100 SXM2)    | 7.9   | 8.3   | 9.2   | 9.2   | 11.7  | 18.5  |
| ms/token (MPS @ M2 Ultra)      | 10.0  | 10.8  | N/A   | N/A   | 14.5  | 22.2  |
| file size                      | 3.3G  | 3.7G  | 4.0G  | 4.4G  | 6.2G  | 12G   |
| mem usage                      | 3.4G  | 3.8G  | 4.1G  | 4.5G  | 6.2G  | 12G   |

ChatGLM4-9B:

|                                | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | F16  |
|--------------------------------|------|------|------|------|------|------|
| ms/token (CPU @ Platinum 8260) | 105  | 105  | 122  | 134  | 158  | 279  |
| ms/token (CUDA @ V100 SXM2)    | 12.1 | 12.5 | 13.8 | 13.9 | 17.7 | 27.7 |
| ms/token (MPS @ M2 Ultra)      | 14.4 | 15.3 | 19.6 | 20.1 | 20.7 | 32.4 |
| file size                      | 5.0G | 5.5G | 6.1G | 6.6G | 9.4G | 18G  |

## Model Quality

We measure model quality by evaluating the perplexity over the WikiText-2 test dataset, following the strided sliding window strategy in https://huggingface.co/docs/transformers/perplexity. Lower perplexity usually indicates a better model.

Download and unzip the dataset from [link](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip). Measure the perplexity with a stride of 512 and max input length of 2048:
```sh
./build/bin/perplexity -m models/chatglm3-base-ggml.bin -f wikitext-2-raw/wiki.test.raw -s 512 -l 2048
```

|                         | Q4_0  | Q4_1  | Q5_0  | Q5_1  | Q8_0  | F16   |
|-------------------------|-------|-------|-------|-------|-------|-------|
| [ChatGLM3-6B-Base][1]   | 6.215 | 6.188 | 6.006 | 6.022 | 5.971 | 5.972 |
| [ChatGLM4-9B-Base][2]   | 6.834 | 6.780 | 6.645 | 6.624 | 6.576 | 6.577 |

[1]: https://huggingface.co/THUDM/chatglm3-6b-base
[2]: https://huggingface.co/THUDM/glm-4-9b

## Development

**Unit Test & Benchmark**

To perform unit tests, add this CMake flag `-DCHATGLM_ENABLE_TESTING=ON` to enable testing. Recompile and run the unit test (including benchmark).
```sh
mkdir -p build && cd build
cmake .. -DCHATGLM_ENABLE_TESTING=ON && make -j
./bin/chatglm_test
```

For benchmark only:
```sh
./bin/chatglm_test --gtest_filter='Benchmark.*'
```

**Lint**

To format the code, run `make lint` inside the `build` folder. You should have `clang-format`, `black` and `isort` pre-installed.

**Performance**

To detect the performance bottleneck, add the CMake flag `-DGGML_PERF=ON`:
```sh
cmake .. -DGGML_PERF=ON && make -j
```
This will print timing for each graph operation when running the model.

## Acknowledgements

* This project is greatly inspired by [@ggerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp) and is based on his NN library [ggml](https://github.com/ggerganov/ggml).
* Thank [@THUDM](https://github.com/THUDM) for the amazing [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B), [ChatGLM3](https://github.com/THUDM/ChatGLM3) and [GLM-4](https://github.com/THUDM/GLM-4) and for releasing the model sources and checkpoints.
