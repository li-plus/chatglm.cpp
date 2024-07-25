#include "chatglm.h"
#include <fstream>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

enum InferenceMode {
    INFERENCE_MODE_CHAT,
    INFERENCE_MODE_GENERATE,
};

static inline InferenceMode to_inference_mode(const std::string &s) {
    static std::unordered_map<std::string, InferenceMode> m{{"chat", INFERENCE_MODE_CHAT},
                                                            {"generate", INFERENCE_MODE_GENERATE}};
    return m.at(s);
}

struct Args {
    std::string model_path = "models/chatglm-ggml.bin";
    InferenceMode mode = INFERENCE_MODE_CHAT;
    bool sync = false;
    std::string prompt = "你好";
    std::string system = "";
    std::string image_path = "";
    int max_length = 2048;
    int max_new_tokens = -1;
    int max_context_length = 512;
    bool interactive = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.0;
    bool verbose = false;
};

static void usage(const std::string &prog) {
    std::cout << "Usage: " << prog << R"( [options]

options:
  -h, --help            show this help message and exit
  -m, --model PATH      model path (default: models/chatglm-ggml.bin)
  --mode                inference mode chosen from {chat, generate} (default: chat)
  --sync                synchronized generation without streaming
  -p, --prompt PROMPT   prompt to start generation with (default: 你好)
  --pp, --prompt_path   path to the plain text file that stores the prompt
  -s, --system SYSTEM   system message to set the behavior of the assistant
  --sp, --system_path   path to the plain text file that stores the system message
  --image               path to the input image for visual language models
  -i, --interactive     run in interactive mode
  -l, --max_length N    max total length including prompt and output (default: 2048)
  --max_new_tokens N    max number of tokens to generate, ignoring the number of prompt tokens
  -c, --max_context_length N
                        max context length (default: 512)
  --top_k N             top-k sampling (default: 0)
  --top_p N             top-p sampling (default: 0.7)
  --temp N              temperature (default: 0.95)
  --repeat_penalty N    penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
  -v, --verbose         display verbose output including config/system/performance info
)";
}

static std::string read_text(std::string path) {
    std::ifstream fin(path);
    CHATGLM_CHECK(fin) << "cannot open file " << path;
    std::ostringstream oss;
    oss << fin.rdbuf();
    return oss.str();
}

static Args parse_args(const std::vector<std::string> &argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string &arg = argv.at(i);

        if (arg == "-h" || arg == "--help") {
            usage(argv.at(0));
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            args.model_path = argv.at(++i);
        } else if (arg == "--mode") {
            args.mode = to_inference_mode(argv.at(++i));
        } else if (arg == "--sync") {
            args.sync = true;
        } else if (arg == "-p" || arg == "--prompt") {
            args.prompt = argv.at(++i);
        } else if (arg == "--pp" || arg == "--prompt_path") {
            args.prompt = read_text(argv.at(++i));
        } else if (arg == "-s" || arg == "--system") {
            args.system = argv.at(++i);
        } else if (arg == "--sp" || arg == "--system_path") {
            args.system = read_text(argv.at(++i));
        } else if (arg == "--image") {
            args.image_path = argv.at(++i);
        } else if (arg == "-i" || arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "-l" || arg == "--max_length") {
            args.max_length = std::stoi(argv.at(++i));
        } else if (arg == "--max_new_tokens") {
            args.max_new_tokens = std::stoi(argv.at(++i));
        } else if (arg == "-c" || arg == "--max_context_length") {
            args.max_context_length = std::stoi(argv.at(++i));
        } else if (arg == "--top_k") {
            args.top_k = std::stoi(argv.at(++i));
        } else if (arg == "--top_p") {
            args.top_p = std::stof(argv.at(++i));
        } else if (arg == "--temp") {
            args.temp = std::stof(argv.at(++i));
        } else if (arg == "--repeat_penalty") {
            args.repeat_penalty = std::stof(argv.at(++i));
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv.at(0));
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char **argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR *wargs = CommandLineToArgvW(GetCommandLineW(), &argc);
    CHATGLM_CHECK(wargs) << "failed to retrieve command line arguments";

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

static bool get_utf8_line(std::string &line) {
#ifdef _WIN32
    std::wstring wline;
    bool ret = !!std::getline(std::wcin, wline);
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    line = converter.to_bytes(wline);
    return ret;
#else
    return !!std::getline(std::cin, line);
#endif
}

static inline void print_message(const chatglm::ChatMessage &message) {
    std::cout << message.content << "\n";
    if (!message.tool_calls.empty() && message.tool_calls.front().type == chatglm::ToolCallMessage::TYPE_CODE) {
        std::cout << message.tool_calls.front().code.input << "\n";
    }
}

static void chat(Args &args) {
    ggml_time_init();
    int64_t start_load_us = ggml_time_us();
    chatglm::Pipeline pipeline(args.model_path, args.max_length);
    int64_t end_load_us = ggml_time_us();

    std::string model_name = pipeline.model->config.model_type_name();

    auto text_streamer = std::make_shared<chatglm::TextStreamer>(std::cout, pipeline.tokenizer.get());
    auto perf_streamer = std::make_shared<chatglm::PerfStreamer>();
    std::vector<std::shared_ptr<chatglm::BaseStreamer>> streamers{perf_streamer};
    if (!args.sync) {
        streamers.emplace_back(text_streamer);
    }
    auto streamer = std::make_unique<chatglm::StreamerGroup>(std::move(streamers));

    chatglm::GenerationConfig gen_config(args.max_length, args.max_new_tokens, args.max_context_length, args.temp > 0,
                                         args.top_k, args.top_p, args.temp, args.repeat_penalty);

    if (args.verbose) {
        std::cout << "system info: | "
                  << "AVX = " << ggml_cpu_has_avx() << " | "
                  << "AVX2 = " << ggml_cpu_has_avx2() << " | "
                  << "AVX512 = " << ggml_cpu_has_avx512() << " | "
                  << "AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << " | "
                  << "AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << " | "
                  << "FMA = " << ggml_cpu_has_fma() << " | "
                  << "NEON = " << ggml_cpu_has_neon() << " | "
                  << "ARM_FMA = " << ggml_cpu_has_arm_fma() << " | "
                  << "F16C = " << ggml_cpu_has_f16c() << " | "
                  << "FP16_VA = " << ggml_cpu_has_fp16_va() << " | "
                  << "WASM_SIMD = " << ggml_cpu_has_wasm_simd() << " | "
                  << "BLAS = " << ggml_cpu_has_blas() << " | "
                  << "SSE3 = " << ggml_cpu_has_sse3() << " | "
                  << "VSX = " << ggml_cpu_has_vsx() << " |\n";

        std::cout << "inference config: | "
                  << "max_length = " << args.max_length << " | "
                  << "max_context_length = " << args.max_context_length << " | "
                  << "top_k = " << args.top_k << " | "
                  << "top_p = " << args.top_p << " | "
                  << "temperature = " << args.temp << " | "
                  << "repetition_penalty = " << args.repeat_penalty << " |\n";

        std::cout << "loaded " << pipeline.model->config.model_type_name() << " model from " << args.model_path
                  << " within: " << (end_load_us - start_load_us) / 1000.f << " ms\n";

        std::cout << std::endl;
    }

    if (args.mode != INFERENCE_MODE_CHAT && args.interactive) {
        std::cerr << "interactive demo is only supported for chat mode, falling back to non-interactive one\n";
        args.interactive = false;
    }

    if (!args.image_path.empty() && pipeline.model->config.model_type != chatglm::ModelType::CHATGLM4V) {
        std::cerr << "image is specified for model without visual ability, falling back to language mode\n";
        args.image_path.clear();
    }

    std::optional<chatglm::Image> image;
    if (!args.image_path.empty()) {
        image = chatglm::Image::open(args.image_path);
    }

    std::vector<chatglm::ChatMessage> system_messages;
    if (!args.system.empty()) {
        system_messages.emplace_back(chatglm::ChatMessage::ROLE_SYSTEM, args.system);
    }

    if (args.interactive) {
        std::cout << R"(    ________          __  ________    __  ___                 )" << '\n'
                  << R"(   / ____/ /_  ____ _/ /_/ ____/ /   /  |/  /_________  ____  )" << '\n'
                  << R"(  / /   / __ \/ __ `/ __/ / __/ /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                  << R"( / /___/ / / / /_/ / /_/ /_/ / /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                  << R"( \____/_/ /_/\__,_/\__/\____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n'
                  << R"(                                              /_/   /_/       )" << '\n'
                  << '\n';

        std::cout
            << "Welcome to ChatGLM.cpp! Ask whatever you want. Type 'clear' to clear context. Type 'stop' to exit.\n"
            << "\n";

        std::vector<chatglm::ChatMessage> messages = system_messages;
        if (!args.system.empty()) {
            std::cout << std::setw(model_name.size()) << std::left << "System"
                      << " > " << args.system << std::endl;
        }
        auto prompt_image = image;
        while (true) {
            std::string role;
            if (!messages.empty() && !messages.back().tool_calls.empty()) {
                const auto &tool_call = messages.back().tool_calls.front();
                if (tool_call.type == chatglm::ToolCallMessage::TYPE_FUNCTION) {
                    // function call
                    std::cout << "Function Call > Please manually call function `" << tool_call.function.name
                              << "` with args `" << tool_call.function.arguments << "` and provide the results below.\n"
                              << "Observation   > " << std::flush;
                } else if (tool_call.type == chatglm::ToolCallMessage::TYPE_CODE) {
                    // code interpreter
                    std::cout << "Code Interpreter > Please manually run the code and provide the results below.\n"
                              << "Observation      > " << std::flush;
                } else {
                    CHATGLM_THROW << "unexpected tool type " << tool_call.type;
                }
                role = chatglm::ChatMessage::ROLE_OBSERVATION;
            } else {
                std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                          << " > " << std::flush;
                role = chatglm::ChatMessage::ROLE_USER;
            }
            std::string prompt;
            if (!get_utf8_line(prompt) || prompt == "stop") {
                break;
            }
            if (prompt.empty()) {
                continue;
            }
            if (prompt == "clear") {
                messages = system_messages;
                prompt_image = image;
                continue;
            }
            messages.emplace_back(std::move(role), std::move(prompt), std::move(prompt_image));
            prompt_image.reset();
            std::cout << model_name << " > ";
            chatglm::ChatMessage output = pipeline.chat(messages, gen_config, streamer.get());
            if (args.sync) {
                print_message(output);
            }
            messages.emplace_back(std::move(output));
            if (args.verbose) {
                std::cout << "\n" << perf_streamer->to_string() << "\n\n";
            }
            perf_streamer->reset();
        }
        std::cout << "Bye\n";
    } else {
        if (args.mode == INFERENCE_MODE_CHAT) {
            std::vector<chatglm::ChatMessage> messages = system_messages;
            messages.emplace_back(chatglm::ChatMessage::ROLE_USER, args.prompt, std::move(image));
            chatglm::ChatMessage output = pipeline.chat(messages, gen_config, streamer.get());
            if (args.sync) {
                print_message(output);
            }
        } else {
            std::string output = pipeline.generate(args.prompt, gen_config, streamer.get());
            if (args.sync) {
                std::cout << output << "\n";
            }
        }
        if (args.verbose) {
            std::cout << "\n" << perf_streamer->to_string() << "\n\n";
        }
    }
}

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    try {
        Args args = parse_args(argc, argv);
        chat(args);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
