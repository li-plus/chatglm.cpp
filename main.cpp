#include "chatglm.h"
#include <iomanip>
#include <iostream>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

struct Args {
    std::string model_path = "chatglm-ggml.bin";
    std::string prompt = "你好";
    int max_length = 2048;
    int max_context_length = 512;
    bool interactive = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    int num_threads = 0;
    bool verbose = false;
};

void usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help              show this help message and exit\n"
              << "  -m, --model PATH        model path (default: chatglm-ggml.bin)\n"
              << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
              << "  -i, --interactive       run in interactive mode\n"
              << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
              << "  -c, --max_context_length N\n"
              << "                          max context length (default: 512)\n"
              << "  --top_k N               top-k sampling (default: 0)\n"
              << "  --top_p N               top-p sampling (default: 0.7)\n"
              << "  --temp N                temperature (default: 0.95)\n"
              << "  -t, --threads N         number of threads for inference\n"
              << "  -v, --verbose           display verbose output including config/system/performance info\n";
}

static Args parse_args(int argc, char **argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (arg == "-m" || arg == "--model") {
            args.model_path = argv[++i];
        } else if (arg == "-p" || arg == "--prompt") {
            args.prompt = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "-l" || arg == "--max_length") {
            args.max_length = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--max_context_length") {
            args.max_context_length = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top_p") {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            args.temp = std::stof(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

#if defined(_WIN32)
static void append_utf8(char32_t ch, std::string &out) {
    if (ch <= 0x7F) {
        out.push_back(static_cast<unsigned char>(ch));
    } else if (ch <= 0x7FF) {
        out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0xFFFF) {
        out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0x10FFFF) {
        out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else {
        // Invalid Unicode code point
    }
}

static bool get_utf8_line(std::string &line) {
    std::wstring prompt;
    std::wcin >> prompt;
    for (auto wc : prompt)
        append_utf8(wc, line);
    return true;
}
#else
static bool get_utf8_line(std::string &line) { return !!std::getline(std::cin, line); }
#endif

void chat(const Args &args) {
    ggml_time_init();
    int64_t start_load_us = ggml_time_us();
    chatglm::Pipeline pipeline(args.model_path);
    int64_t end_load_us = ggml_time_us();

    std::string model_name = pipeline.model->type_name();

    auto text_streamer = std::make_shared<chatglm::TextStreamer>(std::cout, pipeline.tokenizer.get());
    auto perf_streamer = std::make_shared<chatglm::PerfStreamer>();
    auto streamer = std::make_shared<chatglm::StreamerGroup>(
        std::vector<std::shared_ptr<chatglm::BaseStreamer>>{text_streamer, perf_streamer});

    chatglm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                         args.top_p, args.temp, args.num_threads);

#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

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
                  << "num_threads = " << args.num_threads << " |\n";

        std::cout << "loaded " << pipeline.model->type_name() << " model from " << args.model_path
                  << " within: " << (end_load_us - start_load_us) / 1000.f << " ms\n";

        std::cout << std::endl;
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

        std::vector<std::string> history;
        while (1) {
            std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                      << " > " << std::flush;
            std::string prompt;
            if (!get_utf8_line(prompt) || prompt == "stop") {
                break;
            }
            if (prompt.empty()) {
                continue;
            }
            if (prompt == "clear") {
                history.clear();
                continue;
            }
            history.emplace_back(std::move(prompt));
            std::cout << model_name << " > ";
            std::string output = pipeline.chat(history, gen_config, streamer.get());
            history.emplace_back(std::move(output));
            if (args.verbose) {
                std::cout << "\n" << perf_streamer->to_string() << "\n\n";
            }
            perf_streamer->reset();
        }
        std::cout << "Bye\n";
    } else {
        pipeline.chat({args.prompt}, gen_config, streamer.get());
        if (args.verbose) {
            std::cout << "\n" << perf_streamer->to_string() << "\n\n";
        }
    }
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    try {
        chat(args);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
