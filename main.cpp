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
              << "  -t, --threads N         number of threads for inference\n";
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
    chatglm::Pipeline pipeline(args.model_path);
    std::string model_name = pipeline.model->type_name();

    chatglm::TextStreamer streamer(pipeline.tokenizer.get());
    chatglm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                         args.top_p, args.temp, args.num_threads);

#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    if (args.interactive) {
        std::cout << R"(    ________          __  ________    __  ___                 )" << '\n'
                  << R"(   / ____/ /_  ____ _/ /_/ ____/ /   /  |/  /_________  ____  )" << '\n'
                  << R"(  / /   / __ \/ __ `/ __/ / __/ /   / /|_/ // ___/ __ \/ __ \ )" << '\n'
                  << R"( / /___/ / / / /_/ / /_/ /_/ / /___/ /  / // /__/ /_/ / /_/ / )" << '\n'
                  << R"( \____/_/ /_/\__,_/\__/\____/_____/_/  /_(_)___/ .___/ .___/  )" << '\n'
                  << R"(                                              /_/   /_/       )" << '\n';

        std::vector<std::string> history;
        while (1) {
            std::cout << std::setw(model_name.size()) << std::left << "Prompt"
                      << " > " << std::flush;
            std::string prompt;
            if (!get_utf8_line(prompt)) {
                break;
            }
            if (prompt.empty()) {
                continue;
            }
            history.emplace_back(std::move(prompt));
            std::cout << model_name << " > ";
            std::string output = pipeline.chat(history, gen_config, &streamer);
            history.emplace_back(std::move(output));
        }
        std::cout << "Bye\n";
    } else {
        pipeline.chat({args.prompt}, gen_config, &streamer);
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
