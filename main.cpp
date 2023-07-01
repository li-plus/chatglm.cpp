#include "chatglm.h"
#include <getopt.h>
#include <iomanip>
#include <iostream>

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

void usage(char *prog) {
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
    // reference: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
    Args args;

    struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                    {"model", required_argument, 0, 'm'},
                                    {"prompt", required_argument, 0, 'p'},
                                    {"interactive", no_argument, 0, 'i'},
                                    {"max_length", required_argument, 0, 'l'},
                                    {"max_context_length", required_argument, 0, 'c'},
                                    {"top_k", required_argument, 0, '0'},
                                    {"top_p", required_argument, 0, '1'},
                                    {"temp", required_argument, 0, '2'},
                                    {"threads", required_argument, 0, 't'},
                                    {0, 0, 0, 0}};

    int c;
    while ((c = getopt_long(argc, argv, "hm:p:il:c:0:1:2:t:", long_options, nullptr)) != -1) {
        switch (c) {
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        case 'm':
            args.model_path = optarg;
            break;
        case 'p':
            args.prompt = optarg;
            break;
        case 'i':
            args.interactive = true;
            break;
        case 'l':
            args.max_length = std::stoi(optarg);
            break;
        case 'c':
            args.max_context_length = std::stoi(optarg);
            break;
        case '0':
            args.top_k = std::stoi(optarg);
            break;
        case '1':
            args.top_p = std::stof(optarg);
            break;
        case '2':
            args.temp = std::stof(optarg);
            break;
        case 't':
            args.num_threads = std::stoi(optarg);
            break;
        case '?':
            usage(argv[0]);
            exit(EXIT_FAILURE);
        default:
            abort();
        }
    }

    if (optind < argc) {
        std::cerr << "Unknown arguments:";
        for (int i = optind; i < argc; i++) {
            std::cerr << " " << argv[i];
        }
        std::cerr << std::endl;
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    return args;
}

void chat(const Args &args) {
    chatglm::Pipeline pipeline(args.model_path);
    std::string model_name = pipeline.model->type_name();

    chatglm::TextStreamer streamer(pipeline.tokenizer.get());
    chatglm::GenerationConfig gen_config(args.max_length, args.max_context_length, args.temp > 0, args.top_k,
                                         args.top_p, args.temp, args.num_threads);

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
            if (!std::getline(std::cin, prompt)) {
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
