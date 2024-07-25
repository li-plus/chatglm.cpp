#include "chatglm.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

struct Args {
    std::string model_path = "models/chatglm-ggml.bin";
    std::string corpus_path = "data/wikitext-2-raw/wiki.test.raw";
    int max_length = 1024;
    int stride = 512;
};

static void usage(const std::string &prog) {
    std::cout << "Usage: " << prog << R"( [options]

options:
  -h, --help            show this help message and exit
  -m, --model PATH      model path
  -f, --file            path to the corpus
  -l, --max_length N    max total length including prompt and output
  -s, --stride N        stride size of the sliding window
)";
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
        } else if (arg == "-f" || arg == "--file") {
            args.corpus_path = argv.at(++i);
        } else if (arg == "-l" || arg == "--max_length") {
            args.max_length = std::stoi(argv.at(++i));
        } else if (arg == "-s" || arg == "--stride") {
            args.stride = std::stoi(argv.at(++i));
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv.at(0));
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char **argv) {
    std::vector<std::string> argv_vec(argv, argv + argc);
    return parse_args(argv_vec);
}

static std::string read_text(const std::string &path) {
    std::ifstream fin(path);
    CHATGLM_CHECK(fin) << "cannot open file " << path;
    std::ostringstream oss;
    oss << fin.rdbuf();
    return oss.str();
}

static float cross_entropy(const ggml_tensor *input, const ggml_tensor *target) {
    CHATGLM_CHECK(ggml_is_contiguous(input) && ggml_n_dims(input) == 2 && input->type == GGML_TYPE_F32);
    CHATGLM_CHECK(ggml_is_contiguous(target) && ggml_n_dims(target) == 1 && target->type == GGML_TYPE_I32);
    CHATGLM_CHECK(input->ne[1] == target->ne[0]);

    const int num_classes = input->ne[0];
    const int batch_size = input->ne[1];

    float loss = 0.f;
#pragma omp parallel for reduction(+ : loss)
    for (int i = 0; i < batch_size; i++) {
        const int target_i = ((const int *)target->data)[i];
        const float *row = (const float *)input->data + i * input->ne[0];
        const float max_val = *std::max_element(row, row + num_classes);
        float sum = 0.f;
        for (int j = 0; j < num_classes; j++) {
            sum += std::exp(row[j] - max_val);
        }
        loss += -(row[target_i] - max_val - std::log(sum));
    }

    return loss / batch_size;
}

// reference: https://huggingface.co/docs/transformers/perplexity
static void perplexity(Args &args) {
    std::cout << "Loading model from " << args.model_path << " ...\n";
    chatglm::Pipeline pipeline(args.model_path, args.max_length);

    std::cout << "Loading corpus from " << args.corpus_path << " ...\n";
    std::string corpus = read_text(args.corpus_path);

    std::cout << "Tokenizing corpus of " << corpus.size() << " bytes ...\n";
    std::vector<int> corpus_ids = pipeline.tokenizer->encode(corpus, std::numeric_limits<int>::max());
    corpus_ids.erase(corpus_ids.begin(), corpus_ids.begin() + 2);

    std::cout << "Computing perplexity against " << corpus_ids.size() << " tokens ...\n";

    float total_loss = 0.f;
    size_t num_samples = 0;

    std::vector<chatglm::no_init<char>> buf;

    size_t prev_end = 0;
    for (size_t begin = 0; begin < corpus_ids.size(); begin += args.stride) {
        const auto clk_start = std::chrono::system_clock::now();
        size_t end = std::min(begin + args.max_length, corpus_ids.size());
        size_t target_len = std::min(end - prev_end, size_t(args.max_length - 1));
        std::vector<int> input_ids(corpus_ids.begin() + begin, corpus_ids.begin() + end);

        ggml_tensor *lm_logits = pipeline.model->forward_graph_compute(input_ids, std::nullopt, 0, 0, false);

        const auto clk_fwd = std::chrono::system_clock::now();

        buf.resize(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_nbytes(lm_logits) +
                   target_len * ggml_type_size(GGML_TYPE_I32));

        auto ctx = chatglm::make_unique_ggml_context(buf.size(), buf.data(), false);
        ggml_tensor *lm_logits_cpu = ggml_new_tensor(ctx.get(), lm_logits->type, ggml_n_dims(lm_logits), lm_logits->ne);
        ggml_backend_tensor_get(lm_logits, lm_logits_cpu->data, 0, ggml_nbytes(lm_logits));
        ggml_tensor *next_lm_logits =
            ggml_view_2d(ctx.get(), lm_logits_cpu, lm_logits_cpu->ne[0], target_len, lm_logits_cpu->nb[1],
                         (input_ids.size() - target_len - 1) * lm_logits_cpu->nb[1]);

        ggml_tensor *next_input_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, target_len);
        memcpy(next_input_ids->data, input_ids.data() + input_ids.size() - target_len, target_len * sizeof(int));

        const float loss = cross_entropy(next_lm_logits, next_input_ids);

        total_loss += loss * target_len;
        num_samples += target_len;

        const auto clk_end = std::chrono::system_clock::now();

        const auto elapsed_fwd = std::chrono::duration_cast<std::chrono::milliseconds>(clk_fwd - clk_start).count();
        const auto elapsed_ce = std::chrono::duration_cast<std::chrono::milliseconds>(clk_end - clk_fwd).count();

        const int progress = end * 100 / corpus_ids.size();
        std::cout << "[" << progress << "%] chunk [" << end - target_len << ", " << end
                  << ") perplexity: " << std::fixed << std::setprecision(3) << std::exp(loss)
                  << ", forward time: " << elapsed_fwd << " ms, cross entropy time: " << elapsed_ce << " ms\n";

        prev_end = end;
        if (end == corpus_ids.size()) {
            break;
        }
    }

    const float ppl = std::exp(total_loss / num_samples);
    std::cout << "Final perplexity: " << std::fixed << std::setprecision(3) << ppl << "\n";
}

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        perplexity(args);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
