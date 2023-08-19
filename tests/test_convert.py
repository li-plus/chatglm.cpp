import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from chatglm_cpp.convert import quantize_q4_0, quantize_q4_1, quantize_q5_0, quantize_q5_1, quantize_q8_0

HERE = Path(__file__).resolve().parent

# generated by:
#   torch.manual_seed(0)
#   weight = torch.randn(2, 128)

# fmt: off
weight = torch.tensor([[
    -1.1258e+00, -1.1524e+00, -2.5058e-01, -4.3388e-01,  8.4871e-01,  6.9201e-01, -3.1601e-01, -2.1152e+00,
     3.2227e-01, -1.2633e+00,  3.4998e-01,  3.0813e-01,  1.1984e-01,  1.2377e+00,  1.1168e+00, -2.4728e-01,
    -1.3527e+00, -1.6959e+00,  5.6665e-01,  7.9351e-01,  5.9884e-01, -1.5551e+00, -3.4136e-01,  1.8530e+00,
     7.5019e-01, -5.8550e-01, -1.7340e-01,  1.8348e-01,  1.3894e+00,  1.5863e+00,  9.4630e-01, -8.4368e-01,
    -6.1358e-01,  3.1593e-02, -4.9268e-01,  2.4841e-01,  4.3970e-01,  1.1241e-01,  6.4079e-01,  4.4116e-01,
    -1.0231e-01,  7.9244e-01, -2.8967e-01,  5.2507e-02,  5.2286e-01,  2.3022e+00, -1.4689e+00, -1.5867e+00,
    -6.7309e-01,  8.7283e-01,  1.0554e+00,  1.7784e-01, -2.3034e-01, -3.9175e-01,  5.4329e-01, -3.9516e-01,
    -4.4622e-01,  7.4402e-01,  1.5210e+00,  3.4105e+00, -1.5312e+00, -1.2341e+00,  1.8197e+00, -5.5153e-01,
    -5.6925e-01,  9.1997e-01,  1.1108e+00,  1.2899e+00, -1.4782e+00,  2.5672e+00, -4.7312e-01,  3.3555e-01,
    -1.6293e+00, -5.4974e-01, -4.7983e-01, -4.9968e-01, -1.0670e+00,  1.1149e+00, -1.4067e-01,  8.0575e-01,
    -9.3348e-02,  6.8705e-01, -8.3832e-01,  8.9182e-04,  8.4189e-01, -4.0003e-01,  1.0395e+00,  3.5815e-01,
    -2.4600e-01,  2.3025e+00, -1.8817e+00, -4.9727e-02, -1.0450e+00, -9.5650e-01,  3.3532e-02,  7.1009e-01,
     1.6459e+00, -1.3602e+00,  3.4457e-01,  5.1987e-01, -2.6133e+00, -1.6965e+00, -2.2824e-01,  2.7995e-01,
     2.4693e-01,  7.6887e-02,  3.3801e-01,  4.5440e-01,  4.5694e-01, -8.6537e-01,  7.8131e-01, -9.2679e-01,
    -2.1883e-01, -2.4351e+00, -7.2915e-02, -3.3986e-02,  9.6252e-01,  3.4917e-01, -9.2146e-01, -5.6195e-02,
    -6.2270e-01, -4.6372e-01,  1.9218e+00, -4.0255e-01,  1.2390e-01,  1.1648e+00,  9.2337e-01,  1.3873e+00],
   [-8.8338e-01, -4.1891e-01, -8.0483e-01,  5.6561e-01,  6.1036e-01,  4.6688e-01,  1.9507e+00, -1.0631e+00,
    -7.7326e-02,  1.1640e-01, -5.9399e-01, -1.2439e+00, -1.0209e-01, -1.0335e+00, -3.1264e-01,  2.4579e-01,
    -2.5964e-01,  1.1834e-01,  2.4396e-01,  1.1646e+00,  2.8858e-01,  3.8660e-01, -2.0106e-01, -1.1793e-01,
     1.9220e-01, -7.7216e-01, -1.9003e+00,  1.3068e-01, -7.0429e-01,  3.1472e-01,  1.5739e-01,  3.8536e-01,
     9.6715e-01, -9.9108e-01,  3.0161e-01, -1.0732e-01,  9.9846e-01, -4.9871e-01,  7.6111e-01,  6.1830e-01,
     3.1405e-01,  2.1333e-01, -1.2005e-01,  3.6046e-01, -3.1403e-01, -1.0787e+00,  2.4081e-01, -1.3962e+00,
    -6.6144e-02, -3.5836e-01, -1.5616e+00, -3.5464e-01,  1.0811e+00,  1.3148e-01,  1.5735e+00,  7.8143e-01,
    -1.0787e+00, -7.2091e-01,  1.4708e+00,  2.7564e-01,  6.6678e-01, -9.9439e-01, -1.1894e+00, -1.1959e+00,
    -5.5963e-01,  5.3347e-01,  4.0689e-01,  3.9459e-01,  1.7151e-01,  8.7604e-01, -2.8709e-01,  1.0216e+00,
    -7.4395e-02, -1.0922e+00,  3.9203e-01,  5.9453e-01,  6.6227e-01, -1.2063e+00,  6.0744e-01, -5.4716e-01,
     1.1711e+00,  9.7496e-02,  9.6337e-01,  8.4032e-01, -1.2537e+00,  9.8684e-01, -4.9466e-01, -1.2830e+00,
     9.5522e-01,  1.2836e+00, -6.6586e-01,  5.6513e-01,  2.8770e-01, -3.3375e-02, -1.0619e+00, -1.1443e-01,
    -3.4334e-01,  1.5713e+00,  1.9161e-01,  3.7994e-01, -1.4476e-01,  6.3762e-01, -2.8129e-01, -1.3299e+00,
    -1.4201e-01, -5.3415e-01, -5.2338e-01,  8.6150e-01, -8.8696e-01,  8.3877e-01,  1.1529e+00, -1.7611e+00,
    -1.4777e+00, -1.7557e+00,  7.6166e-02, -1.0786e+00,  1.4403e+00, -1.1059e-01,  5.7686e-01, -1.6917e-01,
    -6.4025e-02,  1.0384e+00,  9.0682e-01, -4.7551e-01, -8.7074e-01,  1.4474e-01,  1.9029e+00,  3.9040e-01]])
# fmt: on


def test_quantize_q8_0():
    q_tensor = quantize_q8_0(weight).int()
    # fmt: off
    ggml_q_tensor = torch.tensor([
          68,   36,  -68,  -69,  -15,  -26,   51,   42,  -19, -127,   19,  -76,
          21,   19,    7,   74,   67,  -15,  -81, -102,   34,   48,   36,  -93,
         -20,  111,   45,  -35,  -10,   11,   83,   95,   57,  -51,  -32,   38,
         -23,    1,  -18,    9,   16,    4,   24,   16,   -4,   30,  -11,    2,
          19,   86,  -55,  -59,  -25,   33,   39,    7,   -9,  -15,   20,  -15,
         -17,   28,   57,  127,  -57,  -46,   68,  -21,   45,   37,  -28,   46,
          55,   64,  -73,  127,  -23,   17,  -81,  -27,  -24,  -25,  -53,   55,
          -7,   40,   -5,   34,  -41,    0,   42,  -20,   51,   18,  -12,  114,
         -93,   -2,  -52,  -47,    2,   35,   69,   37,   80,  -66,   17,   25,
        -127,  -82,  -11,   14,   12,    4,   16,   22,   22,  -42,   38,  -45,
         -11, -118,   -4,   -2,   47,   17,  -45,   -3,  -30,  -23,   93,  -20,
           6,   57,   45,   67,  -35,   35,  -58,  -27,  -52,   37,   40,   30,
         127,  -69,   -5,    8,  -39,  -81,   -7,  -67,  -20,   16,  -17,    8,
          16,   76,   19,   25,  -13,   -8,   13,  -50, -124,    9,  -46,   20,
          10,   25,   88,   34,   78,  -80,   24,   -9,   81,  -40,   61,   50,
          25,   17,  -10,   29,  -25,  -87,   19, -113,   -5,  -29, -126,  -29,
          87,   11,  127,   63,  -87,  -58,  119,   22,   54,  -80,  -96,  -97,
          45,   33,  -55,   53,   40,   39,   17,   87,  -28,  101,   -7, -108,
          39,   59,   66, -119,   60,  -54,  116,   10,   95,   83, -124,   98,
         -49, -127,   95,  127,  -66,   56,   28,   -3, -105,  -11,  -84,   35,
         -23,  105,   13,   25,  -10,   43,  -19,  -89,   -9,  -36,  -35,   57,
         -59,   56,   77, -118,  -99, -117,    5,  -72,   96,   -7,   38,  -11,
          -4,   69,   61,  -32,  -58,   10,  127,   26]).view(q_tensor.shape)
    # fmt: on
    assert (q_tensor == ggml_q_tensor).all()


def test_quantize_q4_0():
    q_tensor = quantize_q4_0(weight).int()
    # fmt: off
    ggml_q_tensor = torch.tensor([
          59,   52,   52,   36,  -89,  -74,  -85,   43,  119,  -16,  -71,   99,
         121, -103,  -40,  -19,  -52,   87,  -46,  -74,  -87,  104,  105, -121,
        -105, -104,  118, -105, -104,  102,   73,    8,  -57,  -77,   75, -100,
          34,  -75, -118,  101,  -75, -124,   93, -112,   89,  119,  -99,   26,
         -23, -118,  -69,  -75, -120,  101,   58,   53,  125,   20, -119, -118,
         -80, -109,   87, -119,  105,  120,  -23,  121, -119,  -59,  -70,  -59,
         -50,  -77, -100, -118,  123,   54,  117,  102, -112, -116,  120,  -72,
          -6,  125,  -72,  124,  121,  103,   75,  -78, -125,  -83,  -10,  -87,
          51,  123,    4,   69,  -42,  -57,   25,  118,   90,  -35,  -25,  -17,
          34,  -79,   27,  117,   37,   54,   -9,   35,  -70,  -14,   40,   15,
         -58,   68,  100, -113,  -12, -101,  -99,  -77,  -23,  -15, -121,  -42,
          41, -123,  105,  -98, -119,   74,   74,  -92,  -52,  116,    3,  111]).view(q_tensor.shape)
    # fmt: on
    assert (q_tensor == ggml_q_tensor).all()


def test_quantize_q4_1():
    q_tensor = quantize_q4_1(weight).int()
    # fmt: off
    ggml_q_tensor = torch.tensor([
          60,   52,   59,  -64,   52,   36,  -89,  -74,  -85,   43,  119,  -16,
         -71,   99,  121, -103,  -40,  -19,  -52,   87,   85,   53,   89,  -66,
          51,  117, -125,   86,   70,   69,  103,   70,   52,  119, -108,  -11,
           6,   28,  -96,   48,  -65,   52, -121,  -65,  100, -103,   74,  107,
        -111,   95,  -91, -121,   97,  -28,    5,  101,   51,   58,  102, -103,
         -42,   52,   58,  -63, -114,   20, -118, -102,  -64,  -93,  104, -118,
         121,  121,   -6,  122, -102,  -58,  -53,  -42,   28,   52, -102,  -65,
         100, -122, -124,  -54, -102, -103,  127,  115, -121,   72,    5, -125,
          87, -109, -122, -104,  -80,   50,   63,  -66,  124,   99,    9,  103,
         -36, -123,   -5,  -70,   41,   72,   -9, -103,  -74,   50,   41,   33,
         122,   49,   34,  -67,  -28, -117,  -38,  -54,    9,  -35,   86,   13,
         -41,  -15,   74,  -69, -101,  112,   27,  116,  -47,   51,   11,  -65,
          22,   14, -120,   57,  -41,  122,  -90,  114,  119,  -75,  -75,   91,
          68, -117,   -4, -112]).view(q_tensor.shape)
    # fmt: on
    assert (q_tensor == ggml_q_tensor).all()


def test_quantize_q5_0():
    q_tensor = quantize_q5_0(weight).int()
    # fmt: off
    ggml_q_tensor = torch.tensor([
          59,   48,   48,  125, -100,  121,  103,   55,   78,  109,   86,   69,
         -34,  -32,   98,  -58,  -13,   18,  -79,  -55,  120,  -82,  -46,  -78,
           7,  -51,  -79,  -79,   51,  -64,  -78,   -1,   30,   47,  -35,   46,
          32,  -36, -111,    0,  126,  101,  119,   55,   34,  -79,   81,   95,
          45,  125,   20,  -54,   89,    8,  -71,   32,  -93,  -18,   42,   35,
         -61,    3,  119,  105,    1,  -53,   58,   49, -115,   95,  -68,  -12,
          -6,   24,    2,    3,   96,   38,  -81,    2,  -62,  -48,  -62,  -29,
          19,  123,  101, -118,  -50,  -81, -121,  125,  -63,   22,   39,  -13,
         -25,  107,  -21,  -36,   32,   25,  -31,  111,  -11,   -6,   97,  -40,
         -13,  -34,   75,  -82,   42,  -76,   15,  -29,   22,   74,   -3,   65,
          86,  -11,    8, -118,  -67,  126,   17,  -36, -109,  -85,  -50,  -50,
          34,  -83,   65,  -93,  -48,  -28,   23,   -7,   75,  107,   -2,   69,
         100,  -13,   65,   14, -117, -103,  -56,   15,  -40,   23,  -99,  -81,
         -47, -105,  -85,   25,  -61,  -13,   -2,  -99,   65,   27,  -78,   27,
          17,  116, -124,   73,  119,   -7,    6,  -33]).view(q_tensor.shape)
    # fmt: on
    assert (q_tensor == ggml_q_tensor).all()


def test_quantize_q5_1():
    q_tensor = quantize_q5_1(weight).int()
    # fmt: off
    ggml_q_tensor = torch.tensor([
          25,   48,   59,  -64,   48,  125, -100,  121,  104,   56,   95,  125,
          87,   70,  -18,  -16,   99,  -57,  -13,   35,  -79,  -38, -119,  -81,
          41,   49,   89,  -66,    0,   32,    4,   76,  102,   -6,    7,  -69,
        -115,  123,  -34,  125,  121,  -17,   56,   -6,   13,   40,   81,   96,
        -104,   48, -121,  -65,   46,  -96,  -46, -126,  -55,   36,  117,  -42,
          51,  -81,   74,   15,  -78,  -39,   10,  -38,  102,  101,  -36,   35,
         -82,   48,   58,  -63,  -51,   95,  -67,  -12,   13,   25,   20,   37,
        -128,   70,  -64,   20,  -28,  -14,  -12,  -11,   53,  -84, -121,  -68,
         -13,   47, -102,  -65,  120, -126,   62,  -23,  -40,   12,   25, -108,
          36,   35,  -17,  -25,   31, -112,   11,    5,  -82,   39,   29,   33,
         121,   46,   63,  -66,  -43,   75,  -16,   28,   -7,  -58,    2,  -50,
         -87,   27,   -9,  118,   83, -126,  -18,   35,  108,  101,   66,   66,
          76,   45,   34,  -67,  -66,   92,   47,   27,  -23,   22,  -76,  -92,
           2,  -70,  -84,   12,  -65,  -14,  116,  103,   55,  -15,   55,  -23,
        -112,   47,   11,  -65,   46,  104,   84,  -26,   44,   12,    1,   98,
         -66,  -28,   77,  -44,  -18, -118,  122,  -74, -121,    6,   -7,   32]).view(q_tensor.shape)
    # fmt: on
    assert (q_tensor == ggml_q_tensor).all()


CHATGLM_MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--THUDM--chatglm-6b/snapshots/294cb13118a1e08ad8449ca542624a5c6aecc401"
).expanduser()

CHATGLM2_MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--THUDM--chatglm2-6b/snapshots/0ecfe0b857efd00836a4851b3dd2ed04bd4b197f"
).expanduser()

BAICHUAN13B_MODEL_PATH = Path(
    "~/.cache/huggingface/hub/models--baichuan-inc--Baichuan-13B-Chat/snapshots/a4a558127068f2ce965aa56aeb826bf501a68970"
).expanduser()


def make_data_embedding():
    m = torch.nn.Embedding(4, 3)
    x = torch.tensor([1, 3, 0, 2, 3])
    y = m(x)
    print("w", m.weight.data.flatten())
    print("x", x.flatten())
    print("y", y.flatten())


def make_data_linear():
    w = torch.randn(16, 32)
    b = torch.randn(16)
    x = torch.randn(2, 32)
    y = F.linear(x, w, b)

    with open(HERE / "data/linear.data", "wb") as f:
        w.numpy().tofile(f)
        b.numpy().tofile(f)
        x.numpy().tofile(f)
        y.numpy().tofile(f)


def make_data_layernorm():
    w = torch.randn(64)
    b = torch.randn(64)
    x = torch.randn(3, 64)
    y = F.layer_norm(x, [64], w, b)

    with open(HERE / "data/layer_norm.data", "wb") as f:
        w.numpy().tofile(f)
        b.numpy().tofile(f)
        x.numpy().tofile(f)
        y.numpy().tofile(f)


def make_data_rms_norm():
    sys.path.append(str(CHATGLM2_MODEL_PATH))
    from modeling_chatglm import RMSNorm

    m = RMSNorm(64, eps=1e-5).eval()
    m.weight.data.uniform_()

    x = torch.randn(3, 64)
    with torch.no_grad():
        y = m(x)

    with open(HERE / "data/rms_norm.data", "wb") as f:
        m.weight.data.numpy().tofile(f)
        x.numpy().tofile(f)
        y.numpy().tofile(f)


def make_data_glm_block():
    sys.path.append(str(CHATGLM_MODEL_PATH))
    from modeling_chatglm import GLMBlock

    m = (
        GLMBlock(
            hidden_size=32, num_attention_heads=8, layernorm_epsilon=1e-5, layer_id=3, num_layers=28, empty_init=False
        )
        .float()
        .eval()
    )
    x1 = torch.randn(4, 1, 32)  # [seqlen, bs, hidden]
    position_ids = torch.tensor([[[0, 1, 2, 2], [0, 0, 0, 1]]])
    attention_mask = torch.tensor(
        [
            [
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    y1, layer_past = m(
        x1, position_ids=position_ids, attention_mask=attention_mask, layer_id=m.layer_id, use_cache=True
    )

    # cross attention
    x2 = torch.randn(1, 1, 32)
    position_ids = torch.tensor([[[2], [2]]])
    attention_mask = torch.zeros(1, 1, dtype=torch.bool)
    y2, layer_past = m(
        x2,
        position_ids=position_ids,
        attention_mask=attention_mask,
        layer_id=m.layer_id,
        layer_past=layer_past,
        use_cache=True,
    )

    x3 = torch.randn(1, 1, 32)
    position_ids = torch.tensor([[[2], [3]]])
    attention_mask = torch.zeros(1, 1, dtype=torch.bool)
    y3, layer_past = m(
        x3,
        position_ids=position_ids,
        attention_mask=attention_mask,
        layer_id=m.layer_id,
        layer_past=layer_past,
        use_cache=True,
    )

    print(m)

    with open(HERE / "data/glm_block.data", "wb") as f:
        m.input_layernorm.weight.data.numpy().tofile(f)
        m.input_layernorm.bias.data.numpy().tofile(f)
        m.attention.query_key_value.weight.data.numpy().tofile(f)
        m.attention.query_key_value.bias.data.numpy().tofile(f)
        m.attention.dense.weight.data.numpy().tofile(f)
        m.attention.dense.bias.data.numpy().tofile(f)
        m.post_attention_layernorm.weight.data.numpy().tofile(f)
        m.post_attention_layernorm.bias.data.numpy().tofile(f)
        m.mlp.dense_h_to_4h.weight.data.numpy().tofile(f)
        m.mlp.dense_h_to_4h.bias.data.numpy().tofile(f)
        m.mlp.dense_4h_to_h.weight.data.numpy().tofile(f)
        m.mlp.dense_4h_to_h.bias.data.numpy().tofile(f)

        x1.numpy().tofile(f)
        y1.data.numpy().tofile(f)
        x2.numpy().tofile(f)
        y2.data.numpy().tofile(f)
        x3.numpy().tofile(f)
        y3.data.numpy().tofile(f)


def make_data_glm2_block():
    sys.path.append(str(CHATGLM2_MODEL_PATH))
    from modeling_chatglm import GLMBlock, RotaryEmbedding
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CHATGLM2_MODEL_PATH, trust_remote_code=True)
    config.layernorm_epsilon = 1e-5
    config.hidden_size = 32
    config.num_attention_heads = 8
    config.multi_query_group_num = 2
    config.ffn_hidden_size = 6
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.torch_dtype = torch.float32
    m = GLMBlock(config, layer_number=3).eval()
    m.input_layernorm.weight.data.uniform_()
    m.post_attention_layernorm.weight.data.uniform_()

    seq_length = 3
    rotary_dim = config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
    rotary_pos_emb_module = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope)
    rotary_pos_emb = rotary_pos_emb_module(8)[None, :seq_length].transpose(0, 1).contiguous()

    # self attention
    x1 = torch.randn(seq_length, 1, config.hidden_size)
    with torch.no_grad():
        y1, kv_cache = m(x1, attention_mask=None, rotary_pos_emb=rotary_pos_emb)

    # cross attention
    position_ids = torch.tensor([[seq_length]])
    rotary_pos_emb = rotary_pos_emb_module(8)[position_ids].transpose(0, 1).contiguous()
    x2 = torch.randn(1, 1, config.hidden_size)
    with torch.no_grad():
        y2, kv_cache = m(x2, attention_mask=None, rotary_pos_emb=rotary_pos_emb, kv_cache=kv_cache)

    # cross attention
    position_ids = torch.tensor([[seq_length + 1]])
    rotary_pos_emb = rotary_pos_emb_module(8)[position_ids].transpose(0, 1).contiguous()
    x3 = torch.randn(1, 1, config.hidden_size)
    with torch.no_grad():
        y3, kv_cache = m(x3, attention_mask=None, rotary_pos_emb=rotary_pos_emb, kv_cache=kv_cache)

    print(m)

    with open(HERE / "data/glm2_block.data", "wb") as f:
        m.input_layernorm.weight.data.numpy().tofile(f)
        m.self_attention.query_key_value.weight.data.numpy().tofile(f)
        m.self_attention.query_key_value.bias.data.numpy().tofile(f)
        m.self_attention.dense.weight.data.numpy().tofile(f)
        m.post_attention_layernorm.weight.data.numpy().tofile(f)
        m.mlp.dense_h_to_4h.weight.data.numpy().tofile(f)
        m.mlp.dense_4h_to_h.weight.data.numpy().tofile(f)

        x1.numpy().tofile(f)
        y1.numpy().tofile(f)
        x2.numpy().tofile(f)
        y2.numpy().tofile(f)
        x3.numpy().tofile(f)
        y3.numpy().tofile(f)


def make_data_baichuan13b_block():
    sys.path.append(str(BAICHUAN13B_MODEL_PATH))
    from modeling_baichuan import BaichuanModel
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(BAICHUAN13B_MODEL_PATH, trust_remote_code=True)
    config.hidden_size = 32
    config.num_attention_heads = 8
    config.intermediate_size = config.hidden_size * 3
    config.num_hidden_layers = 1
    config.torch_dtype = torch.float32
    config.vocab_size = 5

    m = BaichuanModel(config).eval()
    for param in m.parameters():
        param.data.uniform_(-0.5, 0.5)

    seq_len = 3

    # self attention
    x1 = torch.arange(seq_len, dtype=torch.int64)[None, :]
    attn_mask = torch.ones(1, seq_len, dtype=torch.int64)
    with torch.no_grad():
        out = m(x1, attention_mask=attn_mask, use_cache=True)
        y1 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x2 = torch.tensor([[seq_len]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 1, dtype=torch.int64)
    with torch.no_grad():
        out = m(x2, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y2 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x3 = torch.tensor([[seq_len + 1]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 2, dtype=torch.int64)
    with torch.no_grad():
        out = m(x3, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y3 = out.last_hidden_state
        kv_cache = out.past_key_values

    print(m)

    with open(HERE / "data/baichuan13b_block.data", "wb") as f:
        m.embed_tokens.weight.data.numpy().tofile(f)
        m.layers[0].input_layernorm.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.W_pack.weight.data.numpy().tofile(f)
        m.layers[0].self_attn.o_proj.weight.data.numpy().tofile(f)
        m.layers[0].post_attention_layernorm.weight.data.numpy().tofile(f)
        m.layers[0].mlp.gate_proj.weight.data.numpy().tofile(f)
        m.layers[0].mlp.down_proj.weight.data.numpy().tofile(f)
        m.layers[0].mlp.up_proj.weight.data.numpy().tofile(f)
        m.norm.weight.data.numpy().tofile(f)

        x1.int().numpy().tofile(f)
        y1.numpy().tofile(f)
        x2.int().numpy().tofile(f)
        y2.numpy().tofile(f)
        x3.int().numpy().tofile(f)
        y3.numpy().tofile(f)


def main():
    torch.manual_seed(0)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    # make_data_linear()
    # make_data_layernorm()
    # make_data_rms_norm()
    # make_data_glm_block()
    # make_data_glm2_block()
    make_data_baichuan13b_block()


if __name__ == "__main__":
    main()
