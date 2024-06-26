import jax
from functools import partial
from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp
import jax.nn

# dims
# ====
# V: seq_len
# M: d_model
# N: n_heads
# H: d_head
# 3H: 3 x d_head
# A: attention dimension (d_head * n_heads)
# S: seq_len
# (no batch size cos vmap)


# should optimise by precomputing the mask
def mask(seq_len: int):
    return jnp.where(jnp.tril(jnp.ones((seq_len, seq_len))), 0, -1e9)


def fused_qkv(params: tuple[Array, Array], residual_SM: Array):
    qkv_NM3H, b_N3H = params
    qkv_NS3H = jnp.einsum("sm,nmh,nh->nsh", residual_SM, qkv_NM3H, b_N3H)
    q_NSH, k_NSH, v_NSH = jnp.split(qkv_NS3H, 3, axis=-1)
    return q_NSH, k_NSH, v_NSH


@jax.jit
def multihead_causal_self_attention(params, residual_SM: Array):
    fused_qkv_params, o_AM = params

    # scaled Q @K^T
    q_NSH, k_NSH, v_NSH = fused_qkv(fused_qkv_params, residual_SM)
    n_heads, S, d_head = q_NSH.shape

    qkt_NSqSk = q_NSH @ jnp.matrix_transpose(k_NSH)  # q * k^T
    D_k = k_NSH.shape[-1]
    qkt_scaled_NSqSk = qkt_NSqSk / jnp.sqrt(D_k)  # scale by sqrt(d_k)

    # masked softmax on attn
    qkt_masked_NSqSk = qkt_scaled_NSqSk + mask(S)
    attn_NSqSk = jax.nn.softmax(qkt_masked_NSqSk, axis=-1)

    # matmul with v
    x_NSH = attn_NSqSk @ v_NSH

    # concat heads
    x_SNH = jnp.permute_dims(x_NSH, (1, 0, 2))  # make last 2 dims 'head' and 'd_head'
    x_SA = x_SNH.reshape((S, n_heads * d_head))

    # Output linear layer
    out_SM = x_SA @ o_AM

    return out_SM


def make_mha_params(d_model, d_head, n_heads, key):
    init = jax.nn.initializers.xavier_normal()
    w_qkv_NM3H = init(key, (n_heads, d_model, 3 * d_head))
    b_qkv_N3H = init(key, (n_heads, 3 * d_head))
    d_attn = n_heads * d_head
    o_AM = init(key, (d_attn, d_model))
    return (w_qkv_NM3H, b_qkv_N3H), o_AM


def swish(beta: float | None, x: Array):
    return x * jax.nn.sigmoid(beta or 1.0 * x)

def swiglu(params: tuple[Array, ...], beta: float, x_SM: Array):
    w1_MH, b1_H, w3_HM, b3_M = params
    return swish(beta, x_SM @ w1_MH + b1_H) * swish(beta, x_SM @ w3_HM + b3_M)

def make_swiglu_params(d_model, mlp_ratio, key):
    hidden_dim = d_model * mlp_ratio
    init = jax.nn.initializers.xavier_normal()


    key, subkey = jax.random.split(key)
    w1_MH = init(subkey, (d_model, hidden_dim))
    b1_H = jnp.zeros(hidden_dim)

    key, subkey = jax.random.split(key)
    w3_HM = init(subkey, (d_model, hidden_dim))

    b3_M = jnp.zeros(hidden_dim)

    return w1_MH, b1_H, w3_HM, b3_M

@partial(jax.jit, static_argnames=('swish_beta',))
def ffn(params: tuple[Array, float], swish_beta: float, x_SM: Array):
    swiglu_params, w2_HM, b2_M = params
    x_SH = swiglu(swiglu_params, swish_beta, x_SM)
    x_SM = x_SH @ w2_HM + b2_M
    return x_SM


def make_ff_params(d_model: int, mlp_ratio: int, key):
    hidden_dim = d_model * mlp_ratio

    key, subkey = jax.random.split(key)
    swiglu_params = make_swiglu_params(d_model, mlp_ratio, subkey)

    init = jax.nn.initializers.xavier_normal()
    w2_HM = init(key, (hidden_dim, d_model))
    b2_M = jnp.zeros(d_model)

    return swiglu_params, w2_HM, b2_M


@partial(jax.jit, static_argnames=('swish_beta',))
def block(params, swish_beta, res_SM: Array):
    ln1_params, qkv_HM3H, ln2_params, ff_params = params
    x = layer_norm(ln1_params, res_SM)
    x = multihead_causal_self_attention(qkv_HM3H, x)
    x = layer_norm(ln2_params, x)
    x = ffn(ff_params, swish_beta, x)
    return x


def make_block_params(d_model, d_head, n_heads, mlp_ratio, key):
    ln1_params = make_layer_norm_params(d_model)
    mha_params = make_mha_params(d_model, d_head, n_heads, key)
    ln2_params = make_layer_norm_params(d_model)
    ff_params = make_ff_params(d_model, mlp_ratio, key)
    return ln1_params, mha_params, ln2_params, ff_params


def make_blocks_params(d_model, d_head, n_heads, mlp_ratio, n_layers, key):
    return [make_block_params(d_model, d_head, n_heads, mlp_ratio, key) for _ in range(n_layers)]


def make_embed_params(d_vocab, d_model, key):
    init = jax.nn.initializers.xavier_normal()
    w_VM = init(key, (d_vocab, d_model))
    return w_VM


def make_unembed_params(d_model, d_vocab, key):
    init = jax.nn.initializers.xavier_normal()
    w_MV = init(key, (d_model, d_vocab))
    return w_MV


def layer_norm(params, x, epsilon=1e-5):
    beta, gamma = params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    normed = (x - mean) / (std + epsilon)
    return gamma * normed + beta


def make_layer_norm_params(d_model: int):
    return jnp.zeros(d_model), jnp.ones(d_model)


@dataclass
class ModelCfg:
    d_vocab: int
    d_model: int
    n_heads: int
    mlp_ratio: int
    n_layers: int
    swish_beta: float = 1.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


@partial(jax.jit, static_argnames=('swish_beta',))
def model(params, swish_beta: float, x_S: Array):
    w_VM, blocks_params, final_layer_norm_params, w_MV = params

    # embed
    x_SM = w_VM[x_S]

    # layers
    for block_params in blocks_params:
        x_SM = block(block_params, swish_beta, x_SM)
    x_SM = layer_norm(final_layer_norm_params, x_SM)

    # unembed
    return x_SM @ w_MV


def make_model_params(cfg: ModelCfg, key):
    key, subkey = jax.random.split(key)
    w_VM = make_embed_params(cfg.d_vocab, cfg.d_model, subkey)

    key, subkey = jax.random.split(key)
    blocks_params = make_blocks_params(cfg.d_model, cfg.d_head, cfg.n_heads, cfg.mlp_ratio, cfg.n_layers, subkey)

    final_layer_norm_params = make_layer_norm_params(cfg.d_model)

    key, subkey = jax.random.split(key)
    w_MV = make_unembed_params(cfg.d_model, cfg.d_vocab, subkey)

    return w_VM, blocks_params, final_layer_norm_params, w_MV


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    cfg = ModelCfg(
        d_vocab=16,
        d_model=128,
        n_heads=4,
        mlp_ratio=4,
        n_layers=2,
        swish_beta=1.0,
    )
    params = make_model_params(cfg, subkey)
    model_b = jax.vmap(model, in_axes=(None, None, 0))
    out = model_b(params, cfg.swish_beta, jnp.array([[4, 5, 3], [1, 2, 3]]))
