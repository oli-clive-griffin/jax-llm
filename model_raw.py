import jax
from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp
import jax.nn


# optimise by precomputing the mask
def mask(s: int):
    return jnp.where(jnp.tril(jnp.ones((s, s))), 0, -1e9)


@jax.jit
def mha(qkv_HDm3Dh: Array, residual_SDm: Array):
    """
    residual_SD: (seq_len, dim_model)
    qkv_HDm3Dh: (n_heads, d_model 3 * d_head)

    returns: out_HSDh: (n_heads, seq_len, d_head)
    """

    # scaled Q @K^T
    qkv_HSDhx3 = residual_SDm @ qkv_HDm3Dh  # compute q, k and v in one go
    q_HSDh, k_HSDh, v_HSDh = jnp.split(qkv_HSDhx3, 3, axis=-1)  # then split them up
    qkt_HSqSk = q_HSDh @ jnp.matrix_transpose(k_HSDh)  # q * k^T
    D_k = k_HSDh.shape[-1]
    qkt_scaled_HSqSk = qkt_HSqSk / jnp.sqrt(D_k)  # scale by sqrt(d_k)
    S = qkt_scaled_HSqSk.shape[-1]

    # masked softmax on attn
    qkt_masked_HSqSk = qkt_scaled_HSqSk + mask(S)

    attn_HSqSk = jax.nn.softmax(qkt_masked_HSqSk, axis=-1)

    # dot with v
    out_HSDh = attn_HSqSk @ v_HSDh

    # concat heads
    out_SHDh = jnp.permute_dims(out_HSDh, (1, 0, 2))  # make last 2 dims 'head' and 'd_head'
    out_SDm = out_SHDh.reshape((out_SHDh.shape[0], -1))  # flatten the last 2 dims

    return out_SDm


def make_mha_params(d_model, d_head, n_heads, key):
    init = jax.nn.initializers.xavier_normal()
    w_qkv_HDm3Dh = init(key, (n_heads, d_model, 3 * d_head))
    return w_qkv_HDm3Dh


@jax.jit
def ff(params: tuple[Array, Array, Array, Array], x_SDm: Array):
    w1_DmDff, b1_Dff, w2_DffDm, b2_Dm = params
    x_SDff = x_SDm @ w1_DmDff + b1_Dff
    x_SDm = x_SDff @ w2_DffDm + b2_Dm
    return x_SDm


def make_ff_params(d_model, mlp_ratio, key):
    init1 = jax.nn.initializers.xavier_normal()
    init2 = jax.nn.initializers.xavier_normal()
    hidden_dim = d_model * mlp_ratio
    w1_DmDff = init1(key, (d_model, hidden_dim))
    b1_Dff = jnp.zeros(hidden_dim)
    w2_DffDm = init2(key, (hidden_dim, d_model))
    b2_Dm = jnp.zeros(d_model)
    return w1_DmDff, b1_Dff, w2_DffDm, b2_Dm


@jax.jit
def block(params, res_SDm: Array):
    ln1_params, w_qkv_HDm3Dh, ln2_params, ff_params = params
    x = layer_norm(ln1_params, res_SDm)
    x = mha(w_qkv_HDm3Dh, x)
    x = layer_norm(ln2_params, x)
    x = ff(ff_params, x)
    return x


def make_block_params(d_model, d_head, n_heads, mlp_ratio, key):
    ln1_params = make_layer_norm_params(d_model)
    mha_params = make_mha_params(d_model, d_head, n_heads, key)
    ln2_params = make_layer_norm_params(d_model)
    ff_params = make_ff_params(d_model, mlp_ratio, key)
    return ln1_params, mha_params, ln2_params, ff_params


@jax.jit
def blocks(blocks_params: Array, x_SDm: Array):
    for block_params in blocks_params:
        x_SDm = block(block_params, x_SDm)
    return x_SDm


def make_blocks_params(d_model, d_head, n_heads, mlp_ratio, n_layers, key):
    return [make_block_params(d_model, d_head, n_heads, mlp_ratio, key) for _ in range(n_layers)]


# there's gotta be a better way to do this with onehot encoding like nn.Embedding
@jax.jit
def embed(w_VDm: Array, x_S: Array):
    return jnp.take(w_VDm, x_S, axis=0)


def make_embed_params(d_vocab, d_model, key):
    init = jax.nn.initializers.xavier_normal()
    w_VDm = init(key, (d_vocab, d_model))
    return w_VDm


@jax.jit
def unembed(w_DmV: Array, x_SDm: Array):
    return x_SDm @ w_DmV


def make_unembed_params(d_model, d_vocab, key):
    init = jax.nn.initializers.xavier_normal()
    w_DmV = init(key, (d_model, d_vocab))
    return w_DmV


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

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


@jax.jit
def model(params, x_S: Array):
    w_VDm, blocks_params, final_layer_norm_params, w_DmV = params
    x_SDm = embed(w_VDm, x_S)
    x_SDm = blocks(blocks_params, x_SDm)
    x_SDm = layer_norm(final_layer_norm_params, x_SDm)
    x_SV = unembed(w_DmV, x_SDm)
    return x_SV


def make_model_params(cfg: ModelCfg, key):
    w_VDm = make_embed_params(cfg.d_vocab, cfg.d_model, key)
    blocks_params = make_blocks_params(cfg.d_model, cfg.d_head, cfg.n_heads, cfg.mlp_ratio, cfg.n_layers, key)
    final_layer_norm_params = make_layer_norm_params(cfg.d_model)
    w_DmV = make_unembed_params(cfg.d_model, cfg.d_vocab, key)
    return w_VDm, blocks_params, final_layer_norm_params, w_DmV


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    params = make_model_params(
        ModelCfg(
            d_vocab=16,
            d_model=128,
            n_heads=4,
            mlp_ratio=4,
            n_layers=2,
        ),
        subkey,
    )
    model_b = jax.vmap(model, in_axes=(None, 0))
    out = model_b(params, jnp.array([[4, 5, 3], [1, 2, 3]]))
