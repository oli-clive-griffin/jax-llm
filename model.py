import jax
from jax import tree_util
from dataclasses import dataclass
from jax import random, Array
import jax.numpy as jnp
from einops import einsum, rearrange
import jax.nn
from functools import partial

@jax.jit
def mha(residual_SDm: Array, w_qkv_HDm3Dh):
    '''
    residual_SD: (seq_len, dim_model)
    w_qkv_HDm3Dh: (n_heads, d_model 3 * d_head)

    returns: out_HSDh: (n_heads, seq_len, d_head)
    '''
    # qkv_HSDhx3 = einsum(residual_SDm, w_qkv_HDm3Dh, "s dm, h dm dh_x3 -> h s dh_x3")
    qkv_HSDhx3 = residual_SDm @ w_qkv_HDm3Dh

    q_HSDh, k_HSDh, v_HSDh = jnp.split(qkv_HSDhx3, 3, axis=-1)

    # qkt_HSqSk = einsum(q_HSDh, k_HSDh, "h sq dh, h sk dh -> h sq sk") # QK^T
    # h sq dh @ h dh sk -> h sq sk
    qkt_HSqSk = q_HSDh @ jnp.matrix_transpose(k_HSDh)

    qkt_scaled_HSqSk = qkt_HSqSk / jnp.sqrt(qkt_HSqSk.shape[-1])
    attn_HSqSk = jax.nn.softmax(qkt_scaled_HSqSk, axis=-1)

    # outs_HSDh = einsum(attn_HSqSk, v_HSDh, 'h sq sk, h sk dh -> h sq dh')
    outs_HSDh = attn_HSqSk @ v_HSDh

    # out_SDh = rearrange(outs_HSDh, 'h s dh -> s (h dh)')
    # out_SDh = jnp.permute_dims(outs_HSDh, (0, 2, 3, 1)).reshape((residual_SDm.shape[0], residual_SDm.shape[1], -1))
    out_SDh = jnp.permute_dims(outs_HSDh, (1, 2, 0)).reshape((residual_SDm.shape[0], -1))

    return out_SDh

def mha_weights(D_model, D_head, n_heads, key):
    init = jax.nn.initializers.xavier_normal()
    w_qkv_HDm3Dh = init(key, (n_heads, D_model, 3 * D_head))
    return w_qkv_HDm3Dh


@jax.jit
def ff(x_SDm: Array, params: tuple[Array, Array, Array, Array]):
    w1_DmDff, b1_Dff, w2_DffDm, b2_Dm = params
    x_SDff = x_SDm @ w1_DmDff + b1_Dff
    x_SDm = x_SDff @ w2_DffDm + b2_Dm
    return x_SDm

def ff_weights(D_model, D_ff, key):
    init1 = jax.nn.initializers.xavier_normal()
    init2 = jax.nn.initializers.xavier_normal()
    w1_DmDff = init1(key, (D_model, D_ff))
    b1_Dff = jnp.zeros(D_ff)
    w2_DffDm = init2(key, (D_ff, D_model))
    b2_Dm = jnp.zeros(D_model)
    return w1_DmDff, b1_Dff, w2_DffDm, b2_Dm


@jax.jit
def block(res_SDm: Array, params):
    w_qkv_HDm3Dh, ff_params = params

    x = layer_norm(res_SDm)

    # x = jax.vmap(mha, in_axes=(0, None))(x, w_qkv_HDm3Dh=w_qkv_HDm3Dh)
    x = mha(x, w_qkv_HDm3Dh=w_qkv_HDm3Dh)

    x = layer_norm(x)

    # x = jax.vmap(ff, in_axes=(0, None))(x, params=ff_params)
    x = ff(x, params=ff_params)

    return x

def block_weights(D_model, D_head, n_heads, D_ff, key):
    w_qkv_HDm3Dh = mha_weights(D_model, D_head, n_heads, key)
    ff_params = ff_weights(D_model, D_ff, key)
    return w_qkv_HDm3Dh, ff_params


@jax.jit
def blocks(x_SDm: Array, blocks_params):
    for block_params in blocks_params:
        # x_SDm = jax.vmap(block, in_axes=(0, None))(x_SDm, block_params)
        x_SDm = block(x_SDm, block_params)
    return x_SDm

def make_blocks_weights(D_model, D_head, n_heads, D_ff, n_blocks, key):
    return [block_weights(D_model, D_head, n_heads, D_ff, key) for _ in range(n_blocks)]


# there's gotta be a better way to do this with onehot encoding like nn.Embedding
@jax.jit
def embed(x_SDi: Array, w_DiDm: Array):
    return x_SDi @ w_DiDm
def make_embed_weights(D_vocab, D_model, key):
    init = jax.nn.initializers.xavier_normal()
    w_DiDm = init(key, (D_vocab, D_model))
    return w_DiDm

@jax.jit
def unembed(x_SDm: Array, w_DmDi: Array):
    return x_SDm @ w_DmDi
def make_unembed_weights(D_model, D_vocab, key):
    init = jax.nn.initializers.xavier_normal()
    w_DmDi = init(key, (D_model, D_vocab))
    return w_DmDi


def layer_norm(x, epsilon=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)


@dataclass
class ModelCfg:
    D_vocab: int
    D_model: int
    D_head: int
    n_heads: int
    D_ff: int
    n_blocks: int

    def __post_init__(self):
        assert self.D_head * self.n_heads == self.D_model

@jax.jit
def model(x_SDi: Array, params):
    w_DiDm, blocks_params, w_DmDi = params
    # x_SDm = jax.vmap(embed, in_axes=(0, None))(x_SDi, w_DiDm)
    x_SDm = embed(x_SDi, w_DiDm)
    # x_SDm = jax.vmap(blocks, in_axes=(0, None))(x_SDm, blocks_params)
    x_SDm = blocks(x_SDm, blocks_params)
    # x_SDi = jax.vmap(unembed, in_axes=(0, None))(x_SDm, w_DmDi)
    x_SDi = unembed(x_SDm, w_DmDi)
    return x_SDi

def make_model_weights(cfg: ModelCfg, key):
    w_DiDm = make_embed_weights(cfg.D_vocab, cfg.D_model, key)
    blocks_params = make_blocks_weights(cfg.D_model, cfg.D_head, cfg.n_heads, cfg.D_ff, cfg.n_blocks, key)
    w_DmDi = make_unembed_weights(cfg.D_model, cfg.D_vocab, key)
    return w_DiDm, blocks_params, w_DmDi

@dataclass
class TrainCfg:
    lr: float
    batch_size: int
    seq_len: int
    n_epochs: int
    model_cfg: ModelCfg

def fake_main():
    cfg = TrainCfg(
        lr = 4e-4,
        n_epochs=1000,
        batch_size = 2,
        seq_len = 10,
        model_cfg = ModelCfg(
            D_vocab = 16,
            D_model = 32,
            D_head = 8,
            n_heads = 4,
            D_ff = 64,
            n_blocks = 8,
        )
    )

    key = random.PRNGKey(0)

    fake_train(cfg, key)

@jax.jit
def loss_fn(params, x, y):
    y_pred = jax.vmap(model, in_axes=(0, None))(x, params)
    # y_pred = model(x, params)
    return jnp.mean(jnp.square(y_pred - y))

@jax.jit
def update(params, grads, lr):
  return tree_util.tree_map(lambda w, g: w - lr * g, params, grads)

def create_ones_batch(batch_size: int, seq_len: int, d_vocab: int, key: Array):
    key, subkey = random.split(key)
    # x = random.randint(subkey, (batch_size, seq_len), 0, d_vocab)
    x = random.normal(subkey, (batch_size, seq_len, d_vocab))
    key, subkey = random.split(key)
    # y = random.randint(subkey, (batch_size, seq_len), 0, d_vocab)
    # y = random.ones(subkey, (batch_size, seq_len, d_vocab))
    y = jnp.ones((batch_size, seq_len, d_vocab))
    return x, y

def fake_train(cfg: TrainCfg, key: Array):
    params = make_model_weights(cfg.model_cfg, key)

    loss_grad = jax.value_and_grad(loss_fn)

    for epoch in range(cfg.n_epochs):
        key, subkey = random.split(key)
        x, y = create_ones_batch(cfg.batch_size, cfg.seq_len, cfg.model_cfg.D_vocab, subkey)
        loss, grads = loss_grad(params, x, y)
        params = update(params, grads, cfg.lr)
        print(f"epoch {epoch}, loss {loss}")

# def create_example(seq_len: int, d_vocab: int, key: Array):
#     key, subkey = random.split(key)
#     # x = random.randint(subkey, (batch_size, seq_len), 0, d_vocab)
#     x = random.normal(subkey, (seq_len, d_vocab))
#     key, subkey = random.split(key)
#     # y = random.randint(subkey, (batch_size, seq_len), 0, d_vocab)
#     y = random.normal(subkey, (seq_len, d_vocab))
#     return x, y, key

if __name__ == "__main__":
    fake_main()

@jax.jit
def attention(residual: Array, w_qkv_Dm3Dh):
    '''
    residual: (batch, seq_len, dim_model)
    w_qkv_Dm3Dh: (d_model, 3 * d_head)
    '''
    qkv = einsum(residual, w_qkv_Dm3Dh, "b s dm, dm dh_x3 -> b s dh_x3")
    q, k, v = jnp.split(qkv, 3, axis=-1)
    qkt = einsum(q, k, "b sq dh, b sk dh -> b sq sk") # QK^T
    qkt_scaled = qkt / jnp.sqrt(qkt.shape[-1])
    attn = jax.nn.softmax(qkt_scaled, axis=-1)
    out = einsum(attn, v, 'b s sk, b s dh -> b s dh')
    return out

#     print(jnp.einsum("batch dim1 dim2, dim2 dim3 -> batch dim1 dim2", jnp.ones((2, 3, 4)), jnp.ones((4, 5))))