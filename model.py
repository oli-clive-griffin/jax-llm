import jax
from jax import random, Array
import jax.numpy as jnp
from einops import einsum, rearrange

def mha(residual_BSDm: Array, w_qkv_HDm3Dh):
    '''
    residual_BSD: (batch, seq_len, dim_model)
    w_qkv_HDm3Dh: (n_heads, d_model 3 * d_head)

    returns: out_BHSDh: (batch, seq_len, n_heads, d_head)
    '''
    qkv_BHSDhx3 = einsum(residual_BSDm, w_qkv_HDm3Dh, "b s dm, h dm dh_x3 -> b h s dh_x3")
    q_BHSDh, k_BHSDh, v_BHSDh = jnp.split(qkv_BHSDhx3, 3, axis=-1)
    qkt_BHSqSk = einsum(q_BHSDh, k_BHSDh, "b h sq dh, b h sk dh -> b h sq sk") # QK^T
    qkt_scaled_BHSqSk = qkt_BHSqSk / jnp.sqrt(qkt_BHSqSk.shape[-1])
    attn_BHSqSk = jax.nn.softmax(qkt_scaled_BHSqSk, axis=-1)
    outs_BHSDh = einsum(attn_BHSqSk, v_BHSDh, 'b h sq sk, b h sk dh -> b h sq dh')
    out_BSDh = rearrange(outs_BHSDh, 'b h s dh -> b s (h dh)')
    return out_BSDh

def mha_weights(D_model, D_head, n_heads, key):
    init = jax.nn.initializers.xavier_normal()
    w_qkv_HDm3Dh = init(key, (n_heads, D_model, 3 * D_head))
    return w_qkv_HDm3Dh


def ff(x_BSDm: Array, params: tuple[Array, Array, Array, Array]):
    w1_DmDff, b1_Dff, w2_DffDm, b2_Dm = params
    x_BSDff = einsum(x_BSDm, w1_DmDff, 'b s dm, dm dff -> b s dff') + b1_Dff
    x_BSDm = einsum(x_BSDff, w2_DffDm, 'b s dff, dff dm -> b s dm') + b2_Dm
    return x_BSDm

def ff_weights(D_model, D_ff, key):
    init1 = jax.nn.initializers.xavier_normal()
    init2 = jax.nn.initializers.xavier_normal()
    w1_DmDff = init1(key, (D_model, D_ff))
    b1_Dff = jnp.zeros(D_ff)
    w2_DffDm = init2(key, (D_ff, D_model))
    b2_Dm = jnp.zeros(D_model)
    return w1_DmDff, b1_Dff, w2_DffDm, b2_Dm


def block(res_BSDm: Array, params):
    w_qkv_HDm3Dh, ff_params = params
    x = layer_norm(res_BSDm)
    x = mha(x, w_qkv_HDm3Dh=w_qkv_HDm3Dh)
    x = layer_norm(x)
    x = ff(x, params=ff_params)
    return x

def block_weights(D_model, D_head, n_heads, D_ff, key):
    w_qkv_HDm3Dh = mha_weights(D_model, D_head, n_heads, key)
    ff_params = ff_weights(D_model, D_ff, key)
    return w_qkv_HDm3Dh, ff_params


def blocks(x_BSDm: Array, blocks_params):
    for block_params in blocks_params:
        x_BSDm = block(x_BSDm, block_params)
    return x_BSDm

def make_blocks_weights(D_model, D_head, n_heads, D_ff, n_blocks, key):
    return [block_weights(D_model, D_head, n_heads, D_ff, key) for _ in range(n_blocks)]

# there's gotta be a better way to do this with onehot encoding like nn.Embedding
def embed(x_BSDi: Array, w_DiDm: Array):
    return einsum(x_BSDi, w_DiDm, 'b s di, di dm -> b s dm')
def make_embed_weights(D_vocab, D_model, key):
    init = jax.nn.initializers.xavier_normal()
    w_DiDm = init(key, (D_vocab, D_model))
    return w_DiDm
def unembed(x_BSDm: Array, w_DmDi: Array):
    return einsum(x_BSDm, w_DmDi, 'b s dm, dm di -> b s di')
def make_unembed_weights(D_model, D_vocab, key):
    init = jax.nn.initializers.xavier_normal()
    w_DmDi = init(key, (D_model, D_vocab))
    return w_DmDi

def layer_norm(x, epsilon=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def model(x_BSDi: Array, params):
    w_DiDm, blocks_params, w_DmDi = params
    x_BSDm = embed(x_BSDi, w_DiDm)
    x_BSDm = blocks(x_BSDm, blocks_params)
    x_BSDi = unembed(x_BSDm, w_DmDi)
    return x_BSDi

def make_model_weights(D_vocab, D_model, D_head, n_heads, D_ff, n_blocks, key):
    w_DiDm = make_embed_weights(D_vocab, D_model, key)
    blocks_params = make_blocks_weights(D_model, D_head, n_heads, D_ff, n_blocks, key)
    w_DmDi = make_unembed_weights(D_model, D_vocab, key)
    return w_DiDm, blocks_params, w_DmDi

if __name__ == "__main__":
    B = 2
    S = 10

    D_vocab = 16
    D_model = 32
    D_head = 8
    n_heads = 4
    assert D_head * n_heads == D_model

    D_ff = 64
    n_blocks = 8
    key = random.PRNGKey(0)

    model_params = make_model_weights(
        D_vocab,
        D_model,
        D_head,
        n_heads,
        D_ff,
        n_blocks,
        key,
    )

    model_grad = jax.grad(model, has_aux=False)

    for _ in range(0, -1):
        x, y = create_example()
        y_hat = model_grad(x, model_params)
        loss = jnp.sum((y - y_hat) ** 2)        
        model_params = jax.tree.map(lambda p, g: p - 0.01 * g, model_params, y_hat)
        

        


    

def attention(residual_BSDm: Array, w_qkv_Dm3Dh):
    '''
    residual_BSD: (batch, seq_len, dim_model)
    w_qkv_Dm3Dh: (d_model, 3 * d_head)
    '''
    qkv = einsum(residual_BSDm, w_qkv_Dm3Dh, "b s dm, dm dh_x3 -> b s dh_x3")
    q_BSDh, k_BSDh, v_BSDh = jnp.split(qkv, 3, axis=-1)
    qkt_BSqSk = einsum(q_BSDh, k_BSDh, "b sq dh, b sk dh -> b sq sk") # QK^T
    qkt_scaled_BSqSk = qkt_BSqSk / jnp.sqrt(qkt_BSqSk.shape[-1])
    attn_BSqSk = jax.nn.softmax(qkt_scaled_BSqSk, axis=-1)
    out = einsum(attn_BSqSk, v_BSDh, 'b s sk, b s dh -> b s dh')
    return out
