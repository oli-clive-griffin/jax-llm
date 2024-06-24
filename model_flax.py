import jax.numpy as jnp
import flax.linen as nn
from model_raw import ModelCfg


class MHA(nn.Module):
    d_model: int
    n_heads: int
    d_head: int

    def setup(self):
        self.w_qkv = self.param(
            "w_qkv",
            nn.initializers.xavier_normal(),
            (self.n_heads, self.d_model, 3 * self.d_head),
        )

    @nn.compact
    def __call__(self, x):
        qkv = jnp.einsum("sd,hde->hshe", x, self.w_qkv)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        attn_weights = jnp.einsum("hsde,htde->hset", q, k) / jnp.sqrt(self.d_head)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        out = jnp.einsum("hset,htde->hsde", attn_weights, v)
        return out.reshape(x.shape[0], -1)


class FeedForward(nn.Module):
    d_model: int
    d_ff: int

    def setup(self):
        self.w1 = self.param(
            "w1", nn.initializers.xavier_normal(), (self.d_model, self.d_ff)
        )
        self.b1 = self.param("b1", nn.initializers.zeros, (self.d_ff,))
        self.w2 = self.param(
            "w2", nn.initializers.xavier_normal(), (self.d_ff, self.d_model)
        )
        self.b2 = self.param("b2", nn.initializers.zeros, (self.d_model,))

    @nn.compact
    def __call__(self, x):
        x = jnp.dot(x, self.w1) + self.b1
        x = jnp.dot(x, self.w2) + self.b2
        return x


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = MHA(self.d_model, self.n_heads, self.d_head)(x)
        x = nn.LayerNorm()(x)
        x = FeedForward(self.d_model, self.d_ff)(x)
        return x


class Model(nn.Module):
    d_vocab: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_blocks: int

    @classmethod
    def from_config(cls, cfg: ModelCfg):
        return cls(
            cfg.d_vocab, cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_ff, cfg.n_layers
        )

    @nn.compact
    def __call__(self, x):
        embed = self.param(
            "embed", nn.initializers.xavier_normal(), (self.d_vocab, self.d_model)
        )
        unembed = self.param(
            "unembed", nn.initializers.xavier_normal(), (self.d_model, self.d_vocab)
        )
        x = x @ embed

        for _ in range(self.n_blocks):
            x = TransformerBlock(self.d_model, self.n_heads, self.d_head, self.d_ff)(x)

        x = x @ unembed
        return x


# Usage
# model = Model(d_vocab=1000, d_model=512, n_heads=8, d_head=64, d_ff=2048, n_blocks=6)
# params = model.init(jax.random.PRNGKey(0), jnp.ones((10, 1000), jnp.float32))
# out = model.apply(params, jnp.ones((10, 1000), jnp.float32))
