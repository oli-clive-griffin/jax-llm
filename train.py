import jax
from jax import tree_util
from dataclasses import dataclass
from jax import random, Array
import jax.numpy as jnp
from model_raw import ModelCfg, model, make_model_weights

@dataclass
class TrainCfg:
    lr: float
    batch_size: int
    seq_len: int
    n_epochs: int
    model_cfg: ModelCfg


def fake_main():
    model_cfg = ModelCfg(
        D_vocab=16,
        D_model=32,
        D_head=8,
        n_heads=4,
        D_ff=64,
        n_blocks=8,
    )

    cfg = TrainCfg(
        lr=4e-4, n_epochs=1000, batch_size=2, seq_len=10, model_cfg=model_cfg
    )

    key = random.PRNGKey(0)

    fake_train(cfg, key)


model_b = jax.vmap(model, in_axes=(0, None))


@jax.jit
def loss_fn(params, x, y):
    y_pred = model_b(x, params)
    return jnp.mean(jnp.square(y_pred - y))


@jax.jit
def update(params, grads, lr):
    return tree_util.tree_map(lambda w, g: w - lr * g, params, grads)


def create_ones_batch(batch_size: int, seq_len: int, d_vocab: int, key: Array):
    key, subkey = random.split(key)
    x = random.normal(subkey, (batch_size, seq_len, d_vocab))
    key, subkey = random.split(key)
    y = jnp.ones((batch_size, seq_len, d_vocab))
    return x, y

def fake_train(cfg: TrainCfg, key: Array):
    params = make_model_weights(cfg.model_cfg, key)

    loss_grad = jax.value_and_grad(loss_fn)

    for epoch in range(cfg.n_epochs):
        key, subkey = random.split(key)
        x, y = create_ones_batch(
            cfg.batch_size, cfg.seq_len, cfg.model_cfg.D_vocab, subkey
        )
        loss, grads = loss_grad(params, x, y)
        params = update(params, grads, cfg.lr)
        print(f"epoch {epoch}, loss {loss}")


if __name__ == "__main__":
    fake_main()