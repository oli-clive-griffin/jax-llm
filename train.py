import jax
from jax import tree_util
from dataclasses import dataclass
from jax import random, Array
import jax.numpy as jnp
from model_raw import ModelCfg, model, make_model_weights
from functools import partial

jax.config.update("jax_debug_nans", True)


@dataclass
class TrainCfg:
    lr: float
    batch_size: int
    seq_len: int
    n_epochs: int
    model_cfg: ModelCfg


def fake_main():
    model_cfg = ModelCfg(
        d_vocab=16,
        d_model=128,
        n_heads=4,
        mlp_ratio=4,
        n_layers=2,
    )

    cfg = TrainCfg(
        lr=4e-4, n_epochs=100_000, batch_size=2, seq_len=10, model_cfg=model_cfg
    )

    key = random.PRNGKey(0)

    fake_train(cfg, key)


model_b = jax.vmap(model, in_axes=(0, None))


@jax.jit
def loss_fn(params, x_BS, y_BSV):
    y_pred_BSV = model_b(x_BS, params)
    return cross_entropy(y_pred_BSV, y_BSV)


@jax.jit
def cross_entropy(logits_x_BSV, y_BSV):
    return -jnp.mean(jnp.sum(y_BSV * jax.nn.log_softmax(logits_x_BSV), axis=-1))


@jax.jit
def update(params, grads, lr):
    return tree_util.tree_map(lambda w, g: w - lr * g, params, grads)


def create_id_batch(batch_size: int, seq_len: int, d_vocab: int, key: Array):
    key, subkey = random.split(key)
    x = random.randint(subkey, (batch_size, seq_len), 0, d_vocab)
    return x, x.copy()


def fake_train(cfg: TrainCfg, key: Array):
    params = make_model_weights(cfg.model_cfg, key)

    # loss_grad = partial(jax.value_and_grad(loss_fn), vocab_size=cfg.model_cfg.D_vocab)
    loss_grad = jax.value_and_grad(loss_fn)

    for epoch in range(cfg.n_epochs):
        key, subkey = random.split(key)
        x, y = create_id_batch(
            cfg.batch_size, cfg.seq_len, cfg.model_cfg.d_vocab, subkey
        )
        y_BSV = jax.nn.one_hot(y, cfg.model_cfg.d_vocab)
        loss, grads = loss_grad(params, x, y_BSV)
        params = update(params, grads, cfg.lr)
        if epoch % 100 == 0:
            print(f"epoch {epoch}, loss {loss}")


if __name__ == "__main__":
    fake_main()
