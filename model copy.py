import jax
from dataclasses import dataclass
from jax import random, Array
import jax.numpy as jnp
from einops import einsum, rearrange
import jax.nn
from functools import partial

# vmap via a decorator


def test(a: Array, b: Array) -> Array:
    return a @ b.T

if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (10, 2, 3))
    y = random.normal(key, (2, 3))

    test_vmap = jax.vmap(test, in_axes=(0, None))
    print(test_vmap(x, y).shape)

    print(test(x, y).shape)


