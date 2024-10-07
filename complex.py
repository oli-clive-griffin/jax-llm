import numpy as np
import jax.numpy as jnp

# Create a sample array
x = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.float32)
x = x.reshape(x     )
print("Original shape:", x.shape)  # (2, 4)

# View as complex
x_complex = x.view(dtype=jnp.complex64)
print("Complex view shape:", x_complex.shape)  # (2, 2)