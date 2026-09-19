import jax, jax.numpy as jnp
print(jax.devices())
x = jnp.ones((200, 800))
print(jnp.dot(x.T, x).sum())