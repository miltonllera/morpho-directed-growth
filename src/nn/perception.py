from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array


conv2d = partial(jax.scipy.signal.convolve2d, mode='same')


def sobel_perpcetion(inputs: Array, key=None):
    kernel_x = jnp.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ]) / 8.0

    kernel_y = kernel_x.T

    x_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, kernel_x)
    y_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, kernel_y)

    return jnp.concat([inputs, x_conv, y_conv], axis=0)
