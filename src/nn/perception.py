from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array


conv2d = partial(jax.scipy.signal.convolve2d, mode='same')

#-------------------------------------------- Kernels --------------------------------------------

sobel_x = jnp.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
]) / 8.0

sobel_y = sobel_x.T

laplace = jnp.array([
    [1.0,  2.0, 1.0],
    [2.0, -8.0, 2.0],
    [1.0,  2.0, 1.0]
]) / 16.0


#--------------------------------------- Perception functions ------------------------------------

def sobel_perception(inputs: Array, key=None):
    x_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, sobel_x)
    y_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, sobel_y)
    return jnp.concat([inputs, x_conv, y_conv], axis=0)


def steerable_perception(inputs: Array, use_laplace=False, key=None):
    state, angle = inputs[:-1], inputs[-1:]
    angle = angle % (2 * jnp.pi)

    x_conv = jax.vmap(conv2d, in_axes=(0, None))(state, sobel_x)
    y_conv = jax.vmap(conv2d, in_axes=(0, None))(state, sobel_y)

    c, s = jnp.cos(angle), jnp.sin(angle)

    rot_grad = jnp.concat([x_conv * c + y_conv * s, y_conv * c - x_conv * s], axis=0)

    if use_laplace:
        state_lap = jax.vmap(conv2d, in_axes=(0, None))(state, laplace)
        features = [state, rot_grad, state_lap]
    else:
        features = [state, rot_grad]

    return jnp.concat(features, axis=0)
