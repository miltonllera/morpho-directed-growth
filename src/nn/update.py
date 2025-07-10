import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Callable


class GrowingUpdate(eqx.Module):
    update_fn: Callable
    alive_threshold: float
    alive_index: int
    update_prob: float
    max_pool: nn.MaxPool2d

    def __init__(self, update_fn, alive_threshold=0.1, alive_index=3, update_prob=0.5):
        assert 0 < alive_threshold < 1.0
        assert 0 < update_prob <= 1.0

        self.update_fn = update_fn
        self.alive_threshold = alive_threshold
        self.alive_index = alive_index
        self.max_pool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.update_prob = update_prob

    def __call__(self, state: jax.Array, perception: jax.Array, key: jax.Array):
        pre_alive = self.alive_state(state)
        state = state + self.update_fn(perception) * self.stochastic_update(state.shape, key)
        alive = self.alive_state(state) * pre_alive
        return state * alive

    def alive_state(self, state: jax.Array):
        max_alive = self.max_pool(state[self.alive_index:self.alive_index+1])
        # return (max_alive > self.alive_threshold)
        return jnp.floor(max_alive.clip(0, 1) + (1 - self.alive_threshold))

    def stochastic_update(self, state_shape, key: jax.Array):
        _, H, W = state_shape
        # return jr.bernoulli(key, self.update_prob, (1, H, W))
        return jnp.floor(jr.uniform(key, (1, H, W)) + self.update_prob)
