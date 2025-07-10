import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Literal

from src.nn.ca import CellularAutomata
from src.nn.perception import sobel_perpcetion
from src.nn.update import GrowingUpdate


class GrowingNCA(eqx.Module):
    img_size: tuple[int, int]
    state_size: int
    ca: CellularAutomata
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        img_size,
        hidden_size = 12,
        perception_type: Literal['sobel', 'fixed'] = 'sobel',
        update_width = 128,
        update_depth = 1,
        update_prob = 0.5,
        alive_threshold = 0.1,
        alive_index = 3,
        num_dev_steps = (48, 96),
        *,
        key
    ) -> None:
        super().__init__()

        state_size = hidden_size + 4
        conv_key, update_key = jr.split(key)

        # Perception function
        if perception_type == 'sobel':
            perception_fn = sobel_perpcetion
        else:
            perception_fn = nn.Conv2d(
                in_channels=state_size,
                out_channels=state_size,
                kernel_size=3,
                padding=1,
                padding_mode='wrap',
                groups=state_size,
                key=conv_key
            )

        # Update function
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        layer_input_size, layers = perception_out_size, []
        for _ in range(update_depth):
            update_depth, conv_key = jr.split(update_key)
            layers.extend([
                nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
                nn.Lambda(jax.nn.relu),
            ])
            layer_input_size = update_width
        layers.append(
            nn.Conv2d(layer_input_size, state_size, kernel_size=1, key=update_key)
        )

        update_fn = GrowingUpdate(
            nn.Sequential(layers),
            alive_threshold,
            alive_index,
            update_prob
        )

        self.img_size = img_size
        self.state_size = state_size
        self.ca = CellularAutomata(perception_fn, update_fn)
        self.num_dev_steps = num_dev_steps

    def init(self, key=None):
        H, W = self.img_size
        return jnp.zeros((self.state_size, H, W)).at[3:, H // 2, W // 2].set(1.0)

    def __call__(self, key, steps=None):
        if steps is None:
            steps = self.num_dev_steps
        init_state = self.init(key)
        cell_states, dev_path = self.ca(init_state, steps, key=key)
        return cell_states[:4], dev_path
