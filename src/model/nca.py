from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from colorsys import hsv_to_rgb
from typing import Literal
from jaxtyping import Array, Float

from src.nn.ca import CellularAutomata
from src.nn.perception import sobel_perception, steerable_perception
from src.nn.update import GrowingUpdate


#----------------------------------------- GrowingNCA ---------------------------------------------

class GrowingNCA(eqx.Module):
    img_size: tuple[int, int]
    state_size: int
    ca: CellularAutomata
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        img_size,
        hidden_size = 12,
        perception_type: Literal['sobel', 'learned'] = 'sobel',
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
            perception_fn = sobel_perception
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


#----------------------------------------- SteerableNCA -------------------------------------------

class SteerableNCA(eqx.Module):
    img_size: tuple[int, int]
    state_size: int
    ca: CellularAutomata
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        img_size,
        hidden_size = 12,
        perception_type: Literal['steerable', 'steerable_with_laplace'] = 'steerable',
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

        state_size = hidden_size + 5
        conv_key, update_key = jr.split(key)

        # Perception function
        if perception_type == 'steerable':
            perception_fn = steerable_perception
        else:
            perception_fn = partial(steerable_perception, use_laplace=True)

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
        return init_radial_seeds(
            input_shape=self.img_size,
            n_seeds=2,
            seed_radius=4.0,
            vis_chn=3,
            hidden_chn=self.state_size - 4,
            angle_chn=1,
        )

    def __call__(self, key, steps=None):
        if steps is None:
            steps = self.num_dev_steps
        init_state = self.init(key)
        cell_states, dev_path = self.ca(init_state, steps, key=key)
        return cell_states[:4], dev_path



def rgb_linspace(n):
    '''Generates n visually distinct rgb combinations'''
    return np.asarray([hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)], dtype=np.float32)


# def sinusoidal_embeddings(state_dim, x, y):
#     t = state_dim // 4
#     freqs = jnp.arange(t)

#     cos_x = jnp.cos(freqs * x[..., None])
#     sin_x = jnp.sin(freqs * x[..., None])

#     cos_y = jnp.cos(freqs * y[..., None])
#     sin_y = jnp.sin(freqs * y[..., None])

#     pos_x = jnp.stack([cos_x, sin_x], axis=-1).flatten(-2, -1)
#     pos_y = jnp.stack([cos_y, sin_y], axis=-1).flatten(-2, -1)

#     pos = jnp.cat([pos_x, pos_y], axis=1)
#     return pos

# def rotate_n(x, n, min=0., max=360.):
#     a = np.linspace(min, max, n)
#     for i, a in zip(range(n), a):
#         x[i] = T.rotate(x[i], a)
#     return x


def init_radial_seeds(
    input_shape: tuple[int, int],
    n_seeds: int,
    seed_radius: float,
    vis_chn: int,
    hidden_chn: int,
    angle_chn: int,
    angle: float = 0.0
) -> Float[Array, "C H W"]:
    H, W = input_shape
    total_channels = vis_chn + hidden_chn + angle_chn
    x = np.zeros((total_channels, H, W), dtype=np.float32)

    if angle_chn == 1:
        x[:, -1] = 0.0

    t = np.linspace(
        start=0.0,
        stop=2.0 * np.pi,
        num=n_seeds  ,
        endpoint=False,
        dtype=np.float32,
    )

    xy = [seed_radius * np.cos(t), seed_radius * np.sin(t)]
    xy = np.stack(xy, axis=0).astype(np.int32) + (H // 2)

    # this is where the morphogens are sampled, currently to fixed values
    # x[:vis_chn, xy[1], xy[0]] = rgb_linspace(xy.shape[1]).astype(np.float32).T
    x[vis_chn:vis_chn + hidden_chn:, xy[1], xy[0]] = 1.0
    x[-angle_chn:, xy[1], xy[0]] = t
    # x[:, :vis_chn, xy[1], xy[0]] = 1.0
    # x[:, vis_chn:vis_chn + hidden_chn, xy[1], xy[0]] = sinusoidal_embeddings(
    #     hidden_chn, xy[1], xy[0]
    # ).T

    # x = rotate_n(x, B, n_seeds) if angle is None else T.rotate(x, angle)
    return x  # type: ignore
