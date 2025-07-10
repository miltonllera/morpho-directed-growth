import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import equinox as eqx
from typing import Callable

State = tuple[jax.Array, jax.Array]


class CellularAutomata(eqx.Module):
    perception_fn: Callable
    update_fn: Callable

    def __init__(self, perception_fn: Callable, update_fn: Callable):
        super().__init__()
        self.perception_fn = perception_fn
        self.update_fn = update_fn

    def __call__(self, states: jax.Array, n_steps: int | tuple[int, int], key: jax.Array):
        carry_key, sample_key = jr.split(key)
        num_dev_steps, max_steps = self.sample_num_steps(n_steps, sample_key)

        def f(carry: State, step):
            cell_states, key = carry
            p_key, u_key, key = jr.split(key, 3)

            perception = self.perception_fn(cell_states, key=p_key)
            updated_states = self.update_fn(cell_states, perception, key=u_key)

            cell_states = lax.select(
                step >= num_dev_steps,
                cell_states,
                updated_states
            )

            return (cell_states, key), cell_states

        init_state = states, carry_key
        _, dev_path = lax.scan(f, init_state, jnp.arange(max_steps))
        return dev_path[num_dev_steps-1], dev_path

    def sample_num_steps(self, n_steps: int | tuple[int, int] , key: jax.Array):
        if isinstance(n_steps, int):
            return n_steps, n_steps
        return jr.randint(key, (1,), *n_steps).squeeze(), n_steps[1]
