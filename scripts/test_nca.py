# import os
# import argparse
# import logging
# import wandb
# import math
# from datetime import datetime
# from pathlib import Path
# from dotenv import load_dotenv

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Callable
from jaxtyping import PyTree

from src.model.nca import GrowingNCA
from src.dataset.emoji import SingleEmojiDataset
from src.dataset.morphologies import SingleCompositeMorphology
from src.visualisation.utils import plot_img, plot_dev_path
from src.utils import save_pytree


def main():
    seed = np.random.choice(2 ** 32 - 1)
    nca = GrowingNCA((64, 64), num_dev_steps=(48, 96), key=jr.key(seed))

    # final_state, trace = jax.vmap(nca)(jr.split(jr.key(42), num=10))

    # dataset = SingleEmojiDataset('salamander', target_size=48, pad=8, batch_size=8)
    # target = jnp.asarray(next(iter(dataset))[1])
    dataset = SingleCompositeMorphology(
        'data/dataset/compositional_morphologies',
        shape_idx=1, batch_size=8
    )
    target = jnp.asarray(dataset[0][1])

    plot_img(target[0]).savefig("plots/target.png")
    # import matplotlib.pyplot as plt
    # plt.show()
    # exit()

    loss_fn = optax.l2_loss

    train_iters = 10_000
    # warmup_iters = 2_000
    # lr_or_schedule = optax.warmup_cosine_decay_schedule(
    #     0.0, 2e-3, warmup_iters, train_iters, 2e-3 / 10 ** 2
    # )
    lr_or_schedule = 2e-3

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_or_schedule),
    )
    opt_state = optim.init(eqx.filter(nca, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: PyTree,
        target: jax.Array,
        opt_state: PyTree,
        key: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, loss_fn, target, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss_value, model, opt_state

    def compute_loss(model: Callable, loss: Callable, target: jax.Array, key: jax.Array):
        batch_key = jr.split(key, target.shape[0])
        preds, _ = jax.vmap(model)(batch_key)
        return jnp.sum(loss(preds, target)) / len(target)

    key = jr.key(seed)
    pbar = tqdm(range(train_iters))
    for i in pbar:
        key, step_key = jr.split(key, 2)
        train_loss, nca, opt_state = train_step(nca, target, opt_state, step_key)
        pbar.set_postfix_str(f"iter: {i}; loss: {np.asarray(train_loss)}")

    output, dev_path = nca(jr.key(23), steps=96)

    plot_img(output).savefig("plots/example.png")
    plot_dev_path(dev_path[:, :4]).save("plots/growth.gif")
    save_pytree(nca, "data/examples/", "model_01")


if __name__ == "__main__":
    main()
