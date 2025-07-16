import os
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from tqdm import tqdm
from typing import Callable
from jaxtyping import PyTree

from src.model.nca import GrowingNCA, SteerableNCA
from src.dataset.emoji import SingleEmojiDataset, Emoji
from src.dataset.morphologies import SingleCompositeMorphology
from src.visualisation.utils import plot_img, plot_dev_path
from src.utils import save_pytree


def main(
    nca_type='steerable',
    rgb_init='none',
    hidden_init='constant',
    initial_angle='value',
    dataset = "compositional",
    target = 0,
    train_iters = 5000,
    lr = 2e-3,
    use_lr_schedule = False,
    save_folder = 'data/logs/temp'
):
    seed = np.random.choice(2 ** 32 - 1)

    # Init model
    if nca_type == "growing":
        nca = GrowingNCA(
            (64, 64),
            num_dev_steps=(48, 96),
            key=jr.key(seed)
        )

    elif nca_type == "growing-full-learnable":
        nca = GrowingNCA(
            (64, 64),
            perception_type="learned",
            num_dev_steps=(48, 96),
            key=jr.key(seed)
        )

    elif nca_type == "steerable":
        nca = SteerableNCA(
            (64, 64),
            perception_type="steerable",
            rgb_init=rgb_init,  # type: ignore
            hidden_init=hidden_init,  # type: ignore
            initial_angle=initial_angle,  # type: ignore
            num_dev_steps=(48, 96),
            key=jr.key(seed)
        )

    elif nca_type == "steerable-with-laplace":
        nca = SteerableNCA(
            (64, 64),
            perception_type="steerable_with_laplace",
            rgb_init=rgb_init,  # type: ignore
            hidden_init=hidden_init,  # type: ignore
            initial_angle=initial_angle,  # type: ignore
            num_dev_steps=(48, 96),
            key=jr.key(seed)
        )

    else:
        raise RuntimeError()

    # final_state, trace = jax.vmap(nca)(jr.split(jr.key(42), num=10))

    # Init dataset
    if dataset == "emoji":
        dataset = SingleEmojiDataset(
            Emoji(target).name,
            target_size=48,
            pad=8,
            batch_size=8
        )
        target = jnp.asarray(next(iter(dataset))[1])

    elif dataset == "compositional":
        dataset = SingleCompositeMorphology(
            'data/dataset/compositional_morphologies',
            shape_idx=1,
            batch_size=8
        )
        target = jnp.asarray(dataset[0][1])

    else:
        raise RuntimeError()

    # plot_img(target[0]).savefig("plots/target.png")
    # import matplotlib.pyplot as plt
    # plt.show()
    # exit()

    # Init training
    if use_lr_schedule:
        warmup_iters = 2_000
        lr_or_schedule = optax.warmup_cosine_decay_schedule(
            0.0, lr, warmup_iters, train_iters, lr / 10 ** 2
        )
    else:
        lr_or_schedule = lr

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_or_schedule),
    )
    opt_state = optim.init(eqx.filter(nca, eqx.is_array))

    # Train
    def compute_loss(model: Callable, target: jax.Array, key: jax.Array):
        batch_key = jr.split(key, target.shape[0])
        preds, _ = jax.vmap(model)(batch_key)
        return jnp.sum(optax.l2_loss(preds, target)) / len(target)

    @eqx.filter_jit
    def train_step(
        model: PyTree,
        target: jax.Array,
        opt_state: PyTree,
        key: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, target, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss_value, model, opt_state

    key = jr.key(seed)
    pbar = tqdm(range(train_iters))
    for i in pbar:
        key, step_key = jr.split(key, 2)
        train_loss, nca, opt_state = train_step(nca, target, opt_state, step_key)
        pbar.set_postfix_str(f"iter: {i}; loss: {np.asarray(train_loss)}")


    # Save results
    output, dev_path = nca(jr.key(23), steps=96)

    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    plot_img(output).savefig(save_folder / "example.png")
    plot_dev_path(dev_path[:, :4]).save(save_folder / "growth.gif")
    save_pytree(nca, save_folder, "checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--nca_type",
        choices=['growing', 'growing-full-learnable', 'steerable', 'steerable-with-laplace'],
        default='steerable',
    )
    parser.add_argument(
        "--rgb_init",
        choices=['none', 'angle_based'],
        default='angle_based',
    )
    parser.add_argument(
        "--hidden_init",
        choices=["constant", "sinusoidal"],
        default='constant',
    )
    parser.add_argument(
        "--initial_angle",
        choices=["value", "radial"],
        default="value",
    )
    parser.add_argument(
        "--dataset",
        choices=["emojis", "compositional"],
        default="compositional"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3
    )
    parser.add_argument(
        "--use_lr_schedule",
        action="store_true"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="data/logs/temp"
    )

    args = parser.parse_args()
    main(**vars(args))
