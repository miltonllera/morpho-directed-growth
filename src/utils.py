import os
import datetime
import yaml
from pathlib import Path
from typing import Any
from jaxtyping import PyTree

import numpy as np
import jax.random as jr
import equinox as eqx


def seed_everything(seed: int | None):
    if seed is None:
        seed = int(datetime.datetime.now().timestamp())

    rng = np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    jax_seed = rng.choice(2 ** 32 - 1)
    np.random.seed(rng.choice(2 ** 32 - 1))

    return jr.key(jax_seed)


#-------------------------------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


#--------------------------------------------- I/O ------------------------------------------------

def save_pytree(model: PyTree, save_folder: str | Path, file_name: str):
    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    eqx.tree_serialise_leaves(save_folder / f"{file_name}.eqx", model)


def load_pytree(save_folder: str | Path,  file_name: str, template: PyTree):
    save_folder = Path(save_folder)
    return eqx.tree_deserialise_leaves(save_folder / f"{file_name}.eqx", template)
