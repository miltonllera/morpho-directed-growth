import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_img(img: np.ndarray | jax.Array):
    img = np.asarray(img).transpose(1, 2, 0).clip(0.0, 1.0)
    plt.imshow(img, origin='lower', vmin=0, vmax=1)
    plt.gca().axis('off')
    return plt.gcf()


def plot_dev_path(dev_path: np.ndarray | jax.Array):
    fig = plt.figure()
    ax = plt.gca()
    ax.axis('off')

    dev_path = dev_path.transpose(0, 2, 3, 1).clip(0.0, 1.0)

    im = plt.imshow(dev_path[0], origin='lower', vmin=0, vmax=1)
    def animate(i):
        ax.set_title(f"Growth step: {i}")
        im.set_array(dev_path[i])
        return im,

    ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(dev_path))
    plt.close(fig)
    return ani
