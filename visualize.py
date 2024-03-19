import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
matplotlib.use("Agg")

def plot_path(path, save_name):
    path = np.array(path)
    _, ax = plt.subplots(1, 1)
    ax.set_xlim(cf.GRID[0])
    ax.set_ylim(cf.GRID[1])
    ax.plot(path[:, 0], path[:, 1], "-o", color="red", alpha=0.4)
    plt.savefig(save_name)
    plt.close("all")

def save_gif(imgs, path, fps=32):
    imageio.mimsave(path, imgs, fps=fps)