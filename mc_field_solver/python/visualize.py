#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize Monte Carlo field solver CSV output")
    parser.add_argument("directory", nargs="?", default="output")
    parser.add_argument("--save", help="write a PNG instead of opening a window")
    args = parser.parse_args()
    root = Path(args.directory)
    v = np.loadtxt(root / "potential.csv", delimiter=",")
    err = np.loadtxt(root / "stderr.csv", delimiter=",")
    ex = np.loadtxt(root / "Ex.csv", delimiter=",")
    ey = np.loadtxt(root / "Ey.csv", delimiter=",")
    mag = np.loadtxt(root / "E.csv", delimiter=",")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, data, title in ((axes[0, 0], v, "Potential V"),
                            (axes[0, 1], mag, "Field magnitude |E|"),
                            (axes[1, 0], err, "Monte Carlo standard error")):
        image = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), aspect="equal")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
    step = max(1, min(v.shape) // 20)
    y, x = np.mgrid[0:1:complex(v.shape[0]), 0:1:complex(v.shape[1])]
    axes[1, 1].quiver(x[::step, ::step], y[::step, ::step],
                      ex[::step, ::step], ey[::step, ::step])
    axes[1, 1].set_title("Electric field vector")
    axes[1, 1].set_aspect("equal")
    if args.save:
        fig.savefig(args.save, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()
