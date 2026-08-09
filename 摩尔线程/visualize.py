#!/usr/bin/env python3
import argparse
import csv


def load_grid(path, np):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data")

    x_values = np.array(sorted({float(row["x"]) for row in rows}))
    y_values = np.array(sorted({float(row["y"]) for row in rows}))
    nx, ny = len(x_values), len(y_values)
    if nx * ny != len(rows):
        raise ValueError("CSV is not a complete rectangular grid")

    fields = {}
    for name in rows[0]:
        if name not in ("x", "y"):
            fields[name] = np.array([float(row[name]) for row in rows]).reshape(ny, nx)
    return x_values, y_values, fields


def main():
    parser = argparse.ArgumentParser(description="Visualize Monte Carlo Laplace validation")
    parser.add_argument("csv", nargs="?", default="field.csv")
    parser.add_argument("-o", "--output", default="field_validation.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Visualization requires NumPy and Matplotlib; install them with: "
            "python3 -m pip install numpy matplotlib"
        ) from error

    x, y, f = load_grid(args.csv, np)
    extent = (x[0], x[-1], y[0], y[-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [
        ("u", "Monte Carlo potential $u_{MC}$", "coolwarm"),
        ("u_true", "Analytic potential $x^2-y^2$", "coolwarm"),
        ("u_error", "Signed potential error", "RdBu_r"),
        ("abs_u_error", "Absolute potential error", "magma"),
        ("Ex", "Computed field $E_x$", "coolwarm"),
        ("Ey", "Computed field $E_y$", "coolwarm"),
    ]
    for axis, (name, title, cmap) in zip(axes.flat, panels):
        data = f[name]
        kwargs = {}
        if name == "u_error":
            limit = max(float(np.max(np.abs(data))), np.finfo(float).eps)
            kwargs.update(vmin=-limit, vmax=limit)
        image = axis.imshow(data, origin="lower", extent=extent, aspect="equal",
                            interpolation="nearest", cmap=cmap, **kwargs)
        axis.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(image, ax=axis, shrink=0.82)

    rmse = np.sqrt(np.mean(f["u_error"] ** 2))
    max_error = np.max(f["abs_u_error"])
    fig.suptitle(f"Unit-square Laplace validation: RMSE={rmse:.4e}, "
                 f"max |error|={max_error:.4e}")
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
