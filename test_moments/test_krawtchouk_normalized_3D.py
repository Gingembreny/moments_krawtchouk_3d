import gzip
import math
import os
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.special import loggamma


def open_text_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def open_binary_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_im_volume(path):
    """
    Load McGill .im/.im.gz voxel files.

    The files observed here store a 1024-byte header followed by a 128^3
    uint8 binary voxel volume.
    """
    with open_binary_maybe_gzip(path) as f:
        data = f.read()

    header_size = 1024
    payload = data[header_size:]
    side = round(len(payload) ** (1.0 / 3.0))

    if side ** 3 != len(payload):
        raise ValueError(
            f"Cannot infer cubic volume from {path}: payload has {len(payload)} bytes"
        )

    volume = np.frombuffer(payload, dtype=np.uint8).reshape((side, side, side))
    return (volume > 0).astype(np.float64)


def downsample_binary_volume(volume, target_size=None):
    """
    Downsample a cubic binary volume with max-pooling.

    target_size=None keeps the original resolution. If set, the original side
    length must be divisible by target_size, e.g. 128 -> 64 or 32.
    """
    if target_size is None or target_size == volume.shape[0]:
        return volume.astype(np.float64)

    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"Expected a cubic 3D volume, got shape {volume.shape}")

    source_size = volume.shape[0]
    if source_size % target_size != 0:
        raise ValueError(
            f"target_size ({target_size}) must divide source_size ({source_size})"
        )

    factor = source_size // target_size
    pooled = volume.reshape(
        target_size,
        factor,
        target_size,
        factor,
        target_size,
        factor,
    ).max(axis=(1, 3, 5))

    return pooled.astype(np.float64)


def load_ply_vertices(path):
    """Load vertices from an ASCII PLY or PLY.GZ file."""
    with open_text_maybe_gzip(path) as f:
        vertex_count = None
        fmt = None

        for line in f:
            line = line.strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line == "end_header":
                break

        if fmt != "ascii":
            raise ValueError(f"Only ASCII PLY is supported for now, got format={fmt!r}")
        if vertex_count is None:
            raise ValueError(f"No vertex count found in PLY header: {path}")

        vertices = np.zeros((vertex_count, 3), dtype=np.float64)
        for i in range(vertex_count):
            parts = f.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Invalid vertex line {i + 1} in {path}")
            vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return vertices


def vertices_to_volume(vertices, grid_size=32, fill_solid=True):
    """Convert point/mesh vertices to a fixed-size binary voxel volume."""
    vertices = np.asarray(vertices, dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)

    normalized = (vertices - mins) / span
    coords = np.rint(normalized * (grid_size - 1)).astype(np.int64)
    coords = np.clip(coords, 0, grid_size - 1)

    volume = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    volume[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    if fill_solid:
        # The PLY vertices describe a surface-like shape. Filling 2D slices
        # gives a simple solid approximation suitable for volume moments.
        for axis in range(3):
            volume = np.moveaxis(volume, axis, 0)
            for i in range(volume.shape[0]):
                volume[i] = binary_fill_holes(volume[i])
            volume = np.moveaxis(volume, 0, axis)

    return volume.astype(np.float64)


def pochhammer(a, n):
    if n == 0:
        return Fraction(1, 1)
    result = Fraction(1, 1)
    for i in range(n):
        result *= a + i
    return result


def krawtchouk_poly(n, x, sample_count, p):
    """
    Paper hypergeometric definition:
    K_n(x; p, N) = 2F1(-n, -x, -N, 1/p), x=0..N.

    Here sample_count is the number of grid samples, so N = sample_count - 1.
    """
    N = sample_count - 1
    a = Fraction(-n, 1)
    b = Fraction(-x, 1)
    c = Fraction(-N, 1)
    z = Fraction(1, 1) / Fraction(p)

    value = Fraction(0, 1)
    for k in range(n + 1):
        num = pochhammer(a, k) * pochhammer(b, k)
        den = pochhammer(c, k) * math.factorial(k)
        value += num / den * (z ** k)

    return float(value)


def log_weight(x, sample_count, p):
    """log w(x;p,N), with N = sample_count - 1 and x=0..N."""
    N = sample_count - 1
    return (
        loggamma(N + 1)
        - loggamma(x + 1)
        - loggamma(N - x + 1)
        + x * math.log(p)
        + (N - x) * math.log(1 - p)
    )


def rho(n, sample_count, p):
    """
    Paper normalization constant:
    rho(n;p,N) = (-1)^n n! / (-N)_n * ((1-p)/p)^n.

    Since (-N)_n = (-1)^n N!/(N-n)!, rho is positive and computed in log form.
    """
    if n == 0:
        return 1.0
    N = sample_count - 1
    log_rho = (
        n * math.log((1 - p) / p)
        + loggamma(n + 1)
        + loggamma(N - n + 1)
        - loggamma(N + 1)
    )
    return math.exp(log_rho)


def krawtchouk_normalized(n, x, sample_count, p):
    K = krawtchouk_poly(n, x, sample_count, p)
    w = math.exp(log_weight(x, sample_count, p))
    r = rho(n, sample_count, p)
    return K * math.sqrt(w / r)


def precompute_K(max_order, sample_count, p):
    if max_order > sample_count:
        raise ValueError(
            f"max_order ({max_order}) must be <= sample_count ({sample_count})"
        )

    K = np.zeros((max_order, sample_count), dtype=np.float64)
    for n in range(max_order):
        for x in range(sample_count):
            K[n, x] = krawtchouk_normalized(n, x, sample_count, p)
    return K


def compute_moments_3d(volume, Kx, Ky, Kz, order):
    Kx_o = Kx[:order]
    Ky_o = Ky[:order]
    Kz_o = Kz[:order]
    return np.einsum("xyz,nx,my,lz->nml", volume, Kx_o, Ky_o, Kz_o, optimize=True)


def reconstruct_3d(moments, Kx, Ky, Kz, order):
    Kx_o = Kx[:order]
    Ky_o = Ky[:order]
    Kz_o = Kz[:order]
    return np.einsum(
        "nml,nx,my,lz->xyz",
        moments[:order, :order, :order],
        Kx_o,
        Ky_o,
        Kz_o,
        optimize=True,
    )


def normalized_error(volume, recon_binary):
    denominator = np.sum(volume ** 2)
    if denominator == 0:
        return 0.0
    return np.sum((volume - recon_binary) ** 2) / denominator


def dice_score(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def hd95_volume(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    if not a.any() or not b.any():
        return np.inf

    dist_to_b = distance_transform_edt(~b)
    dist_to_a = distance_transform_edt(~a)
    directed = np.concatenate([dist_to_b[a], dist_to_a[b]])
    return np.percentile(directed, 95)


def error_analysis_3d(volume, recon_binary):
    volume = volume.astype(bool)
    recon_binary = recon_binary.astype(bool)

    tp = np.logical_and(volume, recon_binary).sum()
    fp = np.logical_and(~volume, recon_binary).sum()
    fn = np.logical_and(volume, ~recon_binary).sum()
    tn = np.logical_and(~volume, ~recon_binary).sum()

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }


def orthogonality_error(K, order):
    G = K[:order] @ K[:order].T
    return np.max(np.abs(G - np.eye(order)))


def plot_middle_slices(volume, reconstructions, orders, metrics):
    z_mid = volume.shape[2] // 2
    rows = 2
    cols = len(reconstructions) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 5.4))

    axes[0, 0].imshow(volume[:, :, z_mid], cmap="gray")
    axes[0, 0].set_title("Original\nXY middle", fontsize=10)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(volume[:, volume.shape[1] // 2, :], cmap="gray")
    axes[1, 0].set_title("Original\nXZ middle", fontsize=10)
    axes[1, 0].axis("off")

    for i, recon in enumerate(reconstructions):
        eps, dice, hd95, _ = metrics[i]
        title = (
            f"Order {orders[i]}\n"
            f"eps={eps:.3f}\n"
            f"DICE={dice:.3f}\n"
            f"HD95={hd95:.2f}"
        )

        axes[0, i + 1].imshow(recon[:, :, z_mid], cmap="gray")
        axes[0, i + 1].set_title(title, fontsize=9)
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(recon[:, recon.shape[1] // 2, :], cmap="gray")
        axes[1, i + 1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.82, wspace=0.28)
    plt.show()


def set_3d_axes_equal(ax, shape):
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    ax.set_box_aspect(shape)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_3d_reconstruction(volume, recon_binary, order):
    volume = volume.astype(bool)
    recon_binary = recon_binary.astype(bool)

    false_positive = np.logical_and(~volume, recon_binary)
    false_negative = np.logical_and(volume, ~recon_binary)
    true_positive = np.logical_and(volume, recon_binary)

    fig = plt.figure(figsize=(13, 4.4))
    ax_original = fig.add_subplot(1, 3, 1, projection="3d")
    ax_recon = fig.add_subplot(1, 3, 2, projection="3d")
    ax_error = fig.add_subplot(1, 3, 3, projection="3d")

    ax_original.voxels(volume, facecolors="#d9d9d9", edgecolor="none", alpha=0.75)
    ax_original.set_title("Original 3D volume")
    set_3d_axes_equal(ax_original, volume.shape)

    ax_recon.voxels(recon_binary, facecolors="#4c78a8", edgecolor="none", alpha=0.75)
    ax_recon.set_title(f"Reconstruction 3D\nOrder {order}")
    set_3d_axes_equal(ax_recon, volume.shape)

    error_volume = np.logical_or(true_positive, np.logical_or(false_positive, false_negative))
    colors = np.empty(error_volume.shape, dtype=object)
    colors[true_positive] = "#bdbdbd"
    colors[false_positive] = "#377eb8"
    colors[false_negative] = "#e41a1c"

    ax_error.voxels(error_volume, facecolors=colors, edgecolor="none", alpha=0.8)
    ax_error.set_title("Error map\nGray=TP, Blue=FP, Red=FN")
    set_3d_axes_equal(ax_error, volume.shape)

    for ax in (ax_original, ax_recon, ax_error):
        ax.view_init(elev=22, azim=35)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86, wspace=0.08)
    plt.show()


def main():
    im_path = "non-articulated/cups/cupsIm/b1.im.gz"
    fallback_im = "non-articulated/cups/cupsIm/b1.im"
    if not os.path.exists(im_path) and os.path.exists(fallback_im):
        im_path = fallback_im

    # Use None for the original 128^3 resolution, or 64/32/16 for faster tests.
    target_grid_size = None
    p = 0.5
    orders = [5, 10, 15, 20, 25, 30]
    max_order = max(orders)

    print(f"Using IM voxel file: {im_path}")
    source_volume = load_im_volume(im_path)
    print(f"Source volume shape: {source_volume.shape}")
    print(f"Source foreground ratio: {source_volume.mean():.4f}")

    volume = downsample_binary_volume(source_volume, target_grid_size)
    print(f"Volume shape: {volume.shape}")
    print(f"Foreground ratio: {volume.mean():.4f}")

    print("Precomputing normalized Krawtchouk bases...")
    Kx = precompute_K(max_order, volume.shape[0], p)
    Ky = precompute_K(max_order, volume.shape[1], p)
    Kz = precompute_K(max_order, volume.shape[2], p)

    for order in orders:
        err = orthogonality_error(Kx, order)
        print(f"Order {order}: 1D orthogonality error = {err:.3e}")

    reconstructions = []
    metrics = []
    for order in orders:
        moments = compute_moments_3d(volume, Kx, Ky, Kz, order)
        recon = reconstruct_3d(moments, Kx, Ky, Kz, order)
        recon_binary = (recon > 0.5).astype(np.float64)

        eps = normalized_error(volume, recon_binary)
        dice = dice_score(volume, recon_binary)
        hd95 = hd95_volume(volume, recon_binary)
        errors = error_analysis_3d(volume, recon_binary)

        reconstructions.append(recon_binary)
        metrics.append((eps, dice, hd95, errors))

        print(f"Order {order}: epsilon = {eps:.6f}")
        print(f"Order {order}: DICE = {dice:.4f}")
        print(f"Order {order}: HD95 = {hd95:.4f}")
        print(f"Order {order}: IoU = {errors['iou']:.4f}")
        print(f"Order {order}: Precision = {errors['precision']:.4f}")
        print(f"Order {order}: Recall = {errors['recall']:.4f}")
        print(f"Order {order}: TP/FP/FN = {errors['tp']}/{errors['fp']}/{errors['fn']}")
        print(f"Order {order}: recon range = {recon.min():.4f} .. {recon.max():.4f}")

    plot_middle_slices(volume, reconstructions, orders, metrics)
    plot_3d_reconstruction(volume, reconstructions[-1], orders[-1])


if __name__ == "__main__":
    main()
