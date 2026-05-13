import gzip
import math
import os
from fractions import Fraction

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.special import loggamma
from sklearn.decomposition import PCA
import mplcursors

def load_nifti_volume(path):
    """Load NIfTI volume."""
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float64)

def extract_muscle_volumes(segmentation):
    """Extract individual muscle volumes from segmentation."""
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
    muscle_volumes = {}
    for label in unique_labels:
        volume = (segmentation == label).astype(np.float64)
        muscle_volumes[int(label)] = volume
    return muscle_volumes

def center_and_pad_volume(volume, canvas_size=256):
    """Center the volume in a fixed-size canvas. canvas_size is the side length of the cubic canvas."""
    
    # Find bounding box of the muscle
    coords = np.argwhere(volume > 0.5)
    if len(coords) == 0:
        return np.zeros((canvas_size, canvas_size, canvas_size), dtype=np.float64)
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    muscle_size = max_coords - min_coords + 1
    
    # Create canvas
    canvas = np.zeros((canvas_size, canvas_size, canvas_size), dtype=np.float64)
    
    # Calculate position to center the muscle in canvas
    canvas_center = canvas_size // 2
    muscle_center_offset = muscle_size // 2
    
    # Start position in canvas to place the muscle
    start_in_canvas = canvas_center - muscle_center_offset
    end_in_canvas = start_in_canvas + muscle_size
    
    # Handle placement
    canvas_start = np.maximum(0, start_in_canvas)
    canvas_end = np.minimum(canvas_size, end_in_canvas)
    
    muscle_start = np.maximum(0, -start_in_canvas)
    muscle_end = muscle_start + (canvas_end - canvas_start)
    
    # Copy the muscle data
    canvas[canvas_start[0]:canvas_end[0],
           canvas_start[1]:canvas_end[1],
           canvas_start[2]:canvas_end[2]] = volume[min_coords[0] + muscle_start[0]:min_coords[0] + muscle_end[0],
                                                   min_coords[1] + muscle_start[1]:min_coords[1] + muscle_end[1],
                                                   min_coords[2] + muscle_start[2]:min_coords[2] + muscle_end[2]]
    
    return canvas

# Copy the Krawtchouk functions from test_krawtchouk_normalized_3D.py
def pochhammer(a, n):
    if n == 0:
        return Fraction(1, 1)
    result = Fraction(1, 1)
    for i in range(n):
        result *= a + i
    return result

def krawtchouk_poly(n, x, sample_count, p):
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
    N = sample_count - 1
    return (
        loggamma(N + 1)
        - loggamma(x + 1)
        - loggamma(N - x + 1)
        + x * math.log(p)
        + (N - x) * math.log(1 - p)
    )

def rho(n, sample_count, p):
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

def crop_to_foreground_pair(volume, recon_binary, padding=4):
    mask = np.logical_or(volume > 0.5, recon_binary > 0.5)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return volume, recon_binary

    min_coords = np.maximum(coords.min(axis=0) - padding, 0)
    max_coords = np.minimum(coords.max(axis=0) + padding + 1, volume.shape)
    slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    return volume[slices], recon_binary[slices]

def set_3d_axes_equal(ax, shape):
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])
    ax.set_box_aspect(shape)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def plot_3d_reconstruction(volume, recon_binary, order, label):
    volume, recon_binary = crop_to_foreground_pair(volume, recon_binary)
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
    ax_original.set_title(f"Original 3D\n{label}")
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

def moment_distance(M1, M2):
    diff = M1 - M2
    norm_M1 = np.linalg.norm(M1)
    if norm_M1 == 0:
        return np.linalg.norm(diff)
    return np.linalg.norm(diff) / norm_M1

def main():
    volume_path = "/Users/jineiya/Desktop/Projet_PIR/shoulder_TG/epaule_002_0000.nii.gz"
    segmentation_path = "/Users/jineiya/Desktop/Projet_PIR/shoulder_TG/epaule_002.nii.gz"
    
    print(f"Loading volume: {volume_path}")
    volume = load_nifti_volume(volume_path)
    print(f"Volume shape: {volume.shape}")
    
    print(f"Loading segmentation: {segmentation_path}")
    segmentation = load_nifti_volume(segmentation_path)
    print(f"Segmentation shape: {segmentation.shape}")
    
    muscle_volumes = extract_muscle_volumes(segmentation)
    print(f"Found {len(muscle_volumes)} muscle segments: {list(muscle_volumes.keys())}")
    
    # Keep the same full-size canvas for both inter-muscle analysis and
    # reconstruction tests so reconstruction quality is evaluated at 256^3.
    canvas_size = 256
    print(
        "Using canvas size for analysis and reconstruction tests: "
        f"{canvas_size}x{canvas_size}x{canvas_size}"
    )
    
    p = 0.5
    analysis_order = 10
    reconstruction_orders = [10]
    
    print("Precomputing normalized Krawtchouk bases for moment analysis...")
    Kx_analysis = precompute_K(analysis_order, canvas_size, p)
    Ky_analysis = precompute_K(analysis_order, canvas_size, p)
    Kz_analysis = precompute_K(analysis_order, canvas_size, p)

    print("Precomputing normalized Krawtchouk bases for reconstruction tests...")
    reconstruction_max_order = max(reconstruction_orders)
    Kx_recon = precompute_K(reconstruction_max_order, canvas_size, p)
    Ky_recon = precompute_K(reconstruction_max_order, canvas_size, p)
    Kz_recon = precompute_K(reconstruction_max_order, canvas_size, p)

    for order in reconstruction_orders:
        err = orthogonality_error(Kx_recon, order)
        print(f"Order {order}: 1D orthogonality error = {err:.3e}")
    
    features = []
    labels = []
    
    for label, vol in muscle_volumes.items():
        print(f"Processing muscle {label}...")
        # Print original size before centering
        coords = np.argwhere(vol > 0.5)
        if len(coords) > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            original_size = max_coords - min_coords + 1
            print(f"  Original bounding box size: {original_size}")
        else:
            print(f"  No foreground voxels found")
        centered_padded = center_and_pad_volume(vol, canvas_size)
        moments = compute_moments_3d(
            centered_padded,
            Kx_analysis,
            Ky_analysis,
            Kz_analysis,
            analysis_order,
        )
        features.append(moments.flatten())
        labels.append(f"Muscle {label}")
    
    features = np.array(features)
    print(f"Moment feature order for distance/PCA: {analysis_order}")
    print(f"Features shape: {features.shape}")

    # Distance matrix
    print("Computing distance matrix for fixed-order muscle moments...")
    N = len(features)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = moment_distance(features[i], features[j])

    plt.figure()
    im = plt.imshow(D, cmap='hot')
    plt.colorbar()
    plt.title("Distance matrix of muscle moments")
    plt.xticks(range(N), labels, rotation=45)
    plt.yticks(range(N), labels)

    # Add hover functionality
    cursor = mplcursors.cursor(im, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i, j = sel.index
        row_muscle = labels[i]
        col_muscle = labels[j]
        distance = D[i, j]
        sel.annotation.set_text(f"Row: {row_muscle}\nCol: {col_muscle}\nDistance: {distance:.4f}")

    plt.show()

    # PCA
    print("Computing PCA for fixed-order muscle moments...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)

    plt.figure()
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(labels)), cmap='viridis')

    # Add hover functionality
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        sel.annotation.set_text(labels[index])

    plt.title("PCA of muscle moments")
    plt.show()

    # Reconstruct the largest muscle using the same 3D moment pipeline.
    print("Starting reconstruction tests after fixed-order moment analysis...")
    largest_label = max(muscle_volumes.keys(), key=lambda l: np.sum(muscle_volumes[l]))
    print(f"Largest muscle label: {largest_label}")
    largest_vol = muscle_volumes[largest_label]
    centered_largest = center_and_pad_volume(largest_vol, canvas_size)
    largest_moments = compute_moments_3d(
        centered_largest,
        Kx_recon,
        Ky_recon,
        Kz_recon,
        reconstruction_max_order,
    )

    reconstructions = []
    metrics = []
    for order in reconstruction_orders:
        recon_largest = reconstruct_3d(
            largest_moments,
            Kx_recon,
            Ky_recon,
            Kz_recon,
            order,
        )
        recon_largest_binary = (recon_largest > 0.5).astype(np.float64)

        eps = normalized_error(centered_largest, recon_largest_binary)
        dice = dice_score(centered_largest, recon_largest_binary)
        hd95 = hd95_volume(centered_largest, recon_largest_binary)
        errors = error_analysis_3d(centered_largest, recon_largest_binary)

        reconstructions.append(recon_largest_binary)
        metrics.append((eps, dice, hd95, errors))

        print(f"Order {order}: epsilon = {eps:.6f}")
        print(f"Order {order}: DICE = {dice:.4f}")
        print(f"Order {order}: HD95 = {hd95:.4f}")
        print(f"Order {order}: IoU = {errors['iou']:.4f}")
        print(f"Order {order}: Precision = {errors['precision']:.4f}")
        print(f"Order {order}: Recall = {errors['recall']:.4f}")
        print(f"Order {order}: TP/FP/FN = {errors['tp']}/{errors['fp']}/{errors['fn']}")
        print(f"Order {order}: recon range = {recon_largest.min():.4f} .. {recon_largest.max():.4f}")

    plot_middle_slices(centered_largest, reconstructions, reconstruction_orders, metrics)
    highest_tested_order = reconstruction_orders[-1]
    largest_label_text = f"Muscle {largest_label}"
    plot_3d_reconstruction(
        centered_largest,
        reconstructions[-1],
        highest_tested_order,
        largest_label_text,
    )

if __name__ == "__main__":
    main()
