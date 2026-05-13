import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from scipy.spatial.distance import cdist
from scipy.special import loggamma
from fractions import Fraction
from sklearn.decomposition import PCA
import mplcursors

def is_image_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_image(path):
    if is_image_file(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image file: {path}")
        return img.astype(np.float32)

    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]
    return arr.astype(np.float32)


def load_images_from_folder(folder_path):
    files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if is_image_file(f)
    ])
    if not files:
        raise ValueError(f"No supported image files found in {folder_path}")
    return files


def preprocess(img):
    img = img.astype(np.float32)

    # normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    # Binarize the image (threshold at 0.5)
    img = (img > 0.5).astype(np.float32)

    return img

def pochhammer(a, n):
    if n == 0:
        return Fraction(1, 1)
    result = Fraction(1, 1)
    for i in range(n):
        result *= (a + i)
    return result


def krawtchouk_poly(n, x, N, mu):
    """Use the paper's 2F1 definition with a terminating finite sum."""
    a = Fraction(-n, 1)
    b = Fraction(-x, 1)
    c = Fraction(-(N - 1), 1)
    z = Fraction(1, 1) / Fraction(mu)

    value = Fraction(0, 1)
    for k in range(n + 1):
        num = pochhammer(a, k) * pochhammer(b, k)
        den = pochhammer(c, k) * math.factorial(k)
        value += num / den * (z ** k)

    return float(value)


def rho(n, N, mu):
    if n == 0:
        return 1.0

    # Stable paper formula using loggamma to avoid factorial overflow:
    # ρK(n) = ((1-µ)/µ)^n * n! * (N-1-n)! / (N-1)!
    log_rho = n * math.log((1 - mu) / mu) + loggamma(n + 1) + loggamma(N - n) - loggamma(N)
    return math.exp(log_rho)


def krawtchouk_normalized(n, x, N, mu):
    K = krawtchouk_poly(n, x, N, mu)
    log_w = (
        loggamma(N)
        - loggamma(x + 1)
        - loggamma(N - x)
        + x * math.log(mu)
        + (N - 1 - x) * math.log(1 - mu)
    )
    w = math.exp(log_w)
    r = rho(n, N, mu)

    return K * math.sqrt(w / r)

def precompute_K(max_order, N, mu):
    if max_order > N:
        raise ValueError(
            f"max_order ({max_order}) must be <= image dimension N ({N}) "
            "for Krawtchouk polynomials (valid n = 0..N-1)."
        )
    K_table = np.zeros((max_order, N))
    for n in range(max_order):
        for x in range(N):
            K_table[n, x] = krawtchouk_normalized(n, x, N, mu)
    return K_table

def compute_moments(img, Kx, Ky, order):
    Kx_order = Kx[:order, :]
    Ky_order = Ky[:order, :]
    return Kx_order @ img @ Ky_order.T

def reconstruct(moments, Kx, Ky, order):
    Kx_order = Kx[:order, :]
    Ky_order = Ky[:order, :]
    return Kx_order.T @ moments[:order, :order] @ Ky_order

def normalized_error(img, recon):
    numerator = np.sum((img - recon) ** 2)
    denominator = np.sum(img ** 2)
    return numerator / denominator

def compute_dice(img1, img2):
    """Compute Dice coefficient for binary images."""
    img1_sitk = sitk.GetImageFromArray(img1.astype(np.uint8))
    img2_sitk = sitk.GetImageFromArray(img2.astype(np.uint8))
    dice_filter = sitk.LabelOverlapMeasuresImageFilter()
    dice_filter.Execute(img1_sitk, img2_sitk)
    return dice_filter.GetDiceCoefficient()

def compute_hd95(img1, img2):
    """Compute 95th percentile Hausdorff distance for binary images."""
    # Get surface points
    surface1 = np.array(np.where(img1 > 0)).T
    surface2 = np.array(np.where(img2 > 0)).T
    
    if len(surface1) == 0 or len(surface2) == 0:
        return np.inf
    
    # Compute all pairwise distances
    distances = cdist(surface1, surface2)
    
    # Hausdorff distance (max of min distances)
    hd1 = np.max(np.min(distances, axis=1))
    hd2 = np.max(np.min(distances, axis=0))
    hd = max(hd1, hd2)
    
    # For HD95, we need 95th percentile of the directed distances
    directed_distances = np.concatenate([np.min(distances, axis=1), np.min(distances, axis=0)])
    hd95 = np.percentile(directed_distances, 95)
    
    return hd95

def compute_assd(img1, img2):
    """Compute Average Surface Distance (ASSD) for binary images."""
    img1_sitk = sitk.GetImageFromArray(img1.astype(np.uint8))
    img2_sitk = sitk.GetImageFromArray(img2.astype(np.uint8))
    assd_filter = sitk.HausdorffDistanceImageFilter()
    assd_filter.Execute(img1_sitk, img2_sitk)
    # Note: SimpleITK's HausdorffDistanceImageFilter actually computes average surface distance
    # when used with binary images
    return assd_filter.GetAverageHausdorffDistance()

def moment_distance(M1, M2):
    # Use relative L2 distance, avoiding division by zero
    diff = M1 - M2
    norm_M1 = np.linalg.norm(M1)
    if norm_M1 == 0:
        return np.linalg.norm(diff)
    return np.linalg.norm(diff) / norm_M1

def symmetric_moment_distance(M1, M2):
    return 0.5 * (moment_distance(M1, M2) + moment_distance(M2, M1))

def center_image(img):
    coords = np.argwhere(img > 0.1)  # Use threshold to avoid noise
    if len(coords) == 0:
        return img
    center = coords.mean(axis=0)
    shift = np.array(img.shape)//2 - center
    return np.roll(img, shift.astype(int), axis=(0,1))

def compute_central_moments(img, Kx, Ky, order):
    # APPROXIMATION: Center the image first, then compute regular moments
    # This is NOT mathematically equivalent to true central moments
    centered_img = center_image(img)
    return compute_moments(centered_img, Kx, Ky, order)

def align_orientation(img):
    coords = np.column_stack(np.where(img > 0.5))
    
    # PCA
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # find the direction in which the characteristic value is maximum
    main_axis = eigvecs[:, np.argmax(eigvals)]
    if main_axis[1] < 0:
        main_axis = -main_axis
    
    angle = np.arctan2(main_axis[0], main_axis[1])
    
    # rotate the image
    angle_deg = np.degrees(angle)
    M = cv2.getRotationMatrix2D(tuple(mean[::-1]), angle_deg, 1.0)
    aligned = cv2.warpAffine(img, M, img.shape[::-1], flags = cv2.INTER_NEAREST, borderValue = 0)
    
    return aligned

def compute_PCA_moments(img, Kx, Ky, order):
    img_centered = center_image(img)
    aligned_img = align_orientation(img_centered)
    return compute_moments(aligned_img, Kx, Ky, order)

def mse_global(Q_list, Qp_list):
    N = len(Q_list)
    mse_total = 0.0
    
    for i in range(N):
        Q = Q_list[i].flatten()
        Qp = Qp_list[i].flatten()
        mse = np.mean((Q - Qp)**2)
        mse_total += mse
        
    return mse_total / N

def pad_image(img, pad):
    img_pad = np.pad(img, ((pad,pad),(pad,pad)), mode='constant')
    return img_pad


# Load image data from PNG folder
image_source = "/Users/jineiya/Desktop/Projet_PIR/selection_png"
image_files = load_images_from_folder(image_source)
print(f"Found {len(image_files)} image files in {image_source}")
image_path = "/Users/jineiya/Desktop/Projet_PIR/selection_png/apple-1.png"
print(f"Using image: {image_path}")
img = load_image(image_path)
img = preprocess(img)

pad = img.shape[0] // 3
img_pad = pad_image(img, pad)
img_used = img_pad

print(f"Image shape: {img.shape}")
print(f"Image value range: {img.min()} - {img.max()}")
print(f"Foreground ratio: {np.mean(img > 0.5):.3f}")

mu = 0.5
data_range = 1.0
# Adjust orders based on image size to balance quality and computation time
if img.shape[0] > 200:
    orders = [1, 5, 10, 20, 30, 40, 50]  # Large images: smaller orders
elif img.shape[0] > 100:
    orders = [5, 10, 15, 20]
else:
    orders = [5, 10, 20, 30, 40]
epsilon_list = []
coef_list = []
Nx, Ny = img_used.shape
max_order = max(orders)
if max_order > min(Nx, Ny):
    max_order = min(Nx, Ny) - 1
    orders = [o for o in orders if o <= max_order]
    print(f"Adjusted orders to {orders} (max_order capped at {max_order})")
    
if not orders:
    raise ValueError(f"No valid orders for image dimension {img.shape}")
    
Kx = precompute_K(max_order, Nx, mu)
Ky = precompute_K(max_order, Ny, mu)

recons = []

for order in orders:
    moments = compute_moments(img_used, Kx, Ky, order)
    recon_pad = reconstruct(moments, Kx, Ky, order)
    recon = recon_pad[pad:-pad, pad:-pad]
    # Binarize the reconstructed image
    recon_binary = (recon > 0.5).astype(np.float32)
    recons.append(recon_binary)
    eps = normalized_error(img, recon)
    epsilon_list.append(eps)
    dice = compute_dice(img, recon_binary)
    hd95 = compute_hd95(img, recon_binary)
    assd = compute_assd(img, recon_binary)
    coef_list.append((dice, hd95, assd))

    print(f"Order {order}: epsilon = {eps:.6f}")
    print(f"Order {order}: DICE = {dice:.4f}")
    print(f"Order {order}: HD95 = {hd95:.4f}")
    print(f"Order {order}: ASSD = {assd:.4f}")
    
num_cols = len(orders) + 1
fig, axes = plt.subplots(1, num_cols, figsize=(2.4 * num_cols, 4.2))

# original
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original", fontsize=10, pad=8)
axes[0].axis('off')

# reconstruction
for i, recon in enumerate(recons):
    ax = axes[i + 1]
    ax.imshow(recon, cmap='gray') 
    eps = epsilon_list[i]
    dice, hd95, assd = coef_list[i]
    ax.set_title(
        f"Order {orders[i]}\n"
        f"eps={eps:.3f}\n"
        f"DICE={dice:.3f}\n"
        f"HD95={hd95:.2f}\n"
        f"ASSD={assd:.2f}",
        fontsize=9,
        pad=8,
        linespacing=1.15,
    )
    ax.axis('off')

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.72, wspace=0.35)
plt.show()

plt.plot(orders, epsilon_list)
plt.xlabel("Order")
plt.ylabel("Reconstruction error")
plt.title("Error vs order")
plt.show()

# Test with regular moments (using only reference-based, not centered/rotated)
order = min(40, max_order)  # Use safe order
M0 = compute_moments(img_used, Kx, Ky, order)

# Simple transforms on original image
translation = np.roll(img, 5, axis=0)
translation_used = pad_image(translation, pad)
M1 = compute_moments(translation_used, Kx, Ky, order)
dt = moment_distance(M0, M1)

rotation = np.rot90(img)
rotation_used = pad_image(rotation, pad)
M2 = compute_moments(rotation_used, Kx, Ky, order)
dr = moment_distance(M0, M2)

scaling = cv2.resize(img, (int(img.shape[1]*0.8), int(img.shape[0]*0.8)))
scaling = cv2.resize(scaling, img.shape[::-1])
scaling_used = pad_image(scaling, pad)

M3 = compute_moments(scaling_used, Kx, Ky, order)
ds = moment_distance(M0, M3)

print(f"Rotation distance: {dr:.6f}")
print(f"Scaling distance: {ds:.6f}")

# MSE global of transformation
Q_list = []
Qp_list = []

for path in image_files[:20]:  
    img = preprocess(load_image(path))
    img_used = pad_image(img, pad)
    
    moments = compute_moments(img_used, Kx, Ky, order)
    Q_list.append(moments)
    
    # transformation(rotation）
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 90, 1)
    img_rot = cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0)
    img_rot_used = pad_image(img_rot, pad)
    moments_rot = compute_moments(img_rot_used, Kx, Ky, order)
    Qp_list.append(moments_rot)

mse_g = mse_global(Q_list, Qp_list)
print("MSE global (rotation):", mse_g)

# invariance curve depending on rotation angle
angles = [0, 30, 60, 90, 120, 150]
mse_list = []

for angle in angles:
    Q_list = []
    Qp_list = []

    for path in image_files[:20]:
        img = preprocess(load_image(path))
        img_used = pad_image(img, pad)
        
        moments = compute_moments(img_used, Kx, Ky, order)
        Q_list.append(moments)

        # rotation
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img_rot = cv2.warpAffine(img, M, img.shape[::-1])
        img_rot_used = pad_image(img_rot, pad)

        moments_rot = compute_moments(img_rot_used, Kx, Ky, order)
        Qp_list.append(moments_rot)

    mse_list.append(mse_global(Q_list, Qp_list))

plt.plot(angles, mse_list)
plt.xlabel("Rotation angle")
plt.ylabel("MSE global")
plt.title("Invariance curve")
plt.show()

# invariance curve depending on translation shift
shifts = [0, 2, 4, 6, 8, 10]
mse_list_t = []

for shift in shifts:
    Q_list = []
    Qp_list = []

    for path in image_files[:20]:
        img = preprocess(load_image(path))
        img_used = pad_image(img, pad)
        
        moments = compute_moments(img_used, Kx, Ky, order)
        Q_list.append(moments)

        # translation (x and y direction)
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        img_trans = cv2.warpAffine(
            img, M, img.shape[::-1],
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )
        img_trans_used = pad_image(img_trans, pad)

        moments_trans = compute_moments(img_trans_used, Kx, Ky, order)
        Qp_list.append(moments_trans)

    mse_list_t.append(mse_global(Q_list, Qp_list))

plt.plot(shifts, mse_list_t)
plt.xlabel("Translation shift (pixels)")
plt.ylabel("MSE global")
plt.title("Translation invariance curve")
plt.show()

# PCA feature distribution
features = []
labels = []

for path in image_files[:50]:
    img = preprocess(load_image(path))
    img_used = pad_image(img, pad)
        
    moments = compute_moments(img_used, Kx, Ky, order)
    features.append(moments.flatten())
    
    # label with filename
    labels.append(os.path.basename(path))
    
features = np.array(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

plt.figure()
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=range(len(labels)), cmap='viridis')

# Add hover functionality
cursor = mplcursors.cursor(scatter, hover=True)
@cursor.connect("add")
def on_add(sel):
    index = sel.index
    sel.annotation.set_text(labels[index])

plt.title("PCA of Krawtchouk moments")
plt.show()

# distance matrix
N = len(features)
D = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        D[i, j] = symmetric_moment_distance(features[i], features[j])

plt.figure()
im = plt.imshow(D, cmap='hot')
plt.colorbar()
plt.title("Distance matrix")

# Add hover functionality for distance matrix
cursor = mplcursors.cursor(im, hover=True)
@cursor.connect("add")
def on_add(sel):
    i, j = sel.index
    row_file = labels[i]
    col_file = labels[j]
    distance = D[i, j]
    sel.annotation.set_text(f"Row: {row_file}\nCol: {col_file}\nDistance: {distance:.4f}")

plt.show()
