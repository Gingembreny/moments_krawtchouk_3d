from __future__ import annotations

import numpy as np

from .krawtchouk import precompute_K


def calculer_moments_3d(volume: np.ndarray, ordre: int, p=0.5) -> np.ndarray:
    """
    Calcule les moments 3D de Tchebichef d'un volume.

    Formule : M[p,q,r] = somme_x,y,z f(x,y,z) E[p,x] E[q,y] E[r,z]

    Le calcul est fait par produits tensoriels successifs pour éviter une triple boucle lente.
    """
    taille = volume.shape[0]
    
    Kx = precompute_K(ordre, taille, p)
    Ky = precompute_K(ordre, taille, p)
    Kz = precompute_K(ordre, taille, p)

    # Projection selon x, puis y, puis z.
    moments = np.einsum(
        "xyz,nx,my,lz->nml",
        volume,
        Kx,
        Ky,
        Kz,
        optimize=True)
    return moments.astype(np.float32)


def reconstruire_volume_3d(moments: np.ndarray, taille: int = 128, p=0.5) -> np.ndarray:
    """
    Reconstruit un volume à partir des moments de Tchebichef.
    """
    ordre = moments.shape[0]

    Kx = precompute_K(ordre, taille, p)
    Ky = precompute_K(ordre, taille, p)
    Kz = precompute_K(ordre, taille, p)
    volume = np.einsum(
        "nml,nx,my,lz->xyz",
        moments,
        Kx,
        Ky,
        Kz,
        optimize=True,
    )
    return volume.astype(np.float32)


def moments_en_vecteur(moments: np.ndarray) -> np.ndarray:
    return moments.reshape(-1).astype(np.float32)


def mse(original: np.ndarray, reconstruit: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float32) - reconstruit.astype(np.float32)) ** 2))


def seuillage_par_volume_original(reconstruit: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Binarise la reconstruction en conservant autant de voxels que dans l'objet original.
    C'est plus adapté pour visualiser une forme 3D qu'un seuil fixe arbitraire.
    """
    nb_voxels = int(np.sum(original > 0.5))
    if nb_voxels <= 0:
        return (reconstruit > 0.5).astype(np.float32)
    flat = reconstruit.reshape(-1)
    nb_voxels = min(nb_voxels, flat.size)
    seuil = np.partition(flat, -nb_voxels)[-nb_voxels]
    return (reconstruit >= seuil).astype(np.float32)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5
    b = b > 0.5
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)
