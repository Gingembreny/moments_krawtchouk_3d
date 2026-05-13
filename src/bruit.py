from __future__ import annotations

import numpy as np


def ajouter_salt_pepper(volume: np.ndarray, densite: float, seed: int | None = None) -> np.ndarray:
    """
    Bruit Salt & Pepper : une proportion de voxels est forcée à 0 ou à 1.
    Dans l'article : 1%, 3%, 5%, 10%, 20%.
    """
    rng = np.random.default_rng(seed)
    noisy = volume.copy().astype(np.float32)
    nb_voxels = noisy.size
    nb_bruites = int(densite * nb_voxels)
    indices = rng.choice(nb_voxels, size=nb_bruites, replace=False)
    moitie = nb_bruites // 2
    flat = noisy.reshape(-1)
    flat[indices[:moitie]] = 1.0
    flat[indices[moitie:]] = 0.0
    return noisy


def ajouter_speckle(volume: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """
    Bruit Speckle multiplicatif : V_bruite = V + V * N(0, sigma).
    Dans l'article : moyenne 0 et sigma = 0.3, 0.5, 0.7.
    """
    rng = np.random.default_rng(seed)
    bruit = rng.normal(loc=0.0, scale=sigma, size=volume.shape)
    noisy = volume + volume * bruit
    return np.clip(noisy, 0, 1).astype(np.float32)
