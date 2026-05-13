from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=64)
def base_tchebichef_orthonormale(taille: int, ordre: int) -> np.ndarray:
    """
    Calcule les polynômes discrets de Tchebichef pondérés et normalisés.

    Paramètres
    ----------
    taille : int
        Taille du volume selon un axe, typiquement 128.
    ordre : int
        Nombre de polynômes utilisés. ordre=20 donne 20×20×20 = 8000 moments,
        ce qui correspond à la notation de l'article.

    Retour
    ------
    E : ndarray, shape (ordre, taille)
        E[p, x] est le polynôme pondéré d'ordre p évalué en x.
        Les lignes sont normalisées pour vérifier E @ E.T ≈ I.
    """
    if ordre < 1 or ordre > taille:
        raise ValueError(f"ordre doit être compris entre 1 et {taille}")

    x = np.arange(taille, dtype=np.float64)
    t = np.zeros((ordre, taille), dtype=np.float64)

    # Récurrence discrète de Tchebichef. On normalise ensuite chaque ligne.
    t[0, :] = 1.0
    if ordre > 1:
        t[1, :] = (2.0 * x + 1.0 - taille) / taille

    for n in range(2, ordre):
        coefficient = (n - 1.0) * (1.0 - ((n - 1.0) ** 2) / (taille ** 2))
        t[n, :] = ((2.0 * n - 1.0) * t[1, :] * t[n - 1, :] - coefficient * t[n - 2, :]) / n

    normes = np.sqrt(np.sum(t ** 2, axis=1, keepdims=True))
    E = t / np.maximum(normes, 1e-12)
    return E.astype(np.float32)


def erreur_orthogonalite(taille: int, ordre: int) -> float:
    """Renvoie ||E E^T - I||_F pour vérifier l'orthogonalité numérique."""
    E = base_tchebichef_orthonormale(taille, ordre).astype(np.float64)
    identite = np.eye(ordre)
    return float(np.linalg.norm(E @ E.T - identite))
