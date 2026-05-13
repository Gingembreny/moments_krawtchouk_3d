from __future__ import annotations

import numpy as np
from scipy import ndimage


def binariser(volume: np.ndarray, seuil: float = 0.5) -> np.ndarray:
    return (volume >= seuil).astype(np.float32)


def translation_3d(volume: np.ndarray, decalage: tuple[int, int, int]) -> np.ndarray:
    """Translation 3D sans interpolation, avec remplissage par zéro."""
    transforme = ndimage.shift(
        volume,
        shift=decalage,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return binariser(transforme)


def rotation_3d(volume: np.ndarray, angle_x: float = 0, angle_y: float = 0, angle_z: float = 0) -> np.ndarray:
    """
    Rotation autour des trois axes.

    reshape=False garde la taille 128×128×128 comme dans l'article.
    order=0 conserve un volume binaire sans interpolation floue.
    """
    v = volume
    if angle_x != 0:
        v = ndimage.rotate(v, angle=angle_x, axes=(1, 2), reshape=False, order=0, mode="constant", cval=0.0)
    if angle_y != 0:
        v = ndimage.rotate(v, angle=angle_y, axes=(0, 2), reshape=False, order=0, mode="constant", cval=0.0)
    if angle_z != 0:
        v = ndimage.rotate(v, angle=angle_z, axes=(0, 1), reshape=False, order=0, mode="constant", cval=0.0)
    return binariser(v)


def mise_a_echelle_3d(volume: np.ndarray, facteur: float) -> np.ndarray:
    """
    Changement d'échelle autour du centre du cube.
    Le résultat est recadré ou complété par des zéros pour revenir à la taille initiale.
    """
    taille = volume.shape[0]
    zoome = ndimage.zoom(volume, zoom=facteur, order=0)
    resultat = np.zeros_like(volume, dtype=np.float32)

    # Recadrage ou padding centré
    min_shape = np.minimum(zoome.shape, volume.shape)

    start_src = [(zoome.shape[i] - min_shape[i]) // 2 for i in range(3)]
    end_src = [start_src[i] + min_shape[i] for i in range(3)]

    start_dst = [(taille - min_shape[i]) // 2 for i in range(3)]
    end_dst = [start_dst[i] + min_shape[i] for i in range(3)]

    resultat[
        start_dst[0]:end_dst[0],
        start_dst[1]:end_dst[1],
        start_dst[2]:end_dst[2],
    ] = zoome[
        start_src[0]:end_src[0],
        start_src[1]:end_src[1],
        start_src[2]:end_src[2],
    ]
    return binariser(resultat)


def transformation_mixte(volume: np.ndarray, translation: tuple[int, int, int], angle: float, facteur: float) -> np.ndarray:
    """Transformation mixte utilisée pour D2 : échelle + rotation + translation."""
    v = mise_a_echelle_3d(volume, facteur)
    v = rotation_3d(v, angle_x=angle, angle_y=angle, angle_z=angle)
    v = translation_3d(v, translation)
    return binariser(v)
