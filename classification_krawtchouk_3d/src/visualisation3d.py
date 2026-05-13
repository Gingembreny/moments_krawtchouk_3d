from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def projection_max(volume: np.ndarray, axe: int = 2) -> np.ndarray:
    return np.max(volume, axis=axe)


def dessiner_surface_3d(ax, volume: np.ndarray, titre: str) -> None:
    """
    Affiche un volume binaire sous forme de surface 3D avec marching cubes.
    Si la surface ne peut pas être extraite, on affiche un nuage de points léger.
    """
    v = (volume > 0.5).astype(np.float32)
    ax.set_title(titre, fontsize=9)
    ax.set_xlim(0, v.shape[0])
    ax.set_ylim(0, v.shape[1])
    ax.set_zlim(0, v.shape[2])
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect((1, 1, 1))

    if v.sum() < 10:
        return

    try:
        verts, faces, _, _ = measure.marching_cubes(v, level=0.5)
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            linewidth=0.1, antialiased=True, alpha=0.9
        )
    except Exception:
        coords = np.argwhere(v > 0.5)
        if len(coords) > 5000:
            rng = np.random.default_rng(0)
            coords = coords[rng.choice(len(coords), size=5000, replace=False)]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, alpha=0.5)


def sauvegarder_grille_reconstruction_3d(volumes: list[np.ndarray], titres: list[str], chemin: str | Path) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.2 * len(volumes), 3.6))
    for i, (volume, titre) in enumerate(zip(volumes, titres), start=1):
        ax = fig.add_subplot(1, len(volumes), i, projection="3d")
        dessiner_surface_3d(ax, volume, titre)
    fig.suptitle("Reconstruction 3D par moments de Tchebichef", fontsize=14)
    plt.tight_layout()
    plt.savefig(chemin, dpi=200)
    plt.close(fig)


def sauvegarder_grille_projection(volumes: list[np.ndarray], titres: list[str], chemin: str | Path) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(volumes), figsize=(3 * len(volumes), 3.2))
    if len(volumes) == 1:
        axes = [axes]
    for ax, volume, titre in zip(axes, volumes, titres):
        ax.imshow(projection_max(volume, axe=2).T, origin="lower", cmap="gray")
        ax.set_title(titre, fontsize=9)
        ax.set_xlabel("Coordonnée x")
        ax.set_ylabel("Coordonnée y")
    plt.tight_layout()
    plt.savefig(chemin, dpi=200)
    plt.close(fig)
