from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def sauvegarder_courbe_accuracy(
    df: pd.DataFrame,
    chemin: str | Path,
    titre: str,
    afficher_article: bool = False,
) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(df["ordre"], df["accuracy_obtenue"], marker="o", label="Résultats obtenus")
    if afficher_article and "accuracy_article" in df.columns:
        plt.plot(df["ordre"], df["accuracy_article"], marker="s", linestyle="--", label="Référence article")
    plt.xlabel("Ordre des moments de Krawtchouk")
    plt.ylabel("Accuracy (%)")
    plt.title(titre)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chemin, dpi=200)
    plt.close()


def sauvegarder_matrice_confusion(cm: np.ndarray, classes: list[str], chemin: str | Path, titre: str) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    ax.set_title(titre)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    plt.tight_layout()
    plt.savefig(chemin, dpi=200)
    plt.close(fig)
