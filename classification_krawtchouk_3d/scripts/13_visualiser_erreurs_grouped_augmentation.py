import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from _bootstrap import RACINE
from src.io_volumes import lire_volume_im
from src.transformations import mise_a_echelle_3d, rotation_3d
from src.utils import charger_config, creer_dossier


def centrer_et_padder(volume: np.ndarray, padding: int) -> np.ndarray:
    coords = np.argwhere(volume > 0)
    if len(coords) > 0:
        centre_masse = coords.mean(axis=0)
        centre_cible = np.array(volume.shape) / 2
        volume = ndi.shift(
            volume,
            shift=centre_cible - centre_masse,
            order=0,
            mode="constant",
            cval=0,
        )
    if padding > 0:
        volume = np.pad(volume, pad_width=padding, mode="constant", constant_values=0)
    return volume.astype(np.float32)


def charger_volume_augmente(row: pd.Series, taille: int, padding: int) -> np.ndarray:
    volume = lire_volume_im(RACINE / row["chemin"], taille=taille)
    volume = centrer_et_padder(volume, padding=padding)
    volume = mise_a_echelle_3d(volume, float(row["scale"]))
    volume = rotation_3d(
        volume,
        angle_x=float(row["angle_x"]),
        angle_y=float(row["angle_y"]),
        angle_z=float(row["angle_z"]),
    )
    return volume.astype(np.float32)


def dessiner_surface(ax, volume: np.ndarray, titre: str) -> None:
    v = (volume > 0.5).astype(np.float32)
    ax.set_title(titre, fontsize=8)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    ax.set_box_aspect((1, 1, 1))
    if v.sum() < 10:
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(v, level=0.5)
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            linewidth=0.05,
            antialiased=True,
            alpha=0.9,
            color="#7aa6c2",
        )
    except Exception:
        coords = np.argwhere(v > 0.5)
        if len(coords) > 5000:
            rng = np.random.default_rng(0)
            coords = coords[rng.choice(len(coords), size=5000, replace=False)]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1, alpha=0.5)


def projection(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.max(volume, axis=axis)


def sauvegarder_grille_echantillons(rows: pd.DataFrame, chemin: Path, titre: str, taille: int, padding: int) -> None:
    if rows.empty:
        return

    max_rows = min(len(rows), 8)
    rows = rows.head(max_rows).reset_index(drop=True)

    fig = plt.figure(figsize=(13, 2.8 * max_rows))
    fig.suptitle(titre, fontsize=14)

    for i, row in rows.iterrows():
        volume = charger_volume_augmente(row, taille=taille, padding=padding)
        label = (
            f"{row['classe_reelle']} -> {row['classe_predite']}\n"
            f"{row['object_key']} | var {int(row['variant_id'])}"
        )

        ax3d = fig.add_subplot(max_rows, 4, 4 * i + 1, projection="3d")
        dessiner_surface(ax3d, volume, label)

        vues = [
            ("XY", projection(volume, axis=2).T),
            ("XZ", projection(volume, axis=1).T),
            ("YZ", projection(volume, axis=0).T),
        ]
        for j, (nom_vue, img) in enumerate(vues, start=2):
            ax = fig.add_subplot(max_rows, 4, 4 * i + j)
            ax.imshow(img, origin="lower", cmap="gray")
            ax.set_title(nom_vue, fontsize=8)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chemin.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chemin, dpi=180)
    plt.close(fig)


def choisir_un_par_objet(df: pd.DataFrame, max_count: int) -> pd.DataFrame:
    return df.sort_values(["object_key", "variant_id"]).drop_duplicates("object_key").head(max_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-set", choices=["D1"], default="D1")
    parser.add_argument("--order", type=int, default=None, help="Ordre à visualiser. Par défaut: meilleur ordre.")
    args = parser.parse_args()

    config = charger_config()
    taille = int(config["taille_volume"])
    padding = int(config.get("padding_generation", 0))

    base_cls = RACINE / config["dossier_resultats"] / "grouped_augmentation_classification" / args.class_set
    base_feat = RACINE / config["dossier_resultats"] / "features_grouped_augmentation" / args.class_set
    sortie = creer_dossier(base_cls / "visual_examples")

    accuracy = pd.read_csv(base_cls / "accuracy_by_order_5fold.csv")
    ordre = args.order
    if ordre is None:
        ordre = int(accuracy.sort_values("accuracy_obtenue", ascending=False).iloc[0]["ordre"])

    classes = pd.read_csv(base_feat / "classes.csv")["classe"].tolist()
    predictions = pd.read_csv(base_cls / "predictions_by_sample_5fold.csv")
    predictions = predictions[predictions["ordre"] == ordre].copy()
    meta = pd.read_csv(base_feat / f"ordre_{ordre:03d}" / "meta.csv")
    meta_cols = ["chemin", "object_key", "variant_id", "scale", "angle_x", "angle_y", "angle_z"]
    data = predictions.merge(meta[meta_cols], on=["chemin", "object_key", "variant_id"], how="left")
    data["correct"] = data["label_reel"] == data["label_predit"]

    y_true = data["label_reel"].to_numpy()
    y_pred = data["label_predit"].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    resume = pd.DataFrame(
        {
            "classe": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values("recall", ascending=False)
    resume.to_csv(sortie / f"class_performance_order_{ordre:03d}.csv", index=False)

    top_classes = resume.head(3)["classe"].tolist()
    low_classes = resume.tail(3)["classe"].tolist()

    bons = []
    for classe in top_classes:
        bons.append(choisir_un_par_objet(data[(data["classe_reelle"] == classe) & data["correct"]], 2))
    bons_df = pd.concat(bons, ignore_index=True) if bons else pd.DataFrame()

    difficiles = []
    for classe in low_classes:
        erreurs = choisir_un_par_objet(data[(data["classe_reelle"] == classe) & ~data["correct"]], 2)
        if len(erreurs) == 0:
            erreurs = choisir_un_par_objet(data[data["classe_reelle"] == classe], 2)
        difficiles.append(erreurs)
    difficiles_df = pd.concat(difficiles, ignore_index=True) if difficiles else pd.DataFrame()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    confusions = []
    for i, vraie in enumerate(classes):
        for j, predite in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], vraie, predite))
    confusions = sorted(confusions, reverse=True)[:4]
    confusion_rows = []
    for _, vraie, predite in confusions:
        selection = choisir_un_par_objet(
            data[(data["classe_reelle"] == vraie) & (data["classe_predite"] == predite)],
            2,
        )
        confusion_rows.append(selection)
    confusions_df = pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame()

    dolphins_correct = choisir_un_par_objet(
        data[(data["classe_reelle"] == "dolphinsIm") & data["correct"]],
        4,
    )
    dolphins_wrong = choisir_un_par_objet(
        data[(data["classe_reelle"] == "dolphinsIm") & ~data["correct"]],
        4,
    )
    dolphins_df = pd.concat([dolphins_correct, dolphins_wrong], ignore_index=True)

    sauvegarder_grille_echantillons(
        bons_df,
        sortie / f"good_classes_order_{ordre:03d}.png",
        f"Classes bien reconnues — ordre {ordre}",
        taille,
        padding,
    )
    sauvegarder_grille_echantillons(
        difficiles_df,
        sortie / f"difficult_classes_order_{ordre:03d}.png",
        f"Classes difficiles — ordre {ordre}",
        taille,
        padding,
    )
    sauvegarder_grille_echantillons(
        confusions_df,
        sortie / f"top_confusions_order_{ordre:03d}.png",
        f"Confusions principales — ordre {ordre}",
        taille,
        padding,
    )
    sauvegarder_grille_echantillons(
        dolphins_df,
        sortie / f"dolphins_correct_vs_wrong_order_{ordre:03d}.png",
        f"Dolphins : corrects puis erreurs — ordre {ordre}",
        taille,
        padding,
    )

    print("Ordre visualisé :", ordre)
    print("Dossier de sortie :", sortie)
    print("Classes fortes :", top_classes)
    print("Classes difficiles :", low_classes)
    print("Confusions principales :", confusions)


if __name__ == "__main__":
    main()
