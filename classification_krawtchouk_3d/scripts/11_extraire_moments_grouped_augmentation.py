import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

from _bootstrap import RACINE
from src.io_volumes import lire_volume_im
from src.krawtchouk import precompute_K
from src.transformations import mise_a_echelle_3d, rotation_3d
from src.utils import charger_config, creer_dossier, lister_fichiers_im, nom_ordre


VARIANTES = [
    (1.00, 0, 0, 0),
    (0.90, 0, 0, 0),
    (1.10, 0, 0, 0),
    (1.00, 20, 0, 0),
    (1.00, -20, 0, 0),
    (1.00, 0, 20, 0),
    (1.00, 0, -20, 0),
    (1.00, 0, 0, 20),
    (1.00, 0, 0, -20),
    (0.95, 20, 20, 0),
    (0.95, 20, 0, 20),
    (0.95, 0, 20, 20),
    (1.05, -20, -20, 0),
    (1.05, -20, 0, -20),
    (1.05, 0, -20, -20),
    (1.00, 20, 20, 20),
]


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


def transformer_volume(volume: np.ndarray, facteur: float, ax: float, ay: float, az: float) -> np.ndarray:
    v = mise_a_echelle_3d(volume, facteur)
    v = rotation_3d(v, angle_x=ax, angle_y=ay, angle_z=az)
    return v.astype(np.float32)


def calculer_moments_precomputed(volume: np.ndarray, Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray) -> np.ndarray:
    return np.einsum("xyz,nx,my,lz->nml", volume, Kx, Ky, Kz, optimize=True).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-set",
        choices=["D1", "D2"],
        default="D1",
        help="Classes à utiliser. D1 correspond aux 10 classes du protocole principal.",
    )
    parser.add_argument(
        "--max-objets-par-classe",
        type=int,
        default=None,
        help="Option de test rapide : limite le nombre d'objets originaux par classe",
    )
    parser.add_argument("--force", action="store_true", help="Recalcule même si les fichiers existent déjà")
    args = parser.parse_args()

    config = charger_config()
    taille = int(config["taille_volume"])
    padding = int(config.get("padding_generation", 0))
    ordres = [int(o) for o in config["ordres_grouped_augmentation"]]
    max_ordre = max(ordres)
    classes = list(config[f"classes_{args.class_set}"])
    source = RACINE / config["dossier_dataset_original"]
    dossier_sortie = creer_dossier(
        RACINE / config["dossier_resultats"] / "features_grouped_augmentation" / args.class_set
    )

    taille_finale = taille + 2 * padding
    nb_objets_attendu = 0
    for classe in classes:
        fichiers = lister_fichiers_im(source / classe)
        if args.max_objets_par_classe is not None:
            fichiers = fichiers[: args.max_objets_par_classe]
        nb_objets_attendu += len(fichiers)
    nb_echantillons_attendu = nb_objets_attendu * len(VARIANTES)

    if not args.force:
        features_completes = True
        for ordre in ordres:
            chemin_X = dossier_sortie / nom_ordre(ordre) / "X.npy"
            chemin_groups = dossier_sortie / nom_ordre(ordre) / "groups.npy"
            if not chemin_X.exists() or not chemin_groups.exists():
                features_completes = False
                break
            X_existant = np.load(chemin_X, mmap_mode="r")
            if X_existant.shape[0] != nb_echantillons_attendu:
                features_completes = False
                break

        if features_completes:
            print("Toutes les features existent déjà avec le bon nombre d'échantillons.")
            print("Utilise --force pour recalculer.")
            return
        print("Features absentes ou incomplètes : recalcul.")

    print("\nExtraction moments avec augmentation groupée par objet")
    print("Class set :", args.class_set)
    print("Ordres :", ordres)
    print("Transformations par objet :", len(VARIANTES))
    print("Taille utilisée :", f"{taille_finale}^3")

    print("Pré-calcul des bases de Krawtchouk...")
    Kx = precompute_K(max_ordre, taille_finale, 0.5)
    Ky = precompute_K(max_ordre, taille_finale, 0.5)
    Kz = precompute_K(max_ordre, taille_finale, 0.5)

    X_par_ordre = {ordre: [] for ordre in ordres}
    y = []
    groups = []
    metas = []
    classes_effectives = []
    group_id = 0

    for classe in classes:
        fichiers = lister_fichiers_im(source / classe)
        if args.max_objets_par_classe is not None:
            fichiers = fichiers[: args.max_objets_par_classe]
        if not fichiers:
            print(f"Classe ignorée car absente ou vide : {classe}")
            continue

        label = len(classes_effectives)
        classes_effectives.append(classe)

        for chemin in tqdm(fichiers, desc=f"{classe}"):
            object_key = f"{classe}/{chemin.stem}"
            volume = lire_volume_im(chemin, taille=taille)
            volume = centrer_et_padder(volume, padding=padding)

            for variant_id, (facteur, ax, ay, az) in enumerate(VARIANTES):
                volume_aug = transformer_volume(volume, facteur, ax, ay, az)
                moments_max = calculer_moments_precomputed(volume_aug, Kx, Ky, Kz)

                for ordre in ordres:
                    X_par_ordre[ordre].append(moments_max[:ordre, :ordre, :ordre].reshape(-1))

                y.append(label)
                groups.append(group_id)
                metas.append(
                    {
                        "chemin": str(chemin.relative_to(RACINE)),
                        "classe": classe,
                        "label": label,
                        "object_id": chemin.stem,
                        "object_key": object_key,
                        "group_id": group_id,
                        "variant_id": variant_id,
                        "scale": facteur,
                        "angle_x": ax,
                        "angle_y": ay,
                        "angle_z": az,
                    }
                )

            group_id += 1

    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)
    meta_df = pd.DataFrame(metas)

    pd.DataFrame({"classe_id": range(len(classes_effectives)), "classe": classes_effectives}).to_csv(
        dossier_sortie / "classes.csv",
        index=False,
    )
    meta_df.to_csv(dossier_sortie / "meta_all.csv", index=False)

    print("Nombre de classes :", len(classes_effectives))
    print("Nombre d'objets originaux :", len(np.unique(groups)))
    print("Nombre d'échantillons augmentés :", len(y))

    for ordre in ordres:
        dossier_ordre = creer_dossier(dossier_sortie / nom_ordre(ordre))
        X = np.vstack(X_par_ordre[ordre]).astype(np.float32)
        np.save(dossier_ordre / "X.npy", X)
        np.save(dossier_ordre / "y.npy", y)
        np.save(dossier_ordre / "groups.npy", groups)
        meta_df.to_csv(dossier_ordre / "meta.csv", index=False)
        print(f"Ordre {ordre}: X shape={X.shape}")

    print("\nExtraction terminée :", dossier_sortie)


if __name__ == "__main__":
    main()
