import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

from _bootstrap import RACINE
from src.io_volumes import lire_volume_im
from src.moments3d_krawtchouk import calculer_moments_3d, moments_en_vecteur
from src.utils import charger_config, creer_dossier, lister_fichiers_im, nom_ordre


def classes_pour_experience(config: dict, nom: str) -> list[str]:
    if nom == "D1":
        return list(config["classes_D1"])
    if nom == "D2":
        return list(config["classes_D2"])
    if nom == "object_cv":
        return list(config["classes_object_cv"])
    if nom == "all":
        dossier = RACINE / config["dossier_dataset_original"]
        return sorted([p.name for p in dossier.iterdir() if p.is_dir()])
    raise ValueError(f"Unknown class set: {nom}")


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
        volume = np.pad(
            volume,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

    return volume.astype(np.float32)


def charger_objets_originaux(config: dict, class_set: str, max_objets_par_classe: int | None):
    taille = int(config["taille_volume"])
    padding = int(config.get("padding_generation", 0))
    source = RACINE / config["dossier_dataset_original"]
    classes = classes_pour_experience(config, class_set)

    volumes = []
    labels = []
    metas = []
    classes_effectives = []

    for classe in classes:
        fichiers = lister_fichiers_im(source / classe)
        if max_objets_par_classe is not None:
            fichiers = fichiers[:max_objets_par_classe]
        if not fichiers:
            print(f"Classe ignorée car absente ou vide : {classe}")
            continue

        label = len(classes_effectives)
        classes_effectives.append(classe)
        for chemin in tqdm(fichiers, desc=f"Lecture {classe}"):
            volume = lire_volume_im(chemin, taille=taille)
            volume = centrer_et_padder(volume, padding=padding)
            volumes.append(volume)
            labels.append(label)
            metas.append(
                {
                    "chemin": str(chemin.relative_to(RACINE)),
                    "classe": classe,
                    "label": label,
                    "object_id": chemin.stem,
                    "shape": "x".join(str(v) for v in volume.shape),
                }
            )

    return classes_effectives, volumes, np.array(labels, dtype=np.int64), metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-set",
        choices=["object_cv", "D1", "D2", "all"],
        default="D1",
        help="Classes à utiliser pour l'expérience object-level CV. D1 = les 10 classes du protocole principal.",
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
    ordres = [int(o) for o in config["ordres_object_cv"]]
    dossier_sortie = creer_dossier(
        RACINE / config["dossier_resultats"] / "features_object_cv" / args.class_set
    )

    print("\nExtraction object-level Krawtchouk moments")
    print("Class set :", args.class_set)
    print("Ordres :", ordres)
    print("Padding :", int(config.get("padding_generation", 0)))

    classes, volumes, y, metas = charger_objets_originaux(
        config,
        class_set=args.class_set,
        max_objets_par_classe=args.max_objets_par_classe,
    )

    if len(classes) < 2:
        raise ValueError("Il faut au moins deux classes non vides pour la classification.")

    pd.DataFrame({"classe_id": range(len(classes)), "classe": classes}).to_csv(
        dossier_sortie / "classes.csv",
        index=False,
    )
    pd.DataFrame(metas).to_csv(dossier_sortie / "objects.csv", index=False)

    print("Nombre de classes :", len(classes))
    print("Nombre d'objets :", len(volumes))
    if volumes:
        print("Shape utilisée :", volumes[0].shape)

    for ordre in ordres:
        dossier_ordre = creer_dossier(dossier_sortie / nom_ordre(ordre))
        chemin_X = dossier_ordre / "X.npy"
        chemin_y = dossier_ordre / "y.npy"
        chemin_meta = dossier_ordre / "meta.csv"

        if not args.force and chemin_X.exists() and chemin_y.exists():
            X_existant = np.load(chemin_X, mmap_mode="r")
            if X_existant.shape[0] == len(volumes):
                print(f"Ordre {ordre}: déjà extrait, passage au suivant.")
                continue
            print(f"Ordre {ordre}: extraction incomplète détectée, recalcul.")

        X = []
        print(f"\nOrdre {ordre} : dimension attendue = {ordre ** 3}")
        for volume in tqdm(volumes, desc=f"Moments ordre {ordre}"):
            moments = calculer_moments_3d(volume, ordre=ordre)
            X.append(moments_en_vecteur(moments))

        X = np.vstack(X).astype(np.float32)
        np.save(chemin_X, X)
        np.save(chemin_y, y)
        pd.DataFrame(metas).to_csv(chemin_meta, index=False)
        print(f"Sauvegardé : {chemin_X} shape={X.shape}")

    print("\nExtraction object-level terminée.")


if __name__ == "__main__":
    main()
