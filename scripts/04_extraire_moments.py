import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import RACINE
from src.io_volumes import lire_volume_npy
from src.moments3d_krawtchouk import calculer_moments_3d, moments_en_vecteur
from src.utils import charger_config, creer_dossier, nom_ordre


def charger_volumes_dataset(dossier_dataset: Path):
    classes = sorted([p.name for p in dossier_dataset.iterdir() if p.is_dir()])
    chemins = []
    labels = []
    for idx, classe in enumerate(classes):
        fichiers = sorted((dossier_dataset / classe).glob("*.npy"))
        for f in fichiers:
            chemins.append(f)
            labels.append(idx)
    return classes, chemins, np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["D1", "D2"], required=True, help="Dataset généré à utiliser")
    parser.add_argument("--max-objets", type=int, default=None, help="Option de test rapide : limite le nombre total de volumes")
    args = parser.parse_args()

    config = charger_config()
    ordres = [int(o) for o in config["ordres_classification"]]
    dossier_dataset = RACINE / config["dossier_dataset_genere"] / args.dataset
    dossier_sortie = creer_dossier(RACINE / config["dossier_resultats"] / f"features_{args.dataset}")

    if not dossier_dataset.exists():
        raise FileNotFoundError(f"Dataset généré introuvable : {dossier_dataset}. Lance d'abord 01_generer_D1.py ou 02_generer_D2.py")

    classes, chemins, y = charger_volumes_dataset(dossier_dataset)
    if args.max_objets is not None:
        chemins = chemins[: args.max_objets]
        y = y[: args.max_objets]

    print(f"\nExtraction des moments pour {args.dataset}")
    print("Nombre de classes :", len(classes))
    print("Classes :", classes)
    print("Nombre de volumes :", len(chemins))

    pd.DataFrame({"classe_id": range(len(classes)), "classe": classes}).to_csv(dossier_sortie / "classes.csv", index=False)

    for ordre in ordres:
        dossier_ordre = creer_dossier(dossier_sortie / nom_ordre(ordre))
        chemin_X = dossier_ordre / "X.npy"
        chemin_y = dossier_ordre / "y.npy"
        chemin_meta = dossier_ordre / "meta.csv"

        if chemin_X.exists() and chemin_y.exists():
            print(f"Ordre {ordre}: déjà extrait, passage au suivant.")
            continue

        X = []
        metas = []
        print(f"\nOrdre {ordre} : dimension attendue = {ordre ** 3}")
        for chemin, label in tqdm(list(zip(chemins, y)), desc=f"Moments ordre {ordre}"):
            volume = lire_volume_npy(chemin)
            moments = calculer_moments_3d(volume, ordre=ordre)
            X.append(moments_en_vecteur(moments))
            metas.append({"chemin": str(chemin.relative_to(RACINE)), "label": int(label), "classe": classes[int(label)]})

        X = np.vstack(X).astype(np.float32)
        np.save(chemin_X, X)
        np.save(chemin_y, y)
        pd.DataFrame(metas).to_csv(chemin_meta, index=False)
        print(f"Sauvegardé : {chemin_X} shape={X.shape}")

    print("\nExtraction terminée.")


if __name__ == "__main__":
    main()
