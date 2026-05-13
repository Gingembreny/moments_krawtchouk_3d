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
        for f in sorted((dossier_dataset / classe).glob("*.npy")):
            chemins.append(f)
            labels.append(idx)
    return classes, chemins, np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bruit", required=True, help="Nom du sous-dossier dans dataset_genere/D2_bruit, ex: salt_pepper_01pct")
    args = parser.parse_args()

    config = charger_config()
    ordres = [int(o) for o in config["ordres_classification"]]
    dossier_dataset = RACINE / config["dossier_dataset_genere"] / "D2_bruit" / args.bruit
    dossier_sortie = creer_dossier(RACINE / config["dossier_resultats"] / f"features_D2_bruit_{args.bruit}")

    if not dossier_dataset.exists():
        raise FileNotFoundError(f"Dataset bruité introuvable : {dossier_dataset}")

    classes, chemins, y = charger_volumes_dataset(dossier_dataset)
    print(f"\nExtraction moments D2 bruité : {args.bruit}")
    print("Classes :", classes)
    print("Volumes :", len(chemins))
    pd.DataFrame({"classe_id": range(len(classes)), "classe": classes}).to_csv(dossier_sortie / "classes.csv", index=False)

    for ordre in ordres:
        dossier_ordre = creer_dossier(dossier_sortie / nom_ordre(ordre))
        if (dossier_ordre / "X.npy").exists():
            print(f"Ordre {ordre}: déjà extrait.")
            continue
        X = []
        for chemin in tqdm(chemins, desc=f"{args.bruit} ordre {ordre}"):
            volume = lire_volume_npy(chemin)
            X.append(moments_en_vecteur(calculer_moments_3d(volume, ordre=ordre)))
        np.save(dossier_ordre / "X.npy", np.vstack(X).astype(np.float32))
        np.save(dossier_ordre / "y.npy", y)

    print("\nExtraction terminée.")


if __name__ == "__main__":
    main()
