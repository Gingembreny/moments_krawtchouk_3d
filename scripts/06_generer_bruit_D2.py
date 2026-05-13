from pathlib import Path

from tqdm import tqdm

from _bootstrap import RACINE
from src.bruit import ajouter_salt_pepper, ajouter_speckle
from src.io_volumes import lire_volume_npy, sauvegarder_volume_npy
from src.utils import charger_config, creer_dossier


def main():
    config = charger_config()
    source = RACINE / config["dossier_dataset_genere"] / "D2"
    destination = RACINE / config["dossier_dataset_genere"] / "D2_bruit"
    creer_dossier(destination)

    if not source.exists():
        raise FileNotFoundError("D2 n'existe pas. Lance d'abord scripts/02_generer_D2.py")

    print("\nGénération des datasets bruités D2 comme dans l'article")

    niveaux_sp = config["bruit_salt_pepper"]
    niveaux_speckle = config["bruit_speckle_sigma"]

    fichiers = sorted(source.glob("*/*.npy"))
    print("Nombre de volumes D2 :", len(fichiers))

    for densite in niveaux_sp:
        nom_bruit = f"salt_pepper_{int(densite*100):02d}pct"
        print("\nBruit", nom_bruit)
        for chemin in tqdm(fichiers):
            rel = chemin.relative_to(source)
            volume = lire_volume_npy(chemin)
            noisy = ajouter_salt_pepper(volume, densite=float(densite), seed=42)
            sauvegarder_volume_npy(noisy, destination / nom_bruit / rel)

    for sigma in niveaux_speckle:
        nom_bruit = f"speckle_sigma_{str(sigma).replace('.', '_')}"
        print("\nBruit", nom_bruit)
        for chemin in tqdm(fichiers):
            rel = chemin.relative_to(source)
            volume = lire_volume_npy(chemin)
            noisy = ajouter_speckle(volume, sigma=float(sigma), seed=42)
            sauvegarder_volume_npy(noisy, destination / nom_bruit / rel)

    print("\nDatasets bruités sauvegardés dans :", destination)


if __name__ == "__main__":
    main()
