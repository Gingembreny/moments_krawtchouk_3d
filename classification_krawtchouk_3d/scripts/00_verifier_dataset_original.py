from pathlib import Path
import sys

from _bootstrap import RACINE
from src.utils import charger_config, lister_fichiers_im


def main():
    config = charger_config()
    dossier = RACINE / config["dossier_dataset_original"]
    print("\nVérification du dataset original")
    print("Dossier :", dossier)

    if not dossier.exists():
        print("ERREUR : le dossier dataset_original n'existe pas.")
        sys.exit(1)

    toutes_classes = sorted(set(config["classes_D1"] + config["classes_D2"]))
    for classe in toutes_classes:
        dossier_classe = dossier / classe
        if not dossier_classe.exists():
            print(f"  {classe:15s} : absent")
            continue
        fichiers = lister_fichiers_im(dossier_classe)
        print(f"  {classe:15s} : {len(fichiers)} fichier(s) .im")

    print("\nD1 nécessite les classes :", ", ".join(config["classes_D1"]))
    print("D2 nécessite les classes :", ", ".join(config["classes_D2"]))


if __name__ == "__main__":
    main()
