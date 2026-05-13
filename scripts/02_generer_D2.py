from pathlib import Path

import numpy as np
from tqdm import tqdm

from _bootstrap import RACINE
from src.io_volumes import lire_volume_im, sauvegarder_volume_npy
from src.transformations import translation_3d, rotation_3d, mise_a_echelle_3d, transformation_mixte
from src.utils import charger_config, creer_dossier, lister_fichiers_im


def main():
    config = charger_config()
    taille = int(config["taille_volume"])
    padding = int(config.get("padding_generation", 16))
    source = RACINE / config["dossier_dataset_original"]
    destination = RACINE / config["dossier_dataset_genere"] / "D2"
    creer_dossier(destination)

    print("\nGénération de D2 comme dans l'article")
    print("D2 nécessite 3 objets source par classe.")
    print("Transformations par objet source : 4 translations + 4 rotations + 4 échelles + 4 mixtes = 16")
    print(f"Padding de génération : {padding} voxels, taille finale = {taille + 2 * padding}")
    print("Destination :", destination)

    translations = [tuple(t) for t in config["translations_D2"]]
    angles = config["angles_rotation_D2"]
    echelles = config["facteurs_echelle_D2"]

    classes_manquantes = []

    for classe in config["classes_D2"]:
        dossier_classe = source / classe
        fichiers = lister_fichiers_im(dossier_classe)
        if len(fichiers) < 3:
            classes_manquantes.append((classe, len(fichiers)))
            continue

        dest_classe = creer_dossier(destination / classe)
        compteur = 0
        for idx, fichier_source in enumerate(fichiers[:3], start=1):
            volume_original = lire_volume_im(fichier_source, taille=taille)
            volume_original = np.pad(
                volume_original,
                pad_width=padding,
                mode="constant",
                constant_values=0,
            )

            for t in translations:
                v = translation_3d(volume_original, t)
                sauvegarder_volume_npy(v, dest_classe / f"src{idx}_translation_{t[0]}_{t[1]}_{t[2]}.npy")
                compteur += 1

            for angle in angles:
                # D2 contient 4 rotations. On applique une rotation 3D simple avec le même angle sur les trois axes.
                v = rotation_3d(volume_original, angle_x=float(angle), angle_y=float(angle), angle_z=float(angle))
                sauvegarder_volume_npy(v, dest_classe / f"src{idx}_rotation_{angle}.npy")
                compteur += 1

            for s in echelles:
                v = mise_a_echelle_3d(volume_original, float(s))
                sauvegarder_volume_npy(v, dest_classe / f"src{idx}_scale_{s:.1f}.npy")
                compteur += 1

            for k in range(4):
                v = transformation_mixte(
                    volume_original,
                    translation=translations[k],
                    angle=float(angles[k]),
                    facteur=float(echelles[k]),
                )
                sauvegarder_volume_npy(v, dest_classe / f"src{idx}_mixed_{k}.npy")
                compteur += 1

        print(f"  {classe}: {compteur} volumes générés")

    if classes_manquantes:
        print("\nATTENTION : D2 n'a pas été généré pour certaines classes :")
        for classe, nb in classes_manquantes:
            print(f"  {classe}: {nb} fichier(s) trouvé(s), il en faut au moins 3")
        print("Si tu n'as pas spiders/tables/teddy, commence par D1.")

    print("\nD2 terminé pour les classes disponibles.")


if __name__ == "__main__":
    main()
