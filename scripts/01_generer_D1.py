from pathlib import Path
from itertools import product
import numpy as np
from tqdm import tqdm
from scipy import ndimage as ndi
from _bootstrap import RACINE
from src.io_volumes import lire_volume_im, sauvegarder_volume_npy
from src.transformations import mise_a_echelle_3d, rotation_3d
from src.utils import charger_config, creer_dossier, lister_fichiers_im


def main():
    config = charger_config()
    taille = int(config["taille_volume"])
    padding = int(config.get("padding_generation", 16))
    source = RACINE / config["dossier_dataset_original"]
    destination = RACINE / config["dossier_dataset_genere"] / "D1"
    creer_dossier(destination)

    print("\nGénération de D1 comme dans l'article")
    print("Transformations : 5 échelles × 4 rotations x × 4 rotations y × 4 rotations z = 320 objets/classe")
    print(f"Padding de génération : {padding} voxels, taille finale = {taille + 2 * padding}")
    print("Destination :", destination)

    echelles = config["facteurs_echelle_D1"]
    angles = config["angles_rotation_D1"]

    for classe in config["classes_D1"]:
        dossier_classe = source / classe
        fichiers = lister_fichiers_im(dossier_classe)
        if not fichiers:
            print(f"Classe ignorée car absente ou vide : {classe}")
            continue

        # L'article génère D1 à partir d'un objet source par classe.
        fichier_source = fichiers[0]
        volume_original = lire_volume_im(fichier_source, taille=taille)

        # Centering
        coords = np.argwhere(volume_original > 0)
        if len(coords) > 0:
            center_mass = coords.mean(axis=0)

            target_center = np.array(volume_original.shape) / 2

            shift = target_center - center_mass

            volume_original = ndi.shift(
                volume_original,
                shift=shift,
                order=0,
                mode="constant",
                cval=0,
            )

        # Small padding to avoid border effects during rotation.
        volume_original = np.pad(
            volume_original,
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )

        dest_classe = creer_dossier(destination / classe)

        compteur = 0
        for s, ax, ay, az in tqdm(list(product(echelles, angles, angles, angles)), desc=classe):
            v = mise_a_echelle_3d(volume_original, float(s))
            v = rotation_3d(v, angle_x=float(ax), angle_y=float(ay), angle_z=float(az))
            nom = f"{classe}_scale{s:.1f}_rx{ax}_ry{ay}_rz{az}.npy"
            sauvegarder_volume_npy(v, dest_classe / nom)
            compteur += 1
        print(f"  {classe}: {compteur} volumes générés à partir de {fichier_source.name}")

    print("\nD1 terminé.")


if __name__ == "__main__":
    main()
