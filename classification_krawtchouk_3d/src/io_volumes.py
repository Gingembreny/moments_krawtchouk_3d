from pathlib import Path
import numpy as np


def lire_volume_im(chemin, taille=128):
    """
    Lit un fichier .im représentant un volume 3D.

    Les fichiers McGill .im peuvent contenir :
    - soit directement 128^3 valeurs uint8,
    - soit un header au début du fichier, souvent 1024 octets,
      puis les 128^3 voxels.

    La fonction détecte automatiquement s'il y a trop de données
    et saute le header si nécessaire.
    """
    chemin = Path(chemin)
    nb_voxels_attendus = taille ** 3

    donnees = np.fromfile(chemin, dtype=np.uint8)

    if donnees.size == nb_voxels_attendus:
        volume = donnees

    elif donnees.size > nb_voxels_attendus:
        taille_header = donnees.size - nb_voxels_attendus
        print(
            f"  Header détecté pour {chemin.name} : "
            f"{taille_header} octets ignorés"
        )
        volume = donnees[taille_header:]

    else:
        raise ValueError(
            f"Taille trop petite pour {chemin}. "
            f"Trouvé {donnees.size} valeurs, attendu {nb_voxels_attendus}."
        )

    volume = volume.reshape((taille, taille, taille))

    # Normalisation en 0/1
    volume = volume.astype(np.float32)
    if volume.max() > 1:
        volume = volume / 255.0

    # Binarisation propre
    volume = (volume > 0.5).astype(np.float32)

    return volume


def sauvegarder_volume_npy(volume, chemin):
    """
    Sauvegarde un volume 3D au format .npy.
    C'est le format utilisé pour les datasets générés D1/D2.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.save(chemin, volume.astype(np.float32))


def lire_volume_npy(chemin):
    """
    Lit un volume sauvegardé en .npy.
    """
    return np.load(chemin).astype(np.float32)