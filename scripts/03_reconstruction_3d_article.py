from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from _bootstrap import RACINE
from src.io_volumes import lire_volume_im
from src.moments3d_krawtchouk import calculer_moments_3d, reconstruire_volume_3d, mse, seuillage_par_volume_original, dice_score
from src.utils import charger_config, creer_dossier, lister_fichiers_im
from src.visualisation3d import sauvegarder_grille_reconstruction_3d, sauvegarder_grille_projection


def main():
    config = charger_config()
    taille = int(config["taille_volume"])
    dossier_original = RACINE / config["dossier_dataset_original"]
    dossier_resultats = creer_dossier(RACINE / config["dossier_resultats"] / "reconstruction")
    ordres = [int(o) for o in config["ordres_reconstruction"]]

    classes_exemples = ["airplanesIm", "antsIm", "chairsIm"]
    lignes = []

    print("\nReconstruction 3D comme dans l'article")
    print("Objets : airplane, ant, chair")
    print("Ordres :", ordres)

    for classe in classes_exemples:
        fichiers = lister_fichiers_im(dossier_original / classe)
        if not fichiers:
            print(f"Classe absente : {classe}")
            continue

        fichier = fichiers[0]
        original = lire_volume_im(fichier, taille=taille)
        volumes_3d = [original]
        titres = [f"Original\n{classe}"]

        print(f"\nObjet {classe} : {fichier.name}")
        for ordre in tqdm(ordres, desc=f"Reconstruction {classe}"):
            moments = calculer_moments_3d(original, ordre=ordre)
            recon = reconstruire_volume_3d(moments, taille=original.shape[0])
            recon_binaire = seuillage_par_volume_original(recon, original)
            valeur_mse = mse(original, recon)
            valeur_dice = dice_score(original, recon_binaire)
            volumes_3d.append(recon_binaire)
            titres.append(f"Ordre {ordre}\nMSE={valeur_mse:.4f}\nDice={valeur_dice:.3f}")
            lignes.append({"classe": classe, "objet": fichier.name, "ordre": ordre, "mse": valeur_mse, "dice": valeur_dice})
            print(f"  ordre {ordre:03d} -> MSE={valeur_mse:.6f}, Dice={valeur_dice:.4f}")

        sauvegarder_grille_reconstruction_3d(
            volumes_3d,
            titres,
            dossier_resultats / f"reconstruction_3d_{classe}.png",
        )
        sauvegarder_grille_projection(
            volumes_3d,
            titres,
            dossier_resultats / f"reconstruction_projection_{classe}.png",
        )

    df = pd.DataFrame(lignes)
    df.to_csv(dossier_resultats / "mse_reconstruction_article_orders.csv", index=False)

    if not df.empty:
        resume = df.groupby("ordre", as_index=False).agg(mse_moyenne=("mse", "mean"), dice_moyen=("dice", "mean"))
        resume.to_csv(dossier_resultats / "mse_dice_reconstruction_resume.csv", index=False)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(resume["ordre"], resume["mse_moyenne"], marker="o", label="MSE moyenne")
        ax1.set_xlabel("Ordre des moments de Tchebichef")
        ax1.set_ylabel("Mean Squared Error (MSE)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(resume["ordre"], resume["dice_moyen"], marker="s", linestyle="--", label="Dice moyen")
        ax2.set_ylabel("Score Dice moyen")

        fig.suptitle("Reconstruction 3D : MSE et Dice en fonction de l'ordre")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
        plt.tight_layout()
        plt.savefig(dossier_resultats / "mse_dice_reconstruction_article_orders.png", dpi=200)
        plt.close(fig)

    print("\nRésultats sauvegardés dans :", dossier_resultats)


if __name__ == "__main__":
    main()
