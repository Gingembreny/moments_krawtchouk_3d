import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _bootstrap import RACINE
from src.entrainement import entrainer_et_evaluer
from src.evaluation import sauvegarder_courbe_accuracy
from src.utils import charger_config, creer_dossier, nom_ordre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bruit", required=True, help="Nom du bruit, ex: salt_pepper_01pct")
    args = parser.parse_args()

    config = charger_config()
    ordres = [int(o) for o in config["ordres_classification"]]
    dossier_features = RACINE / config["dossier_resultats"] / f"features_D2_bruit_{args.bruit}"
    dossier_resultats = creer_dossier(RACINE / config["dossier_resultats"] / "D2_bruit_classification" / args.bruit)

    classes = pd.read_csv(dossier_features / "classes.csv")["classe"].tolist()
    nombre_classes = len(classes)
    lignes = []

    for ordre in ordres:
        X = np.load(dossier_features / nom_ordre(ordre) / "X.npy")
        y = np.load(dossier_features / nom_ordre(ordre) / "y.npy")
        print(f"\nClassification D2 bruité {args.bruit}, ordre {ordre}, 5-fold")
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(config["entrainement"].get("seed", 42)))
        accs = []
        for fold, (idx_train, idx_test) in enumerate(kfold.split(X, y), start=1):
            res = entrainer_et_evaluer(
                X[idx_train], y[idx_train], X[idx_test], y[idx_test],
                nombre_classes=nombre_classes,
                config_entrainement=config["entrainement"],
            )
            accs.append(res.accuracy)
            print(f"  fold {fold}: {100*res.accuracy:.2f}%")
        lignes.append({"bruit": args.bruit, "ordre": ordre, "accuracy_obtenue": 100 * float(np.mean(accs))})

    df = pd.DataFrame(lignes)
    df.to_csv(dossier_resultats / "accuracy_by_order_5fold.csv", index=False)
    sauvegarder_courbe_accuracy(
        df,
        dossier_resultats / "accuracy_vs_order_5fold.png",
        titre=f"D2 bruité {args.bruit} — Accuracy 5-fold",
    )
    print("\nRésultats sauvegardés dans :", dossier_resultats)


if __name__ == "__main__":
    main()
