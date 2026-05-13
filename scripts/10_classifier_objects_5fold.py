import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _bootstrap import RACINE
from src.entrainement import entrainer_et_evaluer
from src.evaluation import sauvegarder_courbe_accuracy, sauvegarder_matrice_confusion
from src.utils import charger_config, creer_dossier, fixer_seed, nom_ordre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-set",
        choices=["object_cv", "D1", "D2", "all"],
        default="D1",
        help="Doit correspondre au --class-set utilisé dans 09_extraire_moments_objects.py",
    )
    parser.add_argument("--folds", type=int, default=5, help="Nombre de folds object-level CV")
    args = parser.parse_args()

    config = charger_config()
    fixer_seed(int(config["entrainement"].get("seed", 42)))

    ordres = [int(o) for o in config["ordres_object_cv"]]
    dossier_features = RACINE / config["dossier_resultats"] / "features_object_cv" / args.class_set
    dossier_resultats = creer_dossier(
        RACINE / config["dossier_resultats"] / "object_cv_classification" / args.class_set
    )

    if not dossier_features.exists():
        raise FileNotFoundError(
            f"Features introuvables : {dossier_features}. "
            "Lance d'abord scripts/09_extraire_moments_objects.py"
        )

    classes = pd.read_csv(dossier_features / "classes.csv")["classe"].tolist()
    nombre_classes = len(classes)

    lignes = []
    meilleur = {"accuracy": -1, "ordre": None, "cm": None}
    predictions_lignes = []

    for ordre in ordres:
        dossier_ordre = dossier_features / nom_ordre(ordre)
        X = np.load(dossier_ordre / "X.npy")
        y = np.load(dossier_ordre / "y.npy")
        meta = pd.read_csv(dossier_ordre / "meta.csv")

        print(f"\nObject-level classification — ordre {ordre} — {args.folds}-fold")
        print("Dimension X :", X.shape)
        print("Objets par classe :", np.bincount(y).tolist())

        kfold = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=int(config["entrainement"].get("seed", 42)),
        )

        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        cms = []
        epochs = []

        for fold, (idx_train, idx_test) in enumerate(kfold.split(X, y), start=1):
            resultat = entrainer_et_evaluer(
                X[idx_train],
                y[idx_train],
                X[idx_test],
                y[idx_test],
                nombre_classes=nombre_classes,
                config_entrainement=config["entrainement"],
                chemin_modele=None,
            )

            print(
                f"  fold {fold}: accuracy={100 * resultat.accuracy:.2f}% "
                f"| f1={100 * resultat.f1:.2f}% "
                f"| meilleur epoch={resultat.meilleur_epoch}"
            )

            accuracies.append(resultat.accuracy)
            precisions.append(resultat.precision)
            recalls.append(resultat.recall)
            f1s.append(resultat.f1)
            cms.append(resultat.matrice_confusion)
            epochs.append(resultat.meilleur_epoch)

            for local_i, pred, vrai in zip(idx_test, resultat.predictions, resultat.vrais_labels):
                predictions_lignes.append(
                    {
                        "ordre": ordre,
                        "fold": fold,
                        "chemin": meta.iloc[int(local_i)]["chemin"],
                        "object_id": meta.iloc[int(local_i)]["object_id"],
                        "classe_reelle": classes[int(vrai)],
                        "classe_predite": classes[int(pred)],
                        "label_reel": int(vrai),
                        "label_predit": int(pred),
                    }
                )

        accuracy = float(np.mean(accuracies))
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))
        cm = np.sum(cms, axis=0)

        lignes.append(
            {
                "dataset": "original_objects",
                "class_set": args.class_set,
                "validation": f"{args.folds}fold_object_level",
                "ordre": ordre,
                "accuracy_obtenue": 100 * accuracy,
                "precision_macro": 100 * precision,
                "recall_macro": 100 * recall,
                "f1_macro": 100 * f1,
                "meilleur_epoch_moyen": float(np.mean(epochs)),
            }
        )

        if accuracy > meilleur["accuracy"]:
            meilleur = {"accuracy": accuracy, "ordre": ordre, "cm": cm}

    df = pd.DataFrame(lignes)
    chemin_csv = dossier_resultats / "accuracy_by_order_5fold.csv"
    df.to_csv(chemin_csv, index=False)
    pd.DataFrame(predictions_lignes).to_csv(dossier_resultats / "predictions_by_object_5fold.csv", index=False)

    sauvegarder_courbe_accuracy(
        df,
        dossier_resultats / "accuracy_vs_order_5fold.png",
        titre=f"Object-level 5-fold CV — {args.class_set}",
    )

    if meilleur["cm"] is not None:
        sauvegarder_matrice_confusion(
            meilleur["cm"],
            classes,
            dossier_resultats / "confusion_matrix_best_order_5fold.png",
            titre=f"Matrice de confusion — meilleur ordre {meilleur['ordre']} — object-level 5-fold",
        )

    print("\nRésultats sauvegardés dans :", dossier_resultats)
    print("Tableau :", chemin_csv)
    print("Meilleur ordre :", meilleur["ordre"], f"accuracy={100 * meilleur['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
