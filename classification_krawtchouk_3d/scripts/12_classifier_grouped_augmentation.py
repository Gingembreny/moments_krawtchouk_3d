import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from _bootstrap import RACINE
from src.entrainement import entrainer_et_evaluer
from src.evaluation import sauvegarder_courbe_accuracy, sauvegarder_matrice_confusion
from src.utils import charger_config, creer_dossier, fixer_seed, nom_ordre


def sauvegarder_courbe_loss(train_losses: list[float], valid_losses: list[float], chemin: Path, titre: str) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(titre)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chemin, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-set", choices=["D1", "D2"], default="D1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-folds", type=int, default=5, help="Validation split inside training objects")
    args = parser.parse_args()

    config = charger_config()
    seed = int(config["entrainement"].get("seed", 42))
    fixer_seed(seed)

    ordres = [int(o) for o in config["ordres_grouped_augmentation"]]
    dossier_features = RACINE / config["dossier_resultats"] / "features_grouped_augmentation" / args.class_set
    dossier_resultats = creer_dossier(
        RACINE / config["dossier_resultats"] / "grouped_augmentation_classification" / args.class_set
    )
    dossier_loss = creer_dossier(dossier_resultats / "loss_curves")

    if not dossier_features.exists():
        raise FileNotFoundError(
            f"Features introuvables : {dossier_features}. "
            "Lance d'abord scripts/11_extraire_moments_grouped_augmentation.py"
        )

    classes = pd.read_csv(dossier_features / "classes.csv")["classe"].tolist()
    nombre_classes = len(classes)

    lignes = []
    predictions_lignes = []
    split_assignment_lignes = []
    meilleur = {"accuracy": -1, "ordre": None, "cm": None}

    for ordre in ordres:
        dossier_ordre = dossier_features / nom_ordre(ordre)
        X = np.load(dossier_ordre / "X.npy")
        y = np.load(dossier_ordre / "y.npy")
        groups = np.load(dossier_ordre / "groups.npy")
        meta = pd.read_csv(dossier_ordre / "meta.csv")

        print(f"\nGrouped augmentation classification — ordre {ordre} — {args.folds}-fold")
        print("Dimension X :", X.shape)
        print("Objets originaux :", len(np.unique(groups)))
        print("Échantillons par classe :", np.bincount(y).tolist())

        outer_cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)

        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        cms = []
        epochs = []

        for fold, (idx_train_val, idx_test) in enumerate(outer_cv.split(X, y, groups), start=1):
            X_train_val = X[idx_train_val]
            y_train_val = y[idx_train_val]
            groups_train_val = groups[idx_train_val]

            inner_cv = StratifiedGroupKFold(n_splits=args.val_folds, shuffle=True, random_state=seed + fold)
            idx_train_local, idx_val_local = next(
                inner_cv.split(X_train_val, y_train_val, groups_train_val)
            )

            idx_train = idx_train_val[idx_train_local]
            idx_val = idx_train_val[idx_val_local]

            if ordre == ordres[0]:
                for role, indices in (
                    ("train", idx_train),
                    ("validation", idx_val),
                    ("test", idx_test),
                ):
                    objets_role = meta.iloc[indices][
                        ["object_key", "object_id", "classe", "label"]
                    ].drop_duplicates("object_key")
                    for _, row in objets_role.iterrows():
                        split_assignment_lignes.append(
                            {
                                "fold": fold,
                                "role": role,
                                "object_key": row["object_key"],
                                "object_id": row["object_id"],
                                "classe": row["classe"],
                                "label": int(row["label"]),
                            }
                        )

            resultat = entrainer_et_evaluer(
                X[idx_train],
                y[idx_train],
                X[idx_test],
                y[idx_test],
                nombre_classes=nombre_classes,
                config_entrainement=config["entrainement"],
                chemin_modele=None,
                X_val=X[idx_val],
                y_val=y[idx_val],
            )

            sauvegarder_courbe_loss(
                resultat.train_losses,
                resultat.valid_losses,
                dossier_loss / f"loss_order_{ordre:03d}_fold_{fold}.png",
                titre=f"Loss — ordre {ordre} — fold {fold}",
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
                        "object_key": meta.iloc[int(local_i)]["object_key"],
                        "variant_id": int(meta.iloc[int(local_i)]["variant_id"]),
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
                "dataset": "grouped_augmentation",
                "class_set": args.class_set,
                "validation": f"{args.folds}fold_grouped_object_level",
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
    suffixe = f"{args.folds}fold"
    chemin_csv = dossier_resultats / f"accuracy_by_order_{suffixe}.csv"
    df.to_csv(chemin_csv, index=False)
    pd.DataFrame(predictions_lignes).to_csv(dossier_resultats / f"predictions_by_sample_{suffixe}.csv", index=False)
    pd.DataFrame(split_assignment_lignes).to_csv(
        dossier_resultats / f"split_assignments_{suffixe}.csv",
        index=False,
    )

    sauvegarder_courbe_accuracy(
        df,
        dossier_resultats / f"accuracy_vs_order_{suffixe}.png",
        titre=f"Grouped augmentation object-level {args.folds}-fold — {args.class_set}",
    )

    if meilleur["cm"] is not None:
        sauvegarder_matrice_confusion(
            meilleur["cm"],
            classes,
            dossier_resultats / f"confusion_matrix_best_order_{suffixe}.png",
            titre=f"Matrice de confusion — meilleur ordre {meilleur['ordre']} — grouped augmentation {args.folds}-fold",
        )

    print("\nRésultats sauvegardés dans :", dossier_resultats)
    print("Tableau :", chemin_csv)
    print("Meilleur ordre :", meilleur["ordre"], f"accuracy={100 * meilleur['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
