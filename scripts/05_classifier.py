import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from _bootstrap import RACINE
from src.entrainement import entrainer_et_evaluer
from src.evaluation import sauvegarder_courbe_accuracy, sauvegarder_matrice_confusion
from src.utils import charger_config, creer_dossier, fixer_seed, nom_ordre


def charger_classes(dossier_features: Path) -> list[str]:
    df = pd.read_csv(dossier_features / "classes.csv")
    return df["classe"].tolist()


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
    parser.add_argument("--dataset", choices=["D1", "D2"], required=True)
    parser.add_argument("--validation", choices=["split", "5fold", "S1", "S2", "S3"], default="split")
    parser.add_argument(
        "--train-size",
        type=float,
        default=None,
        help="Proportion d'entraînement pour les validations split/S1/S2/S3, ex: 0.5 pour 50/50",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Proportion de la partie entraînement utilisée comme validation, ex: 0.2",
    )
    args = parser.parse_args()

    if args.train_size is not None and not 0 < args.train_size < 1:
        parser.error("--train-size doit être strictement entre 0 et 1")
    if not 0 < args.val_size < 1:
        parser.error("--val-size doit être strictement entre 0 et 1")

    config = charger_config()
    fixer_seed(int(config["entrainement"].get("seed", 42)))

    dossier_features = RACINE / config["dossier_resultats"] / f"features_{args.dataset}"
    dossier_resultats = creer_dossier(RACINE / config["dossier_resultats"] / f"{args.dataset}_classification")
    dossier_modeles = creer_dossier(dossier_resultats / "modeles")
    dossier_loss = creer_dossier(dossier_resultats / "loss_curves")

    classes = charger_classes(dossier_features)
    nombre_classes = len(classes)
    ordres = [int(o) for o in config["ordres_classification"]]

    lignes = []
    meilleur = {"accuracy": -1, "ordre": None, "cm": None}

    for ordre in ordres:
        dossier_ordre = dossier_features / nom_ordre(ordre)
        X = np.load(dossier_ordre / "X.npy")
        y = np.load(dossier_ordre / "y.npy")

        print(f"\nClassification {args.dataset} — ordre {ordre} — validation {args.validation}")
        print("Dimension X :", X.shape)

        if args.validation == "5fold":
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(config["entrainement"].get("seed", 42)))
            accuracies = []
            precisions = []
            recalls = []
            f1s = []
            cms = []
            epochs = []
            for fold, (idx_train_val, idx_test) in enumerate(kfold.split(X, y), start=1):
                idx_train, idx_val = train_test_split(
                    idx_train_val,
                    test_size=args.val_size,
                    stratify=y[idx_train_val],
                    random_state=int(config["entrainement"].get("seed", 42)) + fold,
                )
                resultat = entrainer_et_evaluer(
                    X[idx_train], y[idx_train], X[idx_test], y[idx_test],
                    nombre_classes=nombre_classes,
                    config_entrainement=config["entrainement"],
                    chemin_modele=None,
                    X_val=X[idx_val],
                    y_val=y[idx_val],
                )
                sauvegarder_courbe_loss(
                    resultat.train_losses,
                    resultat.valid_losses,
                    dossier_loss / f"loss_{args.dataset}_{args.validation}_ordre_{ordre:03d}_fold_{fold}.png",
                    titre=f"Loss — {args.dataset} ordre {ordre} fold {fold}",
                )
                print(f"  fold {fold}: accuracy={100*resultat.accuracy:.2f}%")
                accuracies.append(resultat.accuracy)
                precisions.append(resultat.precision)
                recalls.append(resultat.recall)
                f1s.append(resultat.f1)
                cms.append(resultat.matrice_confusion)
                epochs.append(resultat.meilleur_epoch)

            accuracy = float(np.mean(accuracies))
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            f1 = float(np.mean(f1s))
            cm = np.sum(cms, axis=0)
            meilleur_epoch = float(np.mean(epochs))
        else:
            if args.train_size is not None:
                train_size = args.train_size
            elif args.dataset == "D1" or args.validation == "split":
                # Article D1 : 40% train / 60% test.
                train_size = 0.40
            elif args.validation == "S1":
                train_size = 0.40
            elif args.validation == "S2":
                train_size = 0.30
            elif args.validation == "S3":
                train_size = 0.20
            else:
                train_size = 0.40

            idx = np.arange(len(y))
            idx_train, idx_test = train_test_split(
                idx,
                train_size=train_size,
                stratify=y,
                random_state=int(config["entrainement"].get("seed", 42)),
            )
            idx_train, idx_val = train_test_split(
                idx_train,
                test_size=args.val_size,
                stratify=y[idx_train],
                random_state=int(config["entrainement"].get("seed", 42)) + ordre,
            )
            resultat = entrainer_et_evaluer(
                X[idx_train], y[idx_train], X[idx_test], y[idx_test],
                nombre_classes=nombre_classes,
                config_entrainement=config["entrainement"],
                chemin_modele=str(dossier_modeles / f"dnn_krawtchouk_{args.dataset}_{args.validation}_train{int(round(train_size * 100)):02d}_ordre_{ordre:03d}.pt"),
                X_val=X[idx_val],
                y_val=y[idx_val],
            )
            sauvegarder_courbe_loss(
                resultat.train_losses,
                resultat.valid_losses,
                dossier_loss / f"loss_{args.dataset}_{args.validation}_train{int(round(train_size * 100)):02d}_ordre_{ordre:03d}.png",
                titre=f"Loss — {args.dataset} ordre {ordre}",
            )
            accuracy = resultat.accuracy
            precision = resultat.precision
            recall = resultat.recall
            f1 = resultat.f1
            cm = resultat.matrice_confusion
            meilleur_epoch = resultat.meilleur_epoch
            print(f"  accuracy={100*accuracy:.2f}% | f1={100*f1:.2f}% | meilleur epoch={meilleur_epoch}")

        if args.dataset == "D1":
            article = config["article_D1_tchebichef_accuracy"].get(ordre)
        elif args.validation == "5fold":
            article = config["article_D2_tchebichef_5fold_accuracy"].get(ordre)
        else:
            article = None

        lignes.append({
            "dataset": args.dataset,
            "validation": args.validation,
            "train_size": None if args.validation == "5fold" else train_size,
            "val_size_in_train": args.val_size,
            "ordre": ordre,
            "accuracy_obtenue": 100 * accuracy,
            "precision_macro": 100 * precision,
            "recall_macro": 100 * recall,
            "f1_macro": 100 * f1,
            "accuracy_article": article,
            "ecart_article": None if article is None else 100 * accuracy - float(article),
            "meilleur_epoch": meilleur_epoch,
        })

        if accuracy > meilleur["accuracy"]:
            meilleur = {"accuracy": accuracy, "ordre": ordre, "cm": cm}

    df = pd.DataFrame(lignes)
    suffixe = args.validation
    if args.validation != "5fold":
        train_size_suffix = args.train_size
        if train_size_suffix is None:
            train_size_suffix = 0.40 if args.dataset == "D1" or args.validation in ("split", "S1") else 0.30 if args.validation == "S2" else 0.20
        suffixe = f"{suffixe}_train{int(round(train_size_suffix * 100)):02d}"

    chemin_csv = dossier_resultats / f"accuracy_by_order_{suffixe}.csv"
    df.to_csv(chemin_csv, index=False)

    sauvegarder_courbe_accuracy(
        df,
        dossier_resultats / f"accuracy_vs_order_{suffixe}.png",
        titre=f"{args.dataset} — Accuracy en fonction de l'ordre ({args.validation})",
    )

    if meilleur["cm"] is not None:
        sauvegarder_matrice_confusion(
            meilleur["cm"], classes,
            dossier_resultats / f"confusion_matrix_best_order_{suffixe}.png",
            titre=f"Matrice de confusion — meilleur ordre {meilleur['ordre']} — {args.dataset}",
        )

    print("\nRésultats sauvegardés dans :", dossier_resultats)
    print("Tableau :", chemin_csv)


if __name__ == "__main__":
    main()
