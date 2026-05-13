from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .modele_dnn import DNNMomentsTchebichef


@dataclass
class ResultatEntrainement:
    accuracy: float
    precision: float
    recall: float
    f1: float
    matrice_confusion: np.ndarray
    predictions: np.ndarray
    vrais_labels: np.ndarray
    meilleur_epoch: int
    train_losses: list[float]
    valid_losses: list[float]


def standardiser_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moyenne = X_train.mean(axis=0, keepdims=True)
    ecart_type = X_train.std(axis=0, keepdims=True)
    ecart_type[ecart_type < 1e-8] = 1.0
    return (X_train - moyenne) / ecart_type, (X_test - moyenne) / ecart_type


def standardiser_avec_train(
    X_train: np.ndarray,
    *tableaux: np.ndarray,
) -> tuple[np.ndarray, ...]:
    moyenne = X_train.mean(axis=0, keepdims=True)
    ecart_type = X_train.std(axis=0, keepdims=True)
    ecart_type[ecart_type < 1e-8] = 1.0
    return tuple((X.astype(np.float32) - moyenne) / ecart_type for X in (X_train, *tableaux))


def entrainer_et_evaluer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    nombre_classes: int,
    config_entrainement: dict,
    chemin_modele: str | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> ResultatEntrainement:
    seed = int(config_entrainement.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if X_val is None or y_val is None:
        X_val = X_test
        y_val = y_test

    X_train, X_val, X_test = standardiser_avec_train(
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
    )
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modele = DNNMomentsTchebichef(
        dimension_entree=X_train.shape[1],
        nombre_classes=nombre_classes,
        dropout=float(config_entrainement.get("dropout", 0.15)),
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=int(config_entrainement.get("batch_size", 64)),
        shuffle=True,
    )

    critere = nn.CrossEntropyLoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=float(config_entrainement.get("learning_rate", 1e-3)))

    epochs = int(config_entrainement.get("epochs", 80))
    patience = int(config_entrainement.get("patience", 12))
    meilleur_loss = float("inf")
    meilleur_etat = None
    meilleur_epoch = 0
    attente = 0

    X_val_tensor = torch.tensor(X_val).to(device)
    y_val_tensor = torch.tensor(y_val).to(device)
    X_test_tensor = torch.tensor(X_test).to(device)
    y_test_tensor = torch.tensor(y_test).to(device)
    train_losses = []
    valid_losses = []

    for epoch in range(1, epochs + 1):
        modele.train()
        train_loss_total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimiseur.zero_grad()
            loss = critere(modele(xb), yb)
            loss.backward()
            optimiseur.step()
            train_loss_total += loss.item() * xb.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)
        train_losses.append(float(train_loss))

        modele.eval()
        with torch.no_grad():
            val_loss = critere(modele(X_val_tensor), y_val_tensor).item()
        valid_losses.append(float(val_loss))

        if val_loss < meilleur_loss:
            meilleur_loss = val_loss
            meilleur_etat = {k: v.cpu().clone() for k, v in modele.state_dict().items()}
            meilleur_epoch = epoch
            attente = 0
        else:
            attente += 1
            if attente >= patience:
                break

    if meilleur_etat is not None:
        modele.load_state_dict(meilleur_etat)

    if chemin_modele is not None:
        torch.save(modele.state_dict(), chemin_modele)

    modele.eval()
    with torch.no_grad():
        logits = modele(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=list(range(nombre_classes)))

    return ResultatEntrainement(
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        matrice_confusion=cm,
        predictions=predictions,
        vrais_labels=y_test,
        meilleur_epoch=meilleur_epoch,
        train_losses=train_losses,
        valid_losses=valid_losses,
    )
