from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def racine_projet() -> Path:
    return Path(__file__).resolve().parents[1]


def charger_config() -> dict[str, Any]:
    chemin = racine_projet() / "config.yaml"
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def creer_dossier(chemin: str | Path) -> Path:
    chemin = Path(chemin)
    chemin.mkdir(parents=True, exist_ok=True)
    return chemin


def fixer_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def nom_ordre(ordre: int) -> str:
    return f"ordre_{ordre:03d}"


def lister_fichiers_im(dossier: str | Path) -> list[Path]:
    dossier = Path(dossier)
    return sorted([p for p in dossier.glob("*.im") if p.is_file()])
