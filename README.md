# Moments 3D de Krawtchouk

Ce dépôt contient les scripts, données légères et résultats principaux utilisés pour les expériences sur les moments 3D de Krawtchouk. Le dossier principal est `classification_krawtchouk_3d/`.

## Structure du dépôt

```text
.
├── README.md
├── classification_krawtchouk_3d/
│   ├── config.yaml
│   ├── requirements.txt
│   ├── data/
│   ├── scripts/
│   ├── src/
│   └── results/
├── selection_png/
├── results_2D/
├── results_shoulder/
├── shoulder_TG/
├── test_moments/
├── articulated.zip
└── non-articulated.zip
```

- `classification_krawtchouk_3d/` : dossier principal du projet 3D.
- `classification_krawtchouk_3d/src/` : fonctions de lecture des volumes, transformations 3D, calcul des moments, classification et visualisation.
- `classification_krawtchouk_3d/scripts/` : scripts exécutables, numérotés dans l'ordre des expériences.
- `classification_krawtchouk_3d/config.yaml` : paramètres principaux : classes, ordres des moments, dossiers, hyperparamètres.
- `classification_krawtchouk_3d/data/` : archives légères pour reproduire les expériences.
- `classification_krawtchouk_3d/results/` : figures et tableaux déjà générés.
- `selection_png/` : images d'aperçu de quelques objets/classes.
- `results_2D/`, `results_shoulder/`, `shoulder_TG/`, `test_moments/` : anciens essais ou résultats complémentaires.
- `articulated.zip`, `non-articulated.zip` : archives des données McGill d'origine.

## Installation

Depuis la racine du dépôt :

```bash
cd classification_krawtchouk_3d
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Les scripts sont prévus pour être lancés depuis `classification_krawtchouk_3d/`.

## Données

Le dossier attendu pour les volumes McGill est :

```text
classification_krawtchouk_3d/dataset_original/
```

Une archive légère est fournie :

```bash
cd classification_krawtchouk_3d
unzip data/dataset_original.zip
```

Pour vérifier que les classes attendues sont présentes :

```bash
.venv/bin/python scripts/00_verifier_dataset_original.py
```

Des caractéristiques pré-calculées sont aussi disponibles pour l'expérience avec augmentation groupée :

```bash
unzip data/features_grouped_augmentation_D1.zip
```

## Utilisation des scripts

### Expérience D1

```bash
cd classification_krawtchouk_3d

# Générer le dataset D1
.venv/bin/python scripts/01_generer_D1.py

# Extraire les moments de Krawtchouk
.venv/bin/python scripts/04_extraire_moments.py --dataset D1

# Lancer la classification
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/05_classifier.py --dataset D1 --validation split
```

Les sorties sont écrites dans :

```text
classification_krawtchouk_3d/dataset_genere/
classification_krawtchouk_3d/results/
```

### Expérience avec généralisation par objet

```bash
cd classification_krawtchouk_3d

# Extraire les moments avec augmentation groupée
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/11_extraire_moments_grouped_augmentation.py --class-set D1

# Classifier avec validation croisée groupée
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/12_classifier_grouped_augmentation.py --class-set D1

# Générer les figures d'analyse des erreurs
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/13_visualiser_erreurs_grouped_augmentation.py --class-set D1
```

## Liste des scripts

| Script | Rôle |
| --- | --- |
| `00_verifier_dataset_original.py` | Vérifie les classes disponibles dans `dataset_original/`. |
| `01_generer_D1.py` | Génère le dataset D1. |
| `02_generer_D2.py` | Génère le dataset D2. |
| `03_reconstruction_3d_article.py` | Produit des reconstructions 3D à différents ordres. |
| `04_extraire_moments.py` | Extrait les moments pour D1 ou D2. |
| `05_classifier.py` | Entraîne et évalue le classifieur DNN. |
| `06_generer_bruit_D2.py` | Génère les versions bruitées de D2. |
| `07_extraire_moments_bruit.py` | Extrait les moments sur un dataset bruité. |
| `08_classifier_bruit.py` | Classifie les données bruitées. |
| `09_extraire_moments_objects.py` | Extrait les moments directement sur les objets originaux. |
| `10_classifier_objects_5fold.py` | Lance une validation croisée par objet. |
| `11_extraire_moments_grouped_augmentation.py` | Extrait les moments avec augmentation groupée. |
| `12_classifier_grouped_augmentation.py` | Classifie avec validation croisée groupée. |
| `13_visualiser_erreurs_grouped_augmentation.py` | Génère les visualisations d'erreurs. |

## Remarques

- Les paramètres importants se modifient dans `classification_krawtchouk_3d/config.yaml`.
- Certains scripts peuvent générer beaucoup de fichiers `.npy`.
- Si `matplotlib` pose un problème de cache, ajouter `MPLCONFIGDIR=.mplconfig` devant la commande.
