# Classification 3D par moments de Krawtchouk

Ce dossier contient une version legere et partageable des scripts et des resultats principaux du projet. Les jeux de donnees volumineux, les matrices de caracteristiques `.npy` et les poids de modeles `.pt` ne sont pas inclus.

## Structure

- `src/` : fonctions de lecture des volumes, transformations 3D, moments de Krawtchouk, DNN et evaluation.
- `scripts/` : scripts experimentaux.
- `config.yaml` : classes utilisees, ordres de moments, hyperparametres.
- `results/` : figures et tableaux selectionnes pour le rapport/PPT.

## Installation

```bash
cd krawtchouk_3d_github
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Les donnees originales McGill doivent etre placees dans `dataset_original/`. Les donnees generees et les matrices de caracteristiques sont ignorees par Git.

Pour faciliter la reproduction, une archive legere du jeu original utilise est incluse :

```text
data/dataset_original.zip
```

Elle peut etre decompressee a la racine du projet avant de lancer les scripts de generation.

Des caracteristiques pre-calculees sont egalement fournies pour eviter de relancer les etapes les plus longues :

```text
data/features_grouped_augmentation_D1.zip
data/features_D1_split.z01 ... data/features_D1_split.z06
data/features_D1_split.zip
```

L'archive `features_D1` est decoupee en plusieurs fichiers afin de rester compatible avec les limites de GitHub. Pour la reconstituer, placer tous les fichiers `features_D1_split.*` dans le meme dossier puis decompresser `features_D1_split.zip`.

## Experience 1 : D1 avec transformations

Objectif : verifier la capacite des moments 3D de Krawtchouk a separer 10 classes lorsque chaque classe est representee par un objet source transforme.

Pipeline :

1. Selection des 10 classes D1 et d'un objet source par classe.
2. Centrage et padding des volumes : `128^3 -> 160^3`.
3. Generation de D1 par transformations : 320 volumes par classe, 3200 volumes au total.
4. Extraction des moments de Krawtchouk d'ordres `4, 6, ..., 20`.
5. Classification par DNN avec split stratifie par classe.

Commandes :

```bash
.venv/bin/python scripts/01_generer_D1.py
.venv/bin/python scripts/04_extraire_moments.py --dataset D1
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/05_classifier.py --dataset D1 --validation split
```

Resultats inclus :

- `results/D1_classification/accuracy_by_order_split_train40.csv`
- `results/D1_classification/accuracy_vs_order_split_train40.png`
- `results/D1_classification/confusion_matrix_best_order_split_train40.png`
- `results/D1_classification/loss_curves/loss_D1_split_train40_ordre_004.png`

Resultat principal : l'ordre 4 donne la meilleure performance sur ce protocole, avec environ `97.7 %` d'accuracy dans le split 40/60 avec validation interne.

## Experience 2 : generalisation par objet avec augmentation groupee

Objectif : tester une vraie generalisation. Le modele doit reconnaitre des objets jamais utilises a l'entrainement, tout en profitant d'une augmentation moderee.

Principe :

- Les 10 classes D1 sont conservees.
- Tous les objets disponibles de ces classes sont utilises.
- Chaque objet genere 16 transformations.
- Le split est fait au niveau des objets : toutes les transformations d'un meme objet restent du meme cote.
- Validation croisee 5-fold avec groupes d'objets.
- Une validation interne sert a choisir le meilleur modele selon la validation loss.
- Les ordres testes sont `4, 5, 6, 7, 8, 9, 10`.

Commandes :

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/11_extraire_moments_grouped_augmentation.py --class-set D1
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/12_classifier_grouped_augmentation.py --class-set D1
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/13_visualiser_erreurs_grouped_augmentation.py --class-set D1
```

Resultats inclus :

- `results/grouped_augmentation_classification/D1/accuracy_by_order_5fold.csv`
- `results/grouped_augmentation_classification/D1/accuracy_vs_order_5fold.png`
- `results/grouped_augmentation_classification/D1/confusion_matrix_best_order_5fold.png`
- `results/grouped_augmentation_classification/D1/loss_curves/`
- `results/grouped_augmentation_classification/D1/visual_examples/`

Resultat principal : l'ordre 4 reste le meilleur, avec environ `78.2 %` d'accuracy au niveau sample. Le vote majoritaire par objet monte autour de `82 %`, ce qui montre que plusieurs transformations d'un meme objet stabilisent la prediction.

## Figures utiles pour le PPT

- Schema D1 : `results/flowcharts/D1_pipeline_flowchart_fr.mmd`
- Schema generalisation : `results/flowcharts/object_generalisation_flowchart_fr.mmd`
- Courbes accuracy : `accuracy_vs_order_*.png`
- Matrices de confusion : `confusion_matrix_*.png`
- Exemples visuels : `visual_examples/*.png`

## Conclusion courte

Les resultats montrent que les moments de Krawtchouk de bas ordre contiennent deja une information discriminante forte sur la forme globale. Quand l'ordre augmente, les caracteristiques deviennent plus sensibles aux details et aux transformations, ce qui reduit la stabilite de classification dans ces experiences.
