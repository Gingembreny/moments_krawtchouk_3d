# Plan PPT

## 1. Objectif

Classification d'objets 3D a partir de moments orthogonaux de Krawtchouk.

Message : tester si des descripteurs de forme globaux peuvent separer des classes d'objets 3D.

## 2. Methode

- Volumes binaires 3D issus du dataset McGill.
- Centrage et padding : `128^3 -> 160^3`.
- Extraction des moments 3D de Krawtchouk.
- Classification par DNN.

Figure conseillee : une visualisation d'objet 3D ou une projection.

## 3. Experience D1

- 10 classes.
- 1 objet source par classe.
- Transformations : rotations et echelles.
- 3200 volumes generes.
- Split stratifie 40 % train / 60 % test, avec validation interne.

Figure conseillee : `results/flowcharts/D1_pipeline_flowchart_fr.mmd`.

## 4. Resultats D1

- Meilleure performance a l'ordre 4.
- Accuracy ordre 4 : environ 97.7 %.
- La performance diminue quand l'ordre augmente.

Figures conseillees :

- `results/D1_classification/accuracy_vs_order_split_train40.png`
- `results/D1_classification/confusion_matrix_best_order_split_train40.png`

## 5. Limite du protocole D1

D1 teste surtout la reconnaissance de transformations d'objets deja proches de l'objet source. Ce n'est pas encore une vraie generalisation a de nouveaux objets d'une meme classe.

## 6. Experience de generalisation

- 10 classes D1.
- Tous les objets disponibles sont utilises.
- Augmentation moderee : 16 transformations par objet.
- Split 5-fold au niveau des objets.
- Les transformations d'un meme objet restent ensemble.
- Validation interne pour choisir le meilleur modele.

Figure conseillee : `results/flowcharts/object_generalisation_flowchart_fr.mmd`.

## 7. Resultats generalisation

- Meilleur ordre : 4.
- Accuracy sample-level : environ 78.2 %.
- Accuracy object-level par vote majoritaire : environ 82 %.
- Les bas ordres sont plus stables que les ordres eleves.

Figures conseillees :

- `results/grouped_augmentation_classification/D1/accuracy_vs_order_5fold.png`
- `results/grouped_augmentation_classification/D1/confusion_matrix_best_order_5fold.png`

## 8. Analyse qualitative

Comparer les classes bien reconnues, difficiles et confondues.

Figures conseillees :

- `visual_examples/good_classes_order_004.png`
- `visual_examples/difficult_classes_order_004.png`
- `visual_examples/top_confusions_order_004.png`
- `visual_examples/dolphins_correct_vs_wrong_order_004.png`

## 9. Validation loss

La validation loss permet de garder le modele au meilleur epoch et de limiter le surapprentissage.

Figure conseillee :

- `results/grouped_augmentation_classification/D1/loss_curves/loss_order_004_fold_1.png`

## 10. Conclusion

- Les moments de Krawtchouk de bas ordre decrivent bien la forme globale.
- L'ordre 4 est le plus robuste dans les deux protocoles.
- L'augmentation groupee ameliore la quantite de donnees tout en conservant un test rigoureux sur objets non vus.
- Les confusions restantes correspondent souvent a des formes globalement proches.
