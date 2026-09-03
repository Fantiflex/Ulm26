# Face_Reconstruction


Objectif principal

À partir d’une série de choix binaires entre deux visages, estimer une
fonction de préférence individuelle sur un ensemble de visages candidats.

Entrée

- un ensemble standardisé de visages ;
- une liste de comparaisons par paires ;
- pour chaque comparaison, le visage sélectionné ;
- les métadonnées du participant et de l’essai.

Sortie

- un score Bradley–Terry par visage ;
- une incertitude associée aux scores ;
- un classement des visages ;
- des poids utilisables pour une reconstruction faciale ;
- des diagnostics de qualité et d’identifiabilité.

Unité d’analyse

Un participant observant plusieurs paires de visages.

Hypothèse initiale

La probabilité de sélectionner le visage i plutôt que le visage j dépend
de la différence entre leurs scores latents.

