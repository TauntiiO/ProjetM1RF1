# ProjetM1RF1

Ce projet a été réalisé par Anaïs Kadic et Titouan Brierre

## Table des matières

1. [Informations générales](#Informations-générales)
2. [Technologies](#Techmologies)
3. [Structure du projet](#Structure-du-projet)
4. [Compilation et exécution](#Compilation-et-exécution)

## Informations générales

Ce projet porte sur l’évaluation et la comparaison de différentes méthodes de reconnaissance de formes dans le cadre d’un projet de Master. Il se compose de plusieurs étapes et implémentations liées à l'analyse de descripteurs d'images pour la classification. Le projet explore l’utilisation des méthodes K-means et K-Nearest Neighbors (KNN) pour classer des images en fonction de leurs caractéristiques extraites. Les données utilisées pour l’évaluation du modèle proviennent de différentes représentations d'images, et les performances des algorithmes sont mesurées à travers des métriques telles que la précision, le rappel et le F1-score.

Des matrices de confusion et des courbes précision-rappel (PR) ont été générées pour comparer les performances des modèles sur différentes classes d'images (10 et 18 classes). Les résultats sont visualisés sous forme de graphiques et de courbes afin d’identifier les modèles les plus efficaces et d'analyser les résultats en fonction des variations du nombre de classes et de l'algorithme choisi.

Ce projet permet de visualiser de manière détaillée les performances des algorithmes et d’approfondir l’analyse comparative en fonction de l’algorithme utilisé et du nombre de classes. Le code inclut des scripts pour générer des visualisations des métriques ainsi que des matrices de confusion, ce qui facilite l'interprétation des résultats pour un usage ultérieur dans la reconnaissance des formes.

## Technologies

### Langages

 - [C++](https://fr.wikipedia.org/wiki/C%2B%2B#:~:text=C%2B%2B%20est%20un%20langage,objet%20et%20la%20programmation%20générique.) : Pour l'implémentation des algorithmes KNN et K-Means
 - [Python](https://www.python.org) : Pour l'implémentation de scripts permettant la visualisation et la comparaison des résultats et métriques des algorithmes

### Bibliothèques

 - C++ : \<cmath>, \<limits>, \<random>, \<unordered_map>, \<algorithm>, \<iostream>, et \<filesystem> ont été utilisées pour gérer les calculs mathématiques, les structures de données, les fichiers, et l'implémentation des logiques d'algorithmes.
 - Python : pandas, matplotlib, seaborn, et sklearn sont utilisées pour le traitement des données et la création des plots.

### Gestion de versions

[Git](https://git-scm.com)

## Structure du projet

 - [src/classifier](https://github.com/TauntiiO/ProjetM1RF1/tree/main/src/classifier) : Contient l'implémentation des algorithmes K-means et KNN en C++.

 - [src/dataRepo](https://github.com/TauntiiO/ProjetM1RF1/tree/main/src/dataRepo) : Contient l'imlémentation de tous les outils nécessaires pour la lecture des fichiers et la structuration des données.

 - [src/evaluation](https://github.com/TauntiiO/ProjetM1RF1/tree/main/src/evaluation) : Dossier contenant les outils nécessaires au calcul des metriques des algorithmes.

 - [src/main](https://github.com/TauntiiO/ProjetM1RF1/tree/main/src/main) : Contient le fichier main, il appelle les les classes déclarées dans les dossiers décrits ci-dessus pour importer les données, entrainer les algorithmes dessus, les tester, puis sauvegarder les résultats dans des csv.
   
 - [scripts/](https://github.com/TauntiiO/ProjetM1RF1/tree/main/scripts) : Contient les scripts Python utilisés pour générer les visualisations et analyser les résultats. Contient aussi un script pour préparer des dossiers avec les jeux d'entrainement et de test.

 - [include/](https://github.com/TauntiiO/ProjetM1RF1/tree/main/include) : Contient les fichiers d'en-tête des classes de [src/](https://github.com/TauntiiO/ProjetM1RF1/tree/main/src/)
   
 - [results/](https://github.com/TauntiiO/ProjetM1RF1/tree/main/results) : Dossier de sortie contenant les matrices de confusion, les graphiques des métriques et les courbes PR.
   
 - [data/](https://github.com/TauntiiO/ProjetM1RF1/tree/main/data) : Contient les jeux de données d'entrée pour l'entraînement et les tests des modèles.

## Compilation et exécution

Pour compiler et exécuter le projet, suivez les étapes ci-dessous :

1. Compiler le projet :

 - Allez à la racine du projet.
 - Tapez la commande suivante pour compiler le projet :
```
make
```
 - Cette commande va compiler les sources et créer l'exécutable.
2. Exécuter le projet :

 - Après la compilation, vous pouvez exécuter le projet avec la commande :
```
./project_metrics
```
3. Recompiler le projet :

 - Si vous apportez des modifications aux fichiers source, il est recommandé de nettoyer les anciens fichiers compilés avant de recompiler.
 - Utilisez la commande suivante pour nettoyer les fichiers précédemment compilés :
```
make clean
```
 - Puis, compilez à nouveau avec la commande make.
