# ProjetM1RF1

Ce projet a été réalisé par Anaïs Kadic et Titouan Brierre

## Table des matières

1. [Informations générales](#Informations-générales)

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
