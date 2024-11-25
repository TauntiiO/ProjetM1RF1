#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include "../dataRepo/DataRepresentation.h"
#include "../dataRepo/Image.h"

class KMeans {
public:
    // Constructeur
    KMeans(int k);

    // Méthodes principales
    void fit(const std::vector<Image>& images);         // Entraîner le modèle
    int predictCluster(const Image& image);            // Prédire le cluster d'une image
    void printClusters(const std::vector<Image>& images) const; // Afficher les clusters et leur contenu

private:
    int k;                                             // Nombre de clusters
    std::vector<std::vector<double>> centroids;        // Centroids des clusters
    std::vector<int> labels;                           // Assignation des clusters pour chaque point
    std::vector<int> clusterLabels;                   // Labels dominants des clusters

    // Méthodes auxiliaires
    void initializeCentroids(const std::vector<Image>& images); // Initialiser les centroids
    void assignClusters(const std::vector<Image>& images);      // Assigner les clusters
    std::vector<double> updateCentroid(const std::vector<Image>& images, int clusterId); // Mettre à jour un centroid
    void updateClusterLabels(const std::vector<Image>& images); // Déterminer les labels dominants des clusters
    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b); // Calculer la distance euclidienne
};


#endif // KMEANS_H
