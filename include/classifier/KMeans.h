#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <unordered_map>
#include "dataRepo/Image.h"

class KMeans {
public:
    KMeans(int numClusters, int numFeatures, int maxIterations = 100, double tolerance = 1e-4);

    // Entraîner KMeans avec un ensemble d'images
    void fit(const std::vector<Image>& images);

    // Prédire le label et retourner un score de confiance pour une image donnée
    std::pair<int, double> predictLabelWithConfidence(const Image& image) const;

private:
    int numClusters;  // Nombre de clusters (classes)
    int numFeatures;  // Nombre de dimensions dans les descripteurs
    int maxIterations;  // Nombre maximum d'itérations
    double tolerance;  // Tolérance pour la convergence

    // Centroids pour chaque représentation
    std::unordered_map<std::string, std::vector<std::vector<double>>> centroidsByRepresentation;
    // Labels associés aux centroids
    std::unordered_map<std::string, std::vector<int>> centroidLabelsByRepresentation;

    // Fonction auxiliaire pour calculer la distance entre deux vecteurs
    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Calculer le score de confiance basé sur les distances aux centroids
    double calculateConfidence(const std::vector<double>& features, int closestCluster) const;

    // Associer un label au centroid pendant l'entraînement
    void associateLabelsToCentroids(const std::vector<Image>& images, const std::vector<int>& assignments, const std::vector<std::vector<double>>& centroids);
};

#endif // KMEANS_H
