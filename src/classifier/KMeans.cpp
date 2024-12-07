#include "classifier/KMeans.h"
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <iostream>

KMeans::KMeans(int numClusters, int numFeatures, int maxIterations, double tolerance)
    : numClusters(numClusters), numFeatures(numFeatures), maxIterations(maxIterations), tolerance(tolerance) {}

void KMeans::fit(const std::vector<Image>& images) {
    // Séparer les images par leur type de représentation
    std::unordered_map<std::string, std::vector<Image>> imagesByRepresentation;
    for (const auto& image : images) {
        imagesByRepresentation[image.getRepresentationType()].push_back(image);
    }

    // Entraîner KMeans sur chaque groupe d'images de la même représentation
    for (const auto& pair : imagesByRepresentation) {
        const std::string& representation = pair.first;
        const std::vector<Image>& repImages = pair.second;

        std::vector<std::vector<double>> centroids(numClusters, std::vector<double>(numFeatures, 0.0));
        std::vector<int> assignments(repImages.size(), -1);

        // Initialisation aléatoire des centroids
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, repImages.size() - 1);
        for (int i = 0; i < numClusters; ++i) {
            centroids[i] = repImages[dist(gen)].getDescripteurs();
        }

        bool converged = false;
        for (int iteration = 0; iteration < maxIterations && !converged; ++iteration) {
            converged = true;

            // Réassigner chaque image à son cluster le plus proche
            for (size_t i = 0; i < repImages.size(); ++i) {
                const auto& features = repImages[i].getDescripteurs();
                double minDistance = std::numeric_limits<double>::max();
                int closestCluster = -1;

                for (int j = 0; j < numClusters; ++j) {
                    double distance = calculateDistance(features, centroids[j]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCluster = j;
                    }
                }

                if (assignments[i] != closestCluster) {
                    assignments[i] = closestCluster;
                    converged = false;
                }
            }

            // Mettre à jour les centroids
            std::vector<std::vector<double>> newCentroids(numClusters, std::vector<double>(numFeatures, 0.0));
            std::vector<int> clusterSizes(numClusters, 0);

            for (size_t i = 0; i < repImages.size(); ++i) {
                int cluster = assignments[i];
                const auto& features = repImages[i].getDescripteurs();
                for (int j = 0; j < numFeatures; ++j) {
                    newCentroids[cluster][j] += features[j];
                }
                ++clusterSizes[cluster];
            }

            for (int j = 0; j < numClusters; ++j) {
                if (clusterSizes[j] > 0) {
                    for (int k = 0; k < numFeatures; ++k) {
                        newCentroids[j][k] /= clusterSizes[j];
                    }
                }
            }

            centroids = newCentroids;
        }

        // Associer les labels aux centroids
        associateLabelsToCentroids(repImages, assignments, centroids);

        // Sauvegarder les centroids et les labels pour cette représentation
        centroidsByRepresentation[representation] = centroids;
    }
}

void KMeans::associateLabelsToCentroids(const std::vector<Image>& images, const std::vector<int>& assignments, const std::vector<std::vector<double>>& centroids) {
    // Associer chaque centroid au label qui est le plus fréquent parmi les images du cluster
    std::unordered_map<int, std::unordered_map<int, int>> clusterLabelCount;  // Cluster -> (Label -> count)
    for (size_t i = 0; i < images.size(); ++i) {
        int cluster = assignments[i];
        int label = images[i].getLabel();
        clusterLabelCount[cluster][label]++;
    }

    std::vector<int> labels(numClusters, -1);
    for (int i = 0; i < numClusters; ++i) {
        int bestLabel = -1;
        int maxCount = -1;
        for (const auto& pair : clusterLabelCount[i]) {
            if (pair.second > maxCount) {
                bestLabel = pair.first;
                maxCount = pair.second;
            }
        }
        labels[i] = bestLabel;
        //std::cout << "Cluster " << i << " is associated with label " << bestLabel << " with " << maxCount << " images." << std::endl;
    }

    centroidLabelsByRepresentation[images[0].getRepresentationType()] = labels;
}

std::pair<int, double> KMeans::predictLabelWithConfidence(const Image& image) const {
    const std::string& representation = image.getRepresentationType();
    auto it = centroidsByRepresentation.find(representation);
    if (it == centroidsByRepresentation.end()) {
        std::cerr << "Erreur : Représentation non trouvée pour la prédiction." << std::endl;
        return {-1, 0.0};
    }

    const auto& centroids = it->second;
    const auto& features = image.getDescripteurs();
    double minDistance = std::numeric_limits<double>::max();
    int closestCluster = -1;

    // Trouver le centroid le plus proche
    for (int i = 0; i < numClusters; ++i) {
        double distance = calculateDistance(features, centroids[i]);
        if (distance < minDistance) {
            minDistance = distance;
            closestCluster = i;
        }
    }

    // Utiliser l'association du label avec le centroid
    auto labelIt = centroidLabelsByRepresentation.find(representation);
    int label = -1;
    if (labelIt != centroidLabelsByRepresentation.end()) {
        label = labelIt->second[closestCluster];
    }

    double confidence = calculateConfidence(features, closestCluster);
    //std::cout << "Predicted label: " << label << " with confidence: " << confidence << std::endl;
    return {label, confidence};
}

double KMeans::calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double KMeans::calculateConfidence(const std::vector<double>& features, int closestCluster) const {
    double distanceToClosest = calculateDistance(features, centroidsByRepresentation.begin()->second[closestCluster]);
    double totalDistance = 0.0;

    for (const auto& centroid : centroidsByRepresentation.begin()->second) {
        totalDistance += calculateDistance(features, centroid);
    }

    return 1.0 - (distanceToClosest / totalDistance);
}
