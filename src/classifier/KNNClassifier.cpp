#include "KNNClassifier.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>

using namespace std;


KNNClassifier::KNNClassifier(const vector<Image>& data, int kValue, const string& distType)
    : dataset(data), k(kValue), distanceType(distType) {}

/**
 * Méthode pour calculer la distance entre deux images, basée sur la méthode de distance spécifiée.
 * 
 * @param img1 : la première image.
 * @param img2 : la deuxième image.
 * @return La distance calculée entre les deux images.
 */
double KNNClassifier::calculateDistance(const Image& img1, const Image& img2) const {
    const vector<double>& descriptors1 = img1.getDescripteurs();
    const vector<double>& descriptors2 = img2.getDescripteurs();
    double distance = 0.0;

    if (distanceType == "euclidean") {
        for (size_t i = 0; i < descriptors1.size(); ++i) {
            distance += pow(descriptors1[i] - descriptors2[i], 2);
        }
        return sqrt(distance);  

    } else if (distanceType == "manhattan") {
        for (size_t i = 0; i < descriptors1.size(); ++i) {
            distance += abs(descriptors1[i] - descriptors2[i]);
        }
        return distance;  // Distance Manhattan
    }
    return -1;  
}

/**
 * Trouve les k voisins les plus proches de l'image donnée, en calculant la distance entre elle et toutes les images du dataset.
 * 
 * @param queryImage : l'image pour laquelle on cherche les k voisins les plus proches.
 * @return Un vecteur de paires (distance, label), trié par ordre croissant de distance.
 */
vector<pair<double, int>> KNNClassifier::findKNearestNeighbors(const Image& queryImage) const {
    vector<pair<double, int>> distances;

    for (const auto& img : dataset) {
        double dist = calculateDistance(queryImage, img);
        distances.push_back(make_pair(dist, img.getLabel()));
    }

    sort(distances.begin(), distances.end());

    return vector<pair<double, int>>(distances.begin(), distances.begin() + k);
}

/**
 * Prédit le label de l'image en fonction des k voisins les plus proches.
 *
 * @param queryImage : l'image à classer.
 * @return Le label prédit en fonction des votes des k voisins les plus proches.
 */
int KNNClassifier::predictLabel(const Image& queryImage) const {
    vector<pair<double, int>> neighbors = findKNearestNeighbors(queryImage);
    unordered_map<int, int> labelVotes;

    for (const auto& neighbor : neighbors) {
        labelVotes[neighbor.second]++;
    }

    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& vote : labelVotes) {
        if (vote.second > maxVotes) {
            predictedLabel = vote.first;
            maxVotes = vote.second;
        }
    }

    return predictedLabel;
}


void KNNClassifier::setK(int kValue) {
    k = kValue;
}
