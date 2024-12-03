#include "KNNClassifier.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <cfloat>

using namespace std;

unordered_map<string, unordered_map<int, vector<pair<string, double>>>> distancesByRepresentationAndLabel;

KNNClassifier::KNNClassifier(const vector<Image>& data, int kValue, const string& distType)
    : dataset(data), k(kValue), distanceType(distType) {
    if (!data.empty()) {
        string expectedType = data[0].getRepresentationType();
        for (const auto& img : data) {
            if (img.getRepresentationType() != expectedType) {
                cerr << "Erreur : Les données fournies à KNNClassifier contiennent des représentations différentes." << endl;
                throw runtime_error("Données non homogènes pour KNNClassifier.");
            }
        }
    }
}

double KNNClassifier::calculateDistance(const Image& img1, const Image& img2) const {
    const vector<double>& descriptors1 = img1.getDescripteurs();
    const vector<double>& descriptors2 = img2.getDescripteurs();

    if (descriptors1.size() != descriptors2.size()) {
        cerr << "Erreur : Taille des descripteurs différente entre deux images." << endl;
        return DBL_MAX; 
    }

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
        return distance;
    } else {
        cerr << "Type de distance non reconnu : " << distanceType << endl;
        throw invalid_argument("Type de distance non reconnu");
    }
}

vector<pair<double, int>> KNNClassifier::findKNearestNeighbors(const Image& queryImage) const {
    vector<pair<double, int>> distances;

    for (const auto& img : dataset) {
        double dist = calculateDistance(queryImage, img);
        distances.push_back(make_pair(dist, img.getLabel()));
    }

    sort(distances.begin(), distances.end());

    return vector<pair<double, int>>(distances.begin(), min(distances.begin() + k, distances.end()));
}

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

    // Afficher la prédiction et la confiance
    double confidence = static_cast<double>(maxVotes) / k;
    cout << "Prédiction finale : " << predictedLabel
         << ", Confiance : " << confidence * 100 << "%" << endl;

    return predictedLabel;
}


void KNNClassifier::setK(int kValue) {
    k = kValue;
}


void KNNClassifier::printDatasetInfo() const {
    if (dataset.empty()) {
        cout << "Dataset vide pour KNN." << endl;
        return;
    }

    cout << "=== Informations sur le dataset ===" << endl;
    cout << "Taille du dataset : " << dataset.size() << endl;
    cout << "Type de représentation : " << dataset[0].getRepresentationType() << endl;
    cout << "====================================" << endl;
}

void KNNClassifier::checkClassBalance(const std::vector<Image>& data) {
    unordered_map<int, int> labelCounts;

    for (const auto& img : data) {
        labelCounts[img.getLabel()]++;
    }
    cout << "=== Répartition des classes dans le dataset ===" << endl;
    for (const auto& entry : labelCounts) {
        cout << "Label " << entry.first << ": " << entry.second << " instances" << endl;
    }
    cout << "==============================================" << endl;
}

void KNNClassifier::calculateAndStoreDistances() {
    distancesByRepresentationAndLabel.clear();

    unordered_map<int, vector<Image>> labelToImages;
    for (const auto& img : dataset) {
        labelToImages[img.getLabel()].push_back(img);
    }

    for (auto& entry : labelToImages) {
        int label = entry.first;
        vector<Image>& imagesWithLabel = entry.second;

        if (imagesWithLabel.empty()) continue;
        sort(imagesWithLabel.begin(), imagesWithLabel.end(), [](const Image& a, const Image& b) {
            return a.getImagePath() < b.getImagePath();
        });
        const Image& referenceImage = imagesWithLabel[0];

        string representationType = referenceImage.getRepresentationType();
        for (const auto& img : imagesWithLabel) {
            double distance = calculateDistance(referenceImage, img);
            distancesByRepresentationAndLabel[representationType][label].emplace_back(img.getImagePath(), distance);
        }
    }
}

void KNNClassifier::printStoredDistances() const {
    for (const auto& representationEntry : distancesByRepresentationAndLabel) {
        const string& representation = representationEntry.first;
        cout << "\n=== Distances pour la représentation : " << representation << " ===" << endl;

        for (const auto& labelEntry : representationEntry.second) {
            int label = labelEntry.first;
            const auto& distances = labelEntry.second;

            cout << "Classe " << label << ": [ ";
            for (const auto& distPair : distances) {
                cout << distPair.second << " "; 
            }
            cout << "]" << endl;
        }
    }
}

std::pair<int, double> KNNClassifier::predictLabelWithConfidence(const Image& queryImage) const {
    // Trouver les K plus proches voisins
    std::vector<std::pair<double, int>> neighbors = findKNearestNeighbors(queryImage);

    // Calculer les votes pour chaque label
    std::unordered_map<int, int> labelVotes;
    for (const auto& neighbor : neighbors) {
        labelVotes[neighbor.second]++;
    }

    // Déterminer le label avec le plus de votes
    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& vote : labelVotes) {
        if (vote.second > maxVotes) {
            predictedLabel = vote.first;
            maxVotes = vote.second;
        }
    }

    // Calculer la confiance
    double confidence = static_cast<double>(maxVotes) / k;

    // Retourner le label prédit et la confiance
    return std::make_pair(predictedLabel, confidence);
}
