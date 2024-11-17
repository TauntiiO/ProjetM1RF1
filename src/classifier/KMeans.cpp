#include "KMeans.h"

KMeans::KMeans(int k) : k(k) {
    srand(static_cast<unsigned>(time(0)));  // Initialiser le générateur de nombres aléatoires
}

void KMeans::fit(const std::vector<Image>& images) {
    // Initialiser les centroids aléatoirement
    initializeCentroids(images);
    
    bool changed;
    do {
        changed = false;
        // Étape 2 : Assignation des clusters
        assignClusters(images);
        
        // Étape 3 : Mise à jour des centroids
        for (int i = 0; i < k; ++i) {
            std::vector<double> newCentroid = updateCentroid(images, i);
            if (newCentroid != centroids[i]) {
                centroids[i] = newCentroid;
                changed = true;  // Un centroid a changé
            }
        }
    } while (changed);  // Répéter jusqu'à ce qu'il n'y ait plus de changement
}

void KMeans::initializeCentroids(const std::vector<Image>& images) {
    // Choisir k centroids aléatoires à partir des images
    for (int i = 0; i < k; ++i) {
        int index = rand() % images.size();
        centroids.push_back(images[index].getDescripteurs());
    }
}

void KMeans::assignClusters(const std::vector<Image>& images) {
    labels.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < k; ++j) {
            double distance = euclideanDistance(images[i].getDescripteurs(), centroids[j]);
            if (distance < minDistance) {
                minDistance = distance;
                labels[i] = j;  // Assigner le cluster le plus proche
            }
        }
    }
}

std::vector<double> KMeans::updateCentroid(const std::vector<Image>& images, int clusterId) {
    std::vector<double> newCentroid(centroids[clusterId].size(), 0.0);
    int count = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        if (labels[i] == clusterId) {
            const std::vector<double>& point = images[i].getDescripteurs();
            for (size_t j = 0; j < point.size(); ++j) {
                newCentroid[j] += point[j];
            }
            count++;
        }
    }

    // Éviter la division par zéro
    if (count > 0) {
        for (size_t j = 0; j < newCentroid.size(); ++j) {
            newCentroid[j] /= count;  // Prendre la moyenne
        }
    }
    return newCentroid;
}

double KMeans::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void KMeans::printClusters(const std::vector<Image>& images) const {
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << " : ";
        for (size_t j = 0; j < labels.size(); ++j) {
            if (labels[j] == i) {
                std::cout << j << " ";  // Affiche l'indice du point
            }
        }
        std::cout << std::endl;
    }
    int totalCorrect = 0;
    int totalImages = images.size();  // Nombre total d'images dans le dataset

    // Parcourir chaque cluster
    for (int i = 0; i < k; ++i) {
        std::cout << "Cluster " << i << " : ";
        
        // Compter les fréquences des labels réels dans le cluster
        std::unordered_map<int, int> labelCounts;
        for (size_t j = 0; j < labels.size(); ++j) {
            if (labels[j] == i) {  // Si l'image appartient au cluster i
                int realLabel = images[j].getLabel();  // Récupérer le label réel de l'image
                labelCounts[realLabel]++;  // Compter les occurrences de chaque label réel
            }
        }

        // Trouver le label le plus fréquent dans ce cluster
        int mostFrequentLabel = -1;
        int maxCount = 0;
        for (const auto& pair : labelCounts) {
            if (pair.second > maxCount) {
                mostFrequentLabel = pair.first;
                maxCount = pair.second;
            }
        }

        // Afficher le label le plus fréquent dans ce cluster
        std::cout << "Label le plus fréquent : " << mostFrequentLabel << " avec " << maxCount << " occurrences." << std::endl;

        // Ajouter les bonnes prédictions à la somme totale
        totalCorrect += maxCount;
    }
    
    // Afficher la statistique des prédictions correctes
    std::cout << "Prédictions correctes : " << totalCorrect << " sur " << totalImages << std::endl;
}
