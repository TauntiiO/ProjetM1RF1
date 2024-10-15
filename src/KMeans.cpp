#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include "dataRepo/DataRepresentation.cpp"
#include "dataRepo/Shape.cpp"

using namespace std;

class KMeans {
private:
    int k;  // Nombre de clusters
    vector<vector<double>> centroids;  // Centroids des clusters
    vector<int> labels;  // Labels pour chaque point

public:
    KMeans(int k) : k(k) {
        srand(static_cast<unsigned>(time(0)));  // Initialiser le générateur de nombres aléatoires
    }

    void fit(const vector<Shape>& shapes) {
        // Initialiser les centroids aléatoirement
        initializeCentroids(shapes);
        
        bool changed;
        do {
            changed = false;
            // Étape 2 : Assignation des clusters
            assignClusters(shapes);
            
            // Étape 3 : Mise à jour des centroids
            for (int i = 0; i < k; ++i) {
                vector<double> newCentroid = updateCentroid(shapes, i);
                if (newCentroid != centroids[i]) {
                    centroids[i] = newCentroid;
                    changed = true;  // Un centroid a changé
                }
            }
        } while (changed);  // Répéter jusqu'à ce qu'il n'y ait plus de changement
    }

    void initializeCentroids(const vector<Shape>& shapes) {
        // Choisir k centroids aléatoires à partir des formes
        for (int i = 0; i < k; ++i) {
            int index = rand() % shapes.size();
            centroids.push_back(shapes[index].getDescripteurs());
        }
    }

    void assignClusters(const vector<Shape>& shapes) {
        labels.resize(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            double minDistance = numeric_limits<double>::max();
            for (int j = 0; j < k; ++j) {
                double distance = euclideanDistance(shapes[i].getDescripteurs(), centroids[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    labels[i] = j;  // Assigner le cluster le plus proche
                }
            }
        }
    }

    vector<double> updateCentroid(const vector<Shape>& shapes, int clusterId) {
        vector<double> newCentroid(centroids[clusterId].size(), 0.0);
        int count = 0;

        for (size_t i = 0; i < shapes.size(); ++i) {
            if (labels[i] == clusterId) {
                const vector<double>& point = shapes[i].getDescripteurs();
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

    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(sum);
    }

    void printClusters() const {
        for (int i = 0; i < k; ++i) {
            cout << "Cluster " << i << " : ";
            for (size_t j = 0; j < labels.size(); ++j) {
                if (labels[j] == i) {
                    cout << j << " ";  // Affiche l'indice du point
                }
            }
            cout << endl;
        }
    }
};

int main() {
    cout << "Chemin actuel : " << filesystem::current_path() << endl;

    string path = "C:/M1/RF/ProjetM1RF1/data/=Signatures/=ART";
    vector<Shape> shapes;

    // Chargement des fichiers de descripteurs
    for (const auto& entry : filesystem::directory_iterator(path)) {
        DataRepresentation rep(entry.path().string());
        if (rep.readFile()) {
            shapes.emplace_back(rep.getData(), shapes.size());  // Crée un objet Shape avec les descripteurs et un label
        } else {
            cerr << "Erreur en lisant le fichier : " << entry.path() << endl;
        }
    }

    // Appliquer K-Means
    KMeans kmeans(18);  // 18 classes
    kmeans.fit(shapes);

    // Imprimer les clusters
    kmeans.printClusters();

    return 0;
}
