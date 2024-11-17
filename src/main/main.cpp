#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "../dataRepo/DataRepresentation.h"
#include "../dataRepo/Image.h"
#include "../dataRepo/DataCollection.h"
#include "../classifier/KNNClassifier.h"
#include "../classifier/KMeans.h"

using namespace std;

int main() {
    string representationsDir = "C:/M1/RF/ProjetM1RF1/data/=Signatures"; 

    // Charger le dataset
    DataCollection dataset;
    dataset.loadDatasetFromDirectory(representationsDir); 
    dataset.printDataset(); 

    // Créer un vecteur contenant les images du dataset
    vector<Image> images = dataset.getImages(); 

    // Conserver uniquement les images dont le label est compris entre 1 et 10
    vector<Image> filteredImages;
    for (const auto& img : images) {
        int label = img.getLabel();
        if (label >= 1 && label <= 10) {
            filteredImages.push_back(img);
        }
    }

    // Partie 1 : Classification avec KNN
    cout << "=== Classification avec KNN ===" << endl;
    int kKNN = 4;
    KNNClassifier knn(filteredImages, kKNN, "euclidean");

    int correctPredictions = 0;
    map<int, int> labelFrequency; 

    for (const auto& img : filteredImages) {
        int predictedLabel = knn.predictLabel(img);
        int realLabel = img.getLabel();

        if (predictedLabel == realLabel) {
            correctPredictions++;
        }

        labelFrequency[predictedLabel]++;
    }

    cout << "Prédictions correctes : " << correctPredictions << " sur " << filteredImages.size() << endl;

    int mostRecognizedLabel = -1;
    int maxCount = 0;
    for (const auto& entry : labelFrequency) {
        if (entry.second > maxCount) {
            mostRecognizedLabel = entry.first;
            maxCount = entry.second;
        }
    }
    cout << "Le label le plus souvent reconnu est : " << mostRecognizedLabel << " avec " << maxCount << " prédictions." << endl;

    // Partie 2 : Clustering avec K-Means

    // Filtrer les images par type de descripteur (YANG, ART, GFD)
    vector<Image> yangImages;
    vector<Image> artImages;
    vector<Image> gfdImages;

    // Séparer les images en fonction du type de descripteur
    for (const auto& img : filteredImages) {
        if (img.getRepresentationType() == "Yang") {
            yangImages.push_back(img);
        } else if (img.getRepresentationType() == "ART") {
            artImages.push_back(img);
        } else if (img.getRepresentationType() == "GFD") {
            gfdImages.push_back(img);
        }
    }

    // Clustering pour les images Yang
    cout << "=== Clustering avec K-Means sur les images Yang ===" << endl;
    int kKMeans = 10;  // Nombre de clusters
    KMeans kmeansYang(kKMeans);
    kmeansYang.fit(yangImages);
    kmeansYang.printClusters(yangImages);

    // Clustering pour les images ART
    cout << "=== Clustering avec K-Means sur les images ART ===" << endl;
    KMeans kmeansArt(kKMeans);
    kmeansArt.fit(artImages);
    kmeansArt.printClusters(artImages);

    // Clustering pour les images GFD
    cout << "=== Clustering avec K-Means sur les images GFD ===" << endl;
    KMeans kmeansGfd(kKMeans);
    kmeansGfd.fit(gfdImages);
    kmeansGfd.printClusters(gfdImages);

    return 0;
}
