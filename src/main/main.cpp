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
#include "../classifier/KNNClassifier.h"
#include "../evaluation/ConfusionMatrix.h"
#include "../evaluation/Metrics.h"

using namespace std;

int main() {
    //string representationsDir = "C:/M1/RF/ProjetM1RF1/data/=Signatures"; 
    DataCollection dataset;
    dataset.loadDatasetFromDirectory("/home/user/Documents/M1/s1/ProjetM1RF1/data/=Signatures");
    dataset.printDataset();

    vector<Image> images = dataset.getImages();

    vector<Image> filteredImages;
    for (const auto& img : images) {
        if (img.getLabel() >= 1 && img.getLabel() <= 10) {
            filteredImages.push_back(img);
        }
    }

    // Diviser les données en train/test (80% train, 20% test)
    vector<Image> trainImages, testImages;
    for (size_t i = 0; i < filteredImages.size(); ++i) {
        if (i % 5 == 0) {
            testImages.push_back(filteredImages[i]); // 20% des données pour test
        } else {
            trainImages.push_back(filteredImages[i]); // 80% des données pour entraînement
        }
    }

    cout << "Train size: " << trainImages.size() << ", Test size: " << testImages.size() << endl;

    auto groupedImages = dataset.groupImagesByRepresentation(trainImages);

    double totalAccuracy = 0.0;
    int totalTestedImages = 0;

    for (const auto& group : groupedImages) {
        const string& representationType = group.first;
        const vector<Image>& groupTrainImages = group.second;

        cout << "\n=== Calcul des distances et classification pour la représentation : " << representationType << " ===" << endl;

        KNNClassifier knn(groupTrainImages, 3, "euclidean"); 
        knn.calculateAndStoreDistances();
        knn.printStoredDistances(); 

        ConfusionMatrix confusionMatrix(10);

        int correctPredictions = 0;
        int totalTestsForThisRepresentation = 0;

        for (const auto& testImage : testImages) {
            if (testImage.getRepresentationType() == representationType) {
                int predictedLabel = knn.predictLabel(testImage); 
                confusionMatrix.addPrediction(testImage.getLabel(), predictedLabel); 

                if (predictedLabel == testImage.getLabel()) {
                    correctPredictions++;
                }

                totalTestsForThisRepresentation++;
            }
        }

        confusionMatrix.printMatrix();

        cout << "\n=== Métriques pour la représentation " << representationType << " ===" << endl;
        Metrics::printMetrics(confusionMatrix.getMatrix());

     
        double accuracy = static_cast<double>(correctPredictions) / totalTestsForThisRepresentation;
        cout << "Précision pour la représentation " << representationType << " : " << accuracy * 100 << "%" << endl;
        totalAccuracy += accuracy * totalTestsForThisRepresentation;
        totalTestedImages += totalTestsForThisRepresentation;
    }

    double globalAccuracy = totalAccuracy / totalTestedImages;
    cout << "\n=== Résumé global ===" << endl;
    cout << "Précision globale sur toutes les représentations : " << globalAccuracy * 100 << "%" << endl;

        
    /*
    
    // Trouver le k optimal pour KNN
    cout << "=== Détermination du k optimal pour KNN ===" << endl;
    int maxK = 10; // Tester jusqu'à k=10
    int numFolds = 5; // Utiliser 5 folds pour la validation croisée

    int optimalK = findOptimalKWithCrossValidation(filteredImages, "euclidean", maxK, numFolds);
    cout << "Le k optimal est : " << optimalK << endl;

    // Partie 1 : Classification avec KNN
    cout << "=== Classification avec KNN ===" << endl;
    int  KNN = 5;
    KNNClassifier knn(filteredImages, optimalK, "euclidean");
    // Vérifiez l'équilibre des classes
    knn.checkClassBalance(filteredImages);

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
    */
}
