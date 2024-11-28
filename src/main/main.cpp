#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <random>
#include <algorithm> 
#include "../dataRepo/DataRepresentation.h"
#include "../dataRepo/Image.h"
#include "../dataRepo/DataCollection.h"
#include "../classifier/KNNClassifier.h"
#include "../classifier/KMeans.h"
#include "../classifier/KNNClassifier.h"
#include "../evaluation/ConfusionMatrix.h"
#include "../evaluation/Metrics.h"

using namespace std;

//créer k folds
vector<vector<Image>> createKFolds(const vector<Image>& data, int k) {
    vector<vector<Image>> folds(k);
    vector<Image> shuffledData = data;
    random_device rd;
    mt19937 g(rd());
    shuffle(shuffledData.begin(), shuffledData.end(), g);

    for (size_t i = 0; i < shuffledData.size(); ++i) {
        folds[i % k].push_back(shuffledData[i]);
    }
    return folds;
}

int main() {
    vector<string> representationDirs = {
        "./data/=Signatures/=ART",
        "./data/=Signatures/=Yang",
        "./data/=Signatures/=GFD",
        "./data/=Signatures/=Zernike7"
    };

    for (const auto& representationDir : representationDirs) {
        string trainDir = representationDir + "/train";
        string testDir = representationDir + "/test";

        DataCollection trainDataset;
        trainDataset.loadDatasetFromDirectory(trainDir);
        vector<Image> trainImages = trainDataset.getImages();

        DataCollection testDataset;
        testDataset.loadDatasetFromDirectory(testDir);
        vector<Image> testImages = testDataset.getImages();

        cout << "\n=== Évaluation pour la représentation : " << representationDir << " ===" << endl;
        cout << "Nombre d'images dans l'ensemble d'entraînement : " << trainImages.size() << endl;
        cout << "Nombre d'images dans l'ensemble de test : " << testImages.size() << endl;

        if (trainImages.empty() || testImages.empty()) {
            cout << "Données insuffisantes pour la représentation : " << representationDir << ". Passé.\n";
            continue;
        }

        cout << "Normalisation des descripteurs...\n";
        vector<Image> normalizedTrainImages = trainImages;
        vector<Image> normalizedTestImages = testImages;

        size_t numDescriptors = trainImages[0].getDescripteurs().size();
        vector<double> minValues(numDescriptors, numeric_limits<double>::max());
        vector<double> maxValues(numDescriptors, numeric_limits<double>::lowest());

        for (const auto& img : trainImages) {
            const auto& descripteurs = img.getDescripteurs();
            for (size_t i = 0; i < descripteurs.size(); ++i) {
                minValues[i] = min(minValues[i], descripteurs[i]);
                maxValues[i] = max(maxValues[i], descripteurs[i]);
            }
        }

        for (auto& img : normalizedTrainImages) {
            vector<double> normalizedDescriptors(numDescriptors);
            const auto& descripteurs = img.getDescripteurs();
            for (size_t i = 0; i < descripteurs.size(); ++i) {
                if (maxValues[i] != minValues[i]) {
                    normalizedDescriptors[i] = (descripteurs[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                } else {
                    normalizedDescriptors[i] = 0.0;
                }
            }
            img.setDescripteurs(normalizedDescriptors);
        }

        for (auto& img : normalizedTestImages) {
            vector<double> normalizedDescriptors(numDescriptors);
            const auto& descripteurs = img.getDescripteurs();
            for (size_t i = 0; i < descripteurs.size(); ++i) {
                if (maxValues[i] != minValues[i]) {
                    normalizedDescriptors[i] = (descripteurs[i] - minValues[i]) / (maxValues[i] - minValues[i]);
                } else {
                    normalizedDescriptors[i] = 0.0;
                }
            }
            img.setDescripteurs(normalizedDescriptors);
        }

        cout << "\n=== Validation croisée KNN ===" << endl;
        int kFolds = 5;
        vector<vector<Image>> folds = createKFolds(normalizedTrainImages, kFolds);

        double bestAccuracy = 0.0;
        int bestK = 3;//defaut

        for (int candidateK = 1; candidateK <= 10; ++candidateK) {
            double totalFoldAccuracy = 0.0;

            for (int i = 0; i < kFolds; ++i) {
                vector<Image> foldTrainImages, foldTestImages = folds[i];
                for (int j = 0; j < kFolds; ++j) {
                    if (j != i) {
                        foldTrainImages.insert(foldTrainImages.end(), folds[j].begin(), folds[j].end());
                    }
                }

                KNNClassifier knn(foldTrainImages, candidateK, "euclidean");

                int correctPredictions = 0;
                for (const auto& testImage : foldTestImages) {
                    int predictedLabel = knn.predictLabel(testImage);
                    if (predictedLabel == testImage.getLabel()) {
                        correctPredictions++;
                    }
                }

                double foldAccuracy = static_cast<double>(correctPredictions) / foldTestImages.size();
                totalFoldAccuracy += foldAccuracy;
            }

            double averageFoldAccuracy = totalFoldAccuracy / kFolds;
            cout << "K=" << candidateK << ", Précision moyenne : " << averageFoldAccuracy * 100 << "%" << endl;

            if (averageFoldAccuracy > bestAccuracy) {
                bestAccuracy = averageFoldAccuracy;
                bestK = candidateK;
            }
        }

        cout << "Meilleur K pour KNN : " << bestK << " avec une précision moyenne de : " << bestAccuracy * 100 << "%" << endl;

        cout << "\n=== Classification avec KNN (K=" << bestK << ") ===" << endl;
        KNNClassifier knn(normalizedTrainImages, bestK, "euclidean");
        ConfusionMatrix confusionMatrix(10);

        int correctPredictions = 0;
        for (const auto& testImage : normalizedTestImages) {
            int predictedLabel = knn.predictLabel(testImage);
            confusionMatrix.addPrediction(testImage.getLabel(), predictedLabel);
            if (predictedLabel == testImage.getLabel()) {
                correctPredictions++;
            }
        }

        confusionMatrix.printMatrix();

        double accuracy = static_cast<double>(correctPredictions) / normalizedTestImages.size();
        cout << "Précision pour la représentation " << representationDir << " : " << accuracy * 100 << "%" << endl;

        Metrics::printMetrics(confusionMatrix.getMatrix());

        cout << "\n=== Clustering avec K-Means ===" << endl;
        KMeans kmeans(10); 
        kmeans.fit(normalizedTrainImages);

        ConfusionMatrix kmeansConfusionMatrix(10);
        correctPredictions = 0;

        for (const auto& testImage : normalizedTestImages) {
            int predictedCluster = kmeans.predictCluster(testImage);
            kmeansConfusionMatrix.addPrediction(testImage.getLabel(), predictedCluster);

            if (predictedCluster == testImage.getLabel()) {
                correctPredictions++;
            }
        }

        kmeansConfusionMatrix.printMatrix();

        accuracy = static_cast<double>(correctPredictions) / normalizedTestImages.size();
        cout << "Précision K-Means pour la représentation " << representationDir << " : " << accuracy * 100 << "%" << endl;
    }








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
