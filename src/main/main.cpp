#include "dataRepo/DataRepresentation.h"
#include "dataRepo/Image.h"
#include "dataRepo/DataCollection.h"
#include "classifier/KNNClassifier.h"
#include "classifier/KMeans.h"
#include "evaluation/ConfusionMatrix.h"
#include "evaluation/Metrics.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

/**
 * Récupère le répertoire racine du projet à partir de l'exécutable.
 * Entrée :
 *   - argv0 (const char*) : Chemin de l'exécutable.
 * Sortie :
 *   - std::string : Chemin du répertoire racine.
 */
std::string getProjectRootDir(const char* argv0) {
    try {
        fs::path execPath = fs::path(argv0);
        if (!execPath.is_absolute()) {
            execPath = fs::current_path() / execPath;
        }
        execPath = execPath.parent_path();
        if (!fs::exists(execPath)) {
            cerr << "Erreur : Le chemin spécifié n'existe pas : " << execPath << endl;
            return {};
        }
        return fs::canonical(execPath).string();
    } catch (const fs::filesystem_error& e) {
        cerr << "Erreur lors de la résolution du chemin : " << e.what() << endl;
        return {};
    }
}


void processRepresentation(const string& representationDir, const string& confusionDir, const string& metricsDir, const string& prDataDir) {
    string trainDir = representationDir + "/train2";
    string testDir = representationDir + "/test2";

    if (!fs::exists(trainDir) || !fs::exists(testDir)) {
        cerr << "Erreur : Les répertoires train/test sont manquants pour : " << representationDir << endl;
        return;
    }

    DataCollection trainDataset, testDataset;
    trainDataset.loadDatasetFromDirectory(trainDir);
    testDataset.loadDatasetFromDirectory(testDir);

    vector<Image> trainImages = trainDataset.getImages();
    vector<Image> testImages = testDataset.getImages();

    if (trainImages.empty() || testImages.empty()) {
        cout << "Données insuffisantes pour la représentation : " << representationDir << ". Passé.\n";
        return;
    }

    // Normalisation
    trainDataset.computeNormalizationBounds(trainImages);
    trainDataset.normalizeDataset(trainImages);
    trainDataset.normalizeDataset(testImages);

    // KNN
    KNNClassifier knn(trainImages, 1, "euclidean");
    ConfusionMatrix confusionMatrix(18);  // Changement pour 18 classes

    for (const auto& testImage : testImages) {
        int predictedLabel;
        double confidenceScore;
        tie(predictedLabel, confidenceScore) = knn.predictLabelWithConfidence(testImage);
        confusionMatrix.addPrediction(testImage.getLabel(), predictedLabel);
    }

    string representationName = fs::path(representationDir).filename().string();
    string confusionCSV = confusionDir + "/" + representationName + "_confusion_matrix.csv";
    confusionMatrix.saveToCSV(confusionCSV);

    string metricsCSV = metricsDir + "/" + representationName + "_metrics.csv";
    Metrics::calculateMetricsFromCSV(confusionCSV, metricsCSV);

    KNNClassifier knnWithFixedK(trainImages, 12, "euclidean");
    vector<int> prTrueLabels;
    vector<double> prConfidenceScores;

    for (const auto& testImage : testImages) {
        int predictedLabel;
        double confidenceScore;
        tie(predictedLabel, confidenceScore) = knnWithFixedK.predictLabelWithConfidence(testImage);
        prTrueLabels.push_back(testImage.getLabel());
        prConfidenceScores.push_back(confidenceScore);
    }

    string prFilename = prDataDir + "/" + representationName + "_pr_data.csv";
    trainDataset.savePRData(prFilename, prTrueLabels, prConfidenceScores);

    KMeans kmeans(10, 100, trainImages[0].getDescripteurs().size());
    kmeans.fit(trainImages);

    ConfusionMatrix confusionMatrixKMeans(18);  // Changement pour 18 classes
    vector<int> prTrueLabelsKMeans;
    vector<double> prConfidenceScoresKMeans;

    for (const auto& testImage : testImages) {
        int predictedLabel;
        double confidenceScore;
        tie(predictedLabel, confidenceScore) = kmeans.predictLabelWithConfidence(testImage);
        confusionMatrixKMeans.addPrediction(testImage.getLabel(), predictedLabel);
        prTrueLabelsKMeans.push_back(testImage.getLabel());
        prConfidenceScoresKMeans.push_back(confidenceScore);
    }

    string confusionCSVKMeans = confusionDir + "/" + representationName + "_KMeans_confusion_matrix.csv";
    confusionMatrixKMeans.saveToCSV(confusionCSVKMeans);

    string metricsCSVKMeans = metricsDir + "/" + representationName + "_KMeans_metrics.csv";
    Metrics::calculateMetricsFromCSV(confusionCSVKMeans, metricsCSVKMeans);

    string prFilenameKMeans = prDataDir + "/" + representationName + "_KMeans_pr_data.csv";
    trainDataset.savePRData(prFilenameKMeans, prTrueLabelsKMeans, prConfidenceScoresKMeans);

    cout << "Traitement terminé pour : " << representationName << endl;
}


int main(int argc, char* argv[]) {
    if (argc < 1 || argv[0] == nullptr) {
        cerr << "Erreur : Impossible de déterminer le chemin du binaire.\n";
        return 1;
    }

    std::string rootDir = getProjectRootDir(argv[0]);
    if (rootDir.empty()) {
        cerr << "Erreur : Impossible de déterminer le répertoire racine.\n";
        return 1;
    }

    rootDir += "/data/=Signatures";

    vector<string> representationDirs = {
        rootDir + "/=ART",
        rootDir + "/=Yang",
        rootDir + "/=GFD",
        rootDir + "/=Zernike7"
    };

    // Configuration pour les 18 classes
    string resultsBaseDir = "results/18_classes";
    string confusionDir = resultsBaseDir + "/confusion_matrices";
    string metricsDir = resultsBaseDir + "/metrics";
    string prDataDir = resultsBaseDir + "/precision_recall_data";

    if (!fs::exists(confusionDir)) fs::create_directories(confusionDir);
    if (!fs::exists(metricsDir)) fs::create_directories(metricsDir);
    if (!fs::exists(prDataDir)) fs::create_directories(prDataDir);

    for (const auto& representationDir : representationDirs) {
        processRepresentation(representationDir, confusionDir, metricsDir, prDataDir);
    }

    cout << "Toutes les matrices de confusion, métriques, et données PR ont été calculées et sauvegardées dans : " 
         << resultsBaseDir << endl;
    return 0;









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
