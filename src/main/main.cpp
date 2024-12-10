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
    ConfusionMatrix confusionMatrix(18);

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

    // KMeans
    KMeans kmeans(10, 100, trainImages[0].getDescripteurs().size());
    kmeans.fit(trainImages);

    ConfusionMatrix confusionMatrixKMeans(18);
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

}
