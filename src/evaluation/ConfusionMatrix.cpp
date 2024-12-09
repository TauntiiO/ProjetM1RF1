#include "evaluation/ConfusionMatrix.h"
#include <fstream>
#include <iostream>

ConfusionMatrix::ConfusionMatrix(int numClasses)
    : numClasses(numClasses), matrix(numClasses, std::vector<int>(numClasses, 0)) {}

void ConfusionMatrix::addPrediction(int trueLabel, int predictedLabel) {
    matrix[trueLabel - 1][predictedLabel - 1]++;
}

void ConfusionMatrix::printMatrix() const {
    std::cout << "\n=== Confusion Matrix ===\n";
    for (const auto& row : matrix) {
        for (int count : row) {
            std::cout << count << " ";
        }
        std::cout << std::endl;
    }
}

const std::vector<std::vector<int>>& ConfusionMatrix::getMatrix() const {
    return matrix;
}

void ConfusionMatrix::saveToCSV(const std::string& filename) const {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier pour écrire la matrice de confusion." << std::endl;
        return;
    }

    // Écrire l'en-tête
    outFile << ",";
    for (int i = 0; i < numClasses; ++i) {
        outFile << "Class" << (i + 1) << (i == numClasses - 1 ? "\n" : ",");
    }

    // Écrire les lignes de la matrice
    for (int i = 0; i < numClasses; ++i) {
        outFile << "Class" << (i + 1) << ",";
        for (int j = 0; j < numClasses; ++j) {
            outFile << matrix[i][j] << (j == numClasses - 1 ? "\n" : ",");
        }
    }

    outFile.close();
    std::cout << "Matrice de confusion sauvegardée au format CSV dans : " << filename << std::endl;
}
