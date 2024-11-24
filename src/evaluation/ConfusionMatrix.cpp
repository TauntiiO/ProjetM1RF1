#include "ConfusionMatrix.h"
#include <iostream>

ConfusionMatrix::ConfusionMatrix(int numClasses)
    : numClasses(numClasses), matrix(numClasses, std::vector<int>(numClasses, 0)) {}

void ConfusionMatrix::addPrediction(int trueLabel, int predictedLabel) {
    matrix[trueLabel - 1][predictedLabel - 1]++; // Assuming labels start from 1
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
