#include "Metrics.h"
#include <iostream>

double Metrics::accuracy(const std::vector<std::vector<int>>& confusionMatrix) {
    int correct = 0, total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        for (size_t j = 0; j < confusionMatrix.size(); ++j) {
            total += confusionMatrix[i][j];
            if (i == j) correct += confusionMatrix[i][j];
        }
    }
    return static_cast<double>(correct) / total;
}

std::vector<double> Metrics::precision(const std::vector<std::vector<int>>& confusionMatrix) {
    std::vector<double> precision(confusionMatrix.size(), 0.0);
    for (size_t j = 0; j < confusionMatrix.size(); ++j) {
        int truePositive = confusionMatrix[j][j];
        int predictedPositive = 0;
        for (size_t i = 0; i < confusionMatrix.size(); ++i) {
            predictedPositive += confusionMatrix[i][j];
        }
        precision[j] = predictedPositive ? static_cast<double>(truePositive) / predictedPositive : 0.0;
    }
    return precision;
}

std::vector<double> Metrics::recall(const std::vector<std::vector<int>>& confusionMatrix) {
    std::vector<double> recall(confusionMatrix.size(), 0.0);
    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        int truePositive = confusionMatrix[i][i];
        int actualPositive = 0;
        for (size_t j = 0; j < confusionMatrix.size(); ++j) {
            actualPositive += confusionMatrix[i][j];
        }
        recall[i] = actualPositive ? static_cast<double>(truePositive) / actualPositive : 0.0;
    }
    return recall;
}

std::vector<double> Metrics::f1Score(const std::vector<std::vector<int>>& confusionMatrix) {
    std::vector<double> precisionValues = precision(confusionMatrix);
    std::vector<double> recallValues = recall(confusionMatrix);
    std::vector<double> f1(confusionMatrix.size(), 0.0);
    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        if (precisionValues[i] + recallValues[i] != 0) {
            f1[i] = 2 * (precisionValues[i] * recallValues[i]) / (precisionValues[i] + recallValues[i]);
        }
    }
    return f1;
}

void Metrics::printMetrics(const std::vector<std::vector<int>>& confusionMatrix) {
    double acc = accuracy(confusionMatrix);
    std::vector<double> precisionValues = precision(confusionMatrix);
    std::vector<double> recallValues = recall(confusionMatrix);
    std::vector<double> f1Values = f1Score(confusionMatrix);

    std::cout << "\n=== Metrics ===\n";
    std::cout << "Accuracy: " << acc * 100 << "%\n";
    for (size_t i = 0; i < precisionValues.size(); ++i) {
        std::cout << "Class " << i + 1 << ": Precision = " << precisionValues[i] * 100
                  << "%, Recall = " << recallValues[i] * 100
                  << "%, F1-Score = " << f1Values[i] * 100 << "%\n";
    }
}
