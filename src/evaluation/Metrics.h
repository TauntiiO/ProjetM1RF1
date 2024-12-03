#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>

class Metrics {
public:
    static double accuracy(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> precision(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> recall(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> f1Score(const std::vector<std::vector<int>>& confusionMatrix);

    static void printMetrics(const std::vector<std::vector<int>>& confusionMatrix);
    static void saveMetricsToCSV(const std::vector<std::vector<int>>& confusionMatrix, const std::string& filename);
    // Nouvelle méthode pour lire la matrice de confusion depuis un CSV et calculer les métriques
    static void calculateMetricsFromCSV(const std::string& inputCSV, const std::string& outputCSV);
};

#endif
