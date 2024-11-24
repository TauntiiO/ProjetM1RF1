#ifndef METRICS_H
#define METRICS_H

#include <vector>

class Metrics {
public:
    static double accuracy(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> precision(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> recall(const std::vector<std::vector<int>>& confusionMatrix);

    static std::vector<double> f1Score(const std::vector<std::vector<int>>& confusionMatrix);

    static void printMetrics(const std::vector<std::vector<int>>& confusionMatrix);
};

#endif
