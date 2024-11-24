#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include <vector>
#include <iostream>

class ConfusionMatrix {
private:
    int numClasses;
    std::vector<std::vector<int>> matrix;

public:
    ConfusionMatrix(int numClasses);

    void addPrediction(int trueLabel, int predictedLabel);

    void printMatrix() const;

    const std::vector<std::vector<int>>& getMatrix() const;
};

#endif
