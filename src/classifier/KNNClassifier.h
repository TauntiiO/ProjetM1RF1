#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include <vector>
#include <string>
#include <utility> 
#include "../dataRepo/Image.h"
#include <unordered_map>


class KNNClassifier {
protected:
    std::vector<Image> dataset; 
    int k;                       
    std::string distanceType;    
    std::unordered_map<std::string, std::unordered_map<int, std::vector<std::pair<std::string, double>>>> distancesByRepresentationAndLabel;

public:
    KNNClassifier(const std::vector<Image>& data, int kValue, const std::string& distType);
    double calculateDistance(const Image& img1, const Image& img2) const;
    std::vector<std::pair<double, int>> findKNearestNeighbors(const Image& queryImage) const;
    int predictLabel(const Image& queryImage) const;
    std::pair<int, double> predictLabelWithConfidence(const Image& queryImage) const; // Ajout
    void setK(int kValue);
    void printDatasetInfo() const;
    void checkClassBalance(const std::vector<Image>& data);
    void calculateAndStoreDistances();

    void printStoredDistances() const;
};

int findOptimalKWithCrossValidation(const std::vector<Image>& data, const std::string& distanceType, int maxK, int numFolds = 5);

#endif
