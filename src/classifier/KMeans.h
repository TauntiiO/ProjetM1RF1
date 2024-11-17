#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include "../dataRepo/DataRepresentation.h"
#include "../dataRepo/Image.h"

class KMeans {
private:
    int k;  // Nombre de clusters
    std::vector<std::vector<double>> centroids;  // Centroids des clusters
    std::vector<int> labels;  // Labels pour chaque point

public:
    KMeans(int k);
    void fit(const std::vector<Image>& images);
    void initializeCentroids(const std::vector<Image>& images);
    void assignClusters(const std::vector<Image>& images);
    std::vector<double> updateCentroid(const std::vector<Image>& images, int clusterId);
    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
    void printClusters(const std::vector<Image>& images) const;
    
};

#endif // KMEANS_H
