#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include <vector>
#include <string>
#include "../dataRepo/Image.h"  

class KNNClassifier {
    protected:

        std::vector<Image> dataset; 
        int k;                      
        std::string distanceType;    

    public:

        KNNClassifier(const std::vector<Image>& data, int kValue, const std::string& distType = "euclidean");


        double calculateDistance(const Image& img1, const Image& img2) const;
        std::vector<std::pair<double, int>> findKNearestNeighbors(const Image& queryImage) const;
        int predictLabel(const Image& queryImage) const;
        void setK(int kValue);
};

#endif 
