#ifndef DATACOLLECTION_H
#define DATACOLLECTION_H

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <filesystem>
#include "Image.h"
#include "DataRepresentation.h"

class DataCollection {
    private:
        std::map<Image, int> dataset; 
        std::string representationType; 
        std::unordered_map<int, int> sampleCounts;

        int extractLabelFromFilename(const std::string& filename);

    public:

        DataCollection();

        bool addDatapoint(const Image& img);
        bool loadDatasetFromDirectory(const std::string& dirPath);

        void printDataset() const;


        std::vector<Image> getImages() const;

        void splitDataset(const std::vector<Image>& dataset, std::vector<Image>& trainSet, std::vector<Image>& testSet, float trainRatio = 0.8);
        std::unordered_map<std::string, std::vector<Image>> groupImagesByRepresentation(const std::vector<Image>& images) const;


};

#endif 
