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
};

#endif 
