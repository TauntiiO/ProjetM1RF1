#include "dataRepo/DataCollection.h"
#include <iostream>
#include <stdexcept>
#include <filesystem> 
#include <algorithm>
#include <random>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
namespace fs = std::filesystem; 

DataCollection::DataCollection() : representationType("") {}

int DataCollection::extractLabelFromFilename(const string& filename) {
    if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
        string labelStr = filename.substr(1, 2);
        return stoi(labelStr);
    } else {
        cerr << "Nom de fichier invalide : " << filename << endl;
        return -1; // Erreur
    }
}


bool DataCollection::addDatapoint(const Image& img) {
    int label = img.getLabel();

    if (label < 1 || label > 18) {
        cerr << "Image ignorée : classe " << label << " hors limite." << endl;
        return false;
    }

    if (dataset.find(img) != dataset.end()) {
        cerr << "Image déjà présente dans le dataset : " << img.getImagePath() << endl;
        return false;  
    }

    if (!img.validateDescriptorsForType()) {
        cerr << "Erreur : Le nombre de descripteurs est incorrect pour le type : " 
             << img.getRepresentationType() << endl;
        return false; 
    }

    dataset[img] = label;  
    sampleCounts[label]++;
    return true;  
}

bool DataCollection::loadDatasetFromDirectory(const string& dirPath) {
    size_t totalImages = 0;
    unordered_map<string, size_t> fileCountsByExtension;

    for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
        if (fs::is_regular_file(entry.path())) {
            string filename = entry.path().filename().string();
            string extension = entry.path().extension().string();

            if (extension == ".txt" || extension == ".yng" || extension == ".gfd" || extension == ".art") {
                if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
                    int label = extractLabelFromFilename(filename);

                    if (label < 1 || label > 18) {
                        continue; 
                    }

                    fileCountsByExtension[extension]++; 

                    try {
                        DataRepresentation rep(entry.path().string());
                        if (rep.readFile()) {
                            vector<double> descriptors = rep.getData();
                            string currentType = rep.getRepresentationType();

                            Image img(descriptors, label, currentType, entry.path().string());
                            if (!addDatapoint(img)) {
                                cerr << "Erreur lors de l'ajout de l'image : " << entry.path() << endl;
                            } else {
                                totalImages++;
                            }
                        }
                    } catch (const runtime_error& e) {
                        cerr << "Erreur lors du traitement de l'image : " << e.what() << endl;
                    }
                }
            }
        }
    }

    cout << "=== Résumé du chargement ===" << endl;
    cout << "Total des images chargées : " << totalImages << endl;
    for (const auto& entry : fileCountsByExtension) {
        cout << "Extension " << entry.first << " : " << entry.second << " fichiers" << endl;
    }
    cout << "============================" << endl;

    return true;
}

// Affichage d'un résumé du dataset
void DataCollection::printDataset() const {
    unordered_map<string, size_t> typeCounts;
    unordered_map<string, size_t> descriptorCountsByType;
    unordered_map<int, size_t> labelCounts;
    unordered_map<string, size_t> descriptorsPerFileByType;
    size_t totalDescriptors = 0;

    for (const auto& entry : dataset) {
        const Image& img = entry.first;
        const string& representationType = img.getRepresentationType();
        size_t numDescriptors = img.getDescripteurs().size();

        typeCounts[representationType]++;
        descriptorCountsByType[representationType] += numDescriptors;
        labelCounts[img.getLabel()]++;
        totalDescriptors += numDescriptors;

        descriptorsPerFileByType[representationType] = numDescriptors; 
    }

    cout << "============================" << endl;
    cout << "=== Résumé du chargement ===" << endl;
    cout << "Total des images chargées : " << dataset.size() << endl;

    for (const auto& typeEntry : typeCounts) {
        cout << "Type de représentation " << typeEntry.first << " : " << typeEntry.second << " fichiers";
        cout << " (" << descriptorCountsByType[typeEntry.first] << " descripteurs au total, "
             << descriptorsPerFileByType[typeEntry.first] << " par fichier)" << endl;
    }

    cout << "--- Répartition par labels ---" << endl;
    for (const auto& labelEntry : labelCounts) {
        cout << "Label " << labelEntry.first << " : " << labelEntry.second << " fichiers" << endl;
    }

    cout << "--- Total des descripteurs ---" << endl;
    cout << "Nombre total de descripteurs (toutes représentations confondues) : " << totalDescriptors << endl;
    cout << "============================" << endl;
}

// Récupération des images
vector<Image> DataCollection::getImages() const {
    vector<Image> images;
    for (const auto& entry : dataset) {
        images.push_back(entry.first);
    }
    return images;
}

// Division du dataset en ensemble d'entraînement et de test
void DataCollection::splitDataset(const std::vector<Image>& dataset, std::vector<Image>& trainSet, std::vector<Image>& testSet, float trainRatio) {
    size_t totalSize = dataset.size();
    size_t trainSize = static_cast<size_t>(totalSize * trainRatio);

    std::vector<Image> shuffledData = dataset;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffledData.begin(), shuffledData.end(), g);

    trainSet.assign(shuffledData.begin(), shuffledData.begin() + trainSize);
    testSet.assign(shuffledData.begin() + trainSize, shuffledData.end());
}


std::unordered_map<std::string, std::vector<Image>> DataCollection::groupImagesByRepresentation(const std::vector<Image>& images) const {
    std::unordered_map<std::string, std::vector<Image>> groupedImages;

    for (const auto& img : images) {
        groupedImages[img.getRepresentationType()].push_back(img);
    }

    return groupedImages;
}


void DataCollection::computeNormalizationBounds(const std::vector<Image>& images) {
    if (images.empty()) return;

    size_t descriptorSize = images[0].getDescripteurs().size();
    minValues.assign(descriptorSize, std::numeric_limits<double>::max());
    maxValues.assign(descriptorSize, std::numeric_limits<double>::lowest());

    for (const auto& img : images) {
        const auto& descriptors = img.getDescripteurs();
        for (size_t i = 0; i < descriptors.size(); ++i) {
            minValues[i] = std::min(minValues[i], descriptors[i]);
            maxValues[i] = std::max(maxValues[i], descriptors[i]);
        }
    }
}

void DataCollection::normalizeDataset(std::vector<Image>& images) {
    for (auto& img : images) {
        auto descriptors = img.getDescripteurs();
        for (size_t i = 0; i < descriptors.size(); ++i) {
            if (maxValues[i] != minValues[i]) {
                descriptors[i] = (descriptors[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            } else {
                descriptors[i] = 0.0; // Cas où les valeurs sont constantes
            }
        }
        img.setDescripteurs(descriptors);
    }
}

void DataCollection::savePRData(const std::string& filename, const std::vector<int>& trueLabels, const std::vector<double>& confidenceScores) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier pour sauvegarder les données PR." << std::endl;
        return;
    }
    outFile << "TrueLabel,ConfidenceScore\n";
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        outFile << trueLabels[i] << "," << confidenceScores[i] << "\n";
    }
    outFile.close();
    std::cout << "Données PR sauvegardées dans : " << filename << std::endl;
}