#include "DataCollection.h"
#include <iostream>
#include <stdexcept>
#include <filesystem> 
#include <algorithm>
#include <random>
#include <unordered_map>

using namespace std;
namespace fs = std::filesystem; 

DataCollection::DataCollection() : representationType("") {}

/**
 * Méthode qui extrait le label d'une image.
 */
int DataCollection::extractLabelFromFilename(const string& filename) {
    if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
        string labelStr = filename.substr(1, 2);
        return stoi(labelStr);
    } else {
        cerr << "Nom de fichier invalide : " << filename << endl;
        return -1; // Erreur
    }
}

/**
 * Méthode qui ajoute un point de données (image) dans le dataset.
 */
bool DataCollection::addDatapoint(const Image& img) {
    int label = img.getLabel();

    if (label < 1 || label > 10) {
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


/**
 * Méthode pour charger un dataset depuis un répertoire.
 */
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

                    if (label < 1 || label > 10) {
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

    // Résumé
    cout << "=== Résumé du chargement ===" << endl;
    cout << "Total des images chargées : " << totalImages << endl;
    for (const auto& entry : fileCountsByExtension) {
        cout << "Extension " << entry.first << " : " << entry.second << " fichiers" << endl;
    }
    cout << "============================" << endl;

    return true;
}


/**
 * Affichage du dataset.
 */

/**
 * Résumé du dataset chargé avec détails sur les descripteurs.
 */
void DataCollection::printDataset() const {
    unordered_map<string, size_t> typeCounts;                // Comptage par type de représentation
    unordered_map<string, size_t> descriptorCountsByType;    // Nombre total de descripteurs par type
    unordered_map<int, size_t> labelCounts;                  // Comptage par label
    unordered_map<string, size_t> descriptorsPerFileByType;  // Nombre de descripteurs par fichier pour chaque type
    size_t totalDescriptors = 0;                             // Nombre total de descripteurs

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


/**
 * Récupération des images.
 */
vector<Image> DataCollection::getImages() const {
    vector<Image> images;
    for (const auto& entry : dataset) {
        images.push_back(entry.first);
    }
    return images;
}


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