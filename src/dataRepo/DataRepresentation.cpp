#include <iostream>
#include "dataRepo/DataRepresentation.h"
#include "dataRepo/Image.h" 
#include <fstream>
#include <filesystem>
namespace filesystem = std::filesystem;

using namespace std;

DataRepresentation::DataRepresentation(const string& path) : filePath(path) {}


bool DataRepresentation::readFile() {
    data.clear();
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Erreur : Impossible d'ouvrir le fichier " << filePath << endl;
        return false;
    }

    double value;
    while (file >> value) {
        data.push_back(value);
    }

    if (file.bad()) {
        cerr << "Erreur lors de la lecture du fichier : " << filePath << endl;
        return false;
    }

    file.close();
    determineRepresentationType();
    return true;
}

const vector<double>& DataRepresentation::getData() const {
    return data;
}

const string& DataRepresentation::getRepresentationType() const {
    return representationType;
}

bool DataRepresentation::loadFromDirectory(const string& dirPath, const string& pgmDir, vector<Image>& images) {
    unordered_map<string, int> sampleCounts;

    for (const auto& entry : filesystem::recursive_directory_iterator(dirPath)) {
        if (filesystem::is_regular_file(entry.path())) {
            string filename = entry.path().filename().string();
            if (filename == ".DS_Store") continue;

            string classLabel = filename.substr(0, 3);
            if (sampleCounts[classLabel] >= 12) continue;

            DataRepresentation rep(entry.path().string());
            if (rep.readFile()) {
                cout << "Nombre de descripteurs chargés pour " << filename << " : " << rep.getData().size() << endl;
                vector<double> limitedData(rep.getData().begin(), rep.getData().begin() + min(12, static_cast<int>(rep.getData().size())));

                string imagePath = pgmDir + "/" + filename.substr(0, 7) + ".pgm";
                int label = extractLabelFromFilename(filename);

                Image img(limitedData, label, rep.getRepresentationType(), imagePath);
                images.push_back(img);
                sampleCounts[classLabel]++;
            } else {
                cerr << "Erreur en lisant le fichier : " << entry.path() << endl;
            }
        }
    }
    return true;
}

void DataRepresentation::determineRepresentationType() {

    if (data.size() == 18) {
        representationType = "Zernike7";
    } else if (data.size() == 100) {
        representationType = "GFD";
    } else if (data.size() == 36) {
        representationType = "ART";
    } else if (data.size() == 29) {
        representationType = "Yang";
    } else {
        cerr << "Type de descripteurs inconnu avec " << data.size() << " descripteurs." << endl;
        representationType = "UNKNOWN";
    }
}

int DataRepresentation::extractLabelFromFilename(const string& filename) {
    if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
        string labelStr = filename.substr(1, 2);
        return stoi(labelStr);
    } else {
        cerr << "Nom de fichier invalide : " << filename << endl;
        return -1;
    }
}
