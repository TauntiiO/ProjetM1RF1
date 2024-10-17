#include <iostream>
#include "DataRepresentation.h"
#include "Image.h" 
#include <fstream>
#include <filesystem>

using namespace std;

/**
    * @param path : chemin du fichier
*/
DataRepresentation::DataRepresentation(const string& path) : filePath(path) {}

/**
    * Méthode pour lire les descripteurs à partir du fichier.
    * Elle ouvre le fichier, extrait les descripteurs,
    * et les stocke dans un vecteur.
    * 
    * @return true si la lecture est réussie, false sinon
*/
bool DataRepresentation::readFile() {
    data.clear();

    cout << "Ouverture du fichier : " << filePath << endl;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Erreur : Impossible d'ouvrir le fichier " << filePath << endl;
        return false;
    }

    double value;
    while (file >> value) {
        data.push_back(value);
    }

    cout << "Nombre de descripteurs lus : " << data.size() << endl;
    if (file.bad()) {
        cerr << "Erreur lors de la lecture du fichier." << endl;
        return false;
    }

    file.close();
    determineRepresentationType();
    return true;
}

/**
    * Renvoie le vecteur de descripteurs lu.
    * 
    * @return vecteur de descripteurs
*/
const vector<double>& DataRepresentation::getData() const {
    return data;
}


/**
    * Renvoie le type de représentation associé à l'image.
    * 
    * @return type de représentation(en chaine)
*/
const string& DataRepresentation::getRepresentationType() const {
    return representationType;
}

/**
    * Charge les fichiers de descripteurs depuis un répertoire.
    * Limite l'ajout à 12 fichiers par classe.
    * Les images créées à partir des descripteurs sont ajoutées à un vecteur d'images.
    * 
    * @param dirPath : chemin vers le répertoire des descripteurs
    * @param pgmDir : chemin vers le répertoire des fichiers PGM
    * @param images : vecteur où les images seront stockées
    * @return true si tous les fichiers ont été correctement chargés
*/
bool DataRepresentation::loadFromDirectory(const string& dirPath, const string& pgmDir, vector<Image>& images) {
    unordered_map<string, int> sampleCounts;

    for (const auto& entry : filesystem::recursive_directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
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

/**
 * Détermine le type de représentation (GFD, Yang, etc.) en fonction de la taille des descripteurs.
 */
void DataRepresentation::determineRepresentationType() {
    if (data.size() == 16) {
        representationType = "E34";
    } else if (data.size() == 18) {
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
    cout << "Traitement du fichier avec type : " << representationType << endl;
}

/**
    * Extrait le label d'une image à partir de son nom de fichier.
    * Le format est par exemple 's01n005' où '01' est le label.
    * 
    * @param filename : nom du fichier
    * @return le label sous forme d'entier, ou -1 en cas d'erreur
*/
int DataRepresentation::extractLabelFromFilename(const string& filename) {
    if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
        string labelStr = filename.substr(1, 2);
        return stoi(labelStr);
    } else {
        cerr << "Nom de fichier invalide : " << filename << endl;
        return -1;
    }
}
