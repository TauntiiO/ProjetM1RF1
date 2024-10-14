#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

class DataRepresentation {
protected:
    string filePath;  // Stocker le chemin du fichier de descripteurs
    vector<double> data;  // Stocker les descripteurs extraits
    string representationType; // Type de représentation (ART, ZRK, etc.)

public:
    DataRepresentation(const std::string& path) : filePath(path) {}

    bool readFile() {
        ifstream file(filePath);

        // Vérifier si le fichier est ouvert
        if (!file.is_open()) {
            cerr << "Erreur : Impossible d'ouvrir le fichier " << filePath << endl;
            return false;  // Échec
        }

        double value;
        while (file >> value) {
            data.push_back(value);  // Ajouter à la liste des descripteurs
        }

        // Vérifie si le fichier a été lu correctement
        if (file.bad()) {
            cerr << "Erreur lors de la lecture du fichier." << endl;
            return false;  // Échec de la lecture
        }

        file.close();
        determineRepresentationType(); // Déterminer le type de représentation
        return true;
    }

    const vector<double>& getData() const {
        return data;
    }

    const string& getRepresentationType() const {
        return representationType;
    }

    bool loadFromDirectory(const string& dirPath) {
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        // On ne vérifie plus l'extension
        DataRepresentation rep(entry.path().string());
        if (!rep.readFile()) {
            cerr << "Erreur en lisant le fichier : " << entry.path() << endl;
        } else {
            // Ajouter les descripteurs au vecteur data
            data.insert(data.end(), rep.getData().begin(), rep.getData().end());
            determineRepresentationType(); // Déterminez le type après avoir chargé les données
        }
    }
    return true;  
}


    private:
        void determineRepresentationType() {
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
                representationType = "UNKNOWN"; 
            }
        }
};

