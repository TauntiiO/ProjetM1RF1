#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <unordered_map>
#include <stdexcept>
#include "Image.h"
#include "DataRepresentation.h"

using namespace std;

class DataCollection {

    private:
        map<Image, int> dataset; 
        string representationType; 
        unordered_map<int, int> sampleCounts;

    private:
        /**
            * Méthode qui extrait le label d'une image.
            * 
            * Si le format du nom est valide (ex: s01n005), la méthode extrait
            * les deux chiffres après le 's' pour déterminer le label. 
            * Si le format est incorrect, elle renvoie une erreur.
        */
        int extractLabelFromFilename(const string& filename) {
            if (filename.length() >= 7 && filename[0] == 's' && filename[3] == 'n') {
                string labelStr = filename.substr(1, 2);
                return stoi(labelStr);
            } else {
                cerr << "Nom de fichier invalide : " << filename << endl;
                return -1; //erreur
            }
        }

    public:
        DataCollection() : representationType("") {}

    /**
        * Méthode qui ajoute un point de données (image) dans le dataset.
        * 
        * Elle vérifie si l'image existe déjà pour éviter les doublons, valide la taille des descripteurs
        * en fonction du type de représentation, puis ajoute l'image au dataset si elle est correcte.
        * 
        * @param img : l'image à ajouter
        * @return true si l'ajout est réussi, false en cas d'erreur (doublon ou problème de validation)
    */
    bool addDatapoint(const Image& img) {
        int label = img.getLabel();
        string currentType = img.getRepresentationType();  

        cout << "Ajout de l'image avec type de représentation : " << currentType << endl;

        
        cerr << "Classe : " << label << ", Nombre d'échantillons : " << sampleCounts[label] << endl;

        
        if (dataset.find(img) != dataset.end()) {
            cerr << "Image déjà présente dans le dataset : " << img.getImagePath() << endl;
            return false;  
        }

        
        if (!img.validateDescriptorsForType()) {
            cerr << "Erreur : Le nombre de descripteurs est incorrect pour le type de représentation : " << img.getRepresentationType() << endl;
            return false; 
        }

        dataset[img] = label;  
        sampleCounts[label]++; 

        return true;  
    }

    /**
        * Méthode pour charger un dataset depuis un répertoire.
        * 
        * Parcourt tous les fichiers d'un répertoire, vérifie leur extension, puis 
        * tente d'extraire les descripteurs pour chaque fichier. 
        * Si le fichier est valide, une image est créée et ajoutée au dataset via la méthode `addDatapoint`.
        * 
        * @param dirPath : le chemin vers le répertoire contenant les fichiers à traiter
        * @return true si tous les fichiers sont correctement traités
    */
    bool loadDatasetFromDirectory(const string& dirPath) {
        for (const auto& entry : filesystem::recursive_directory_iterator(dirPath)) {
            if (entry.is_regular_file()) {
                string extension = entry.path().extension().string();

                if (extension == ".zrk.txt" || extension == ".yng" || extension == ".gfd" || extension == ".E34" || extension == ".art") {
                    cout << "Fichier trouvé avec extension valide : " << extension << endl;

                    try {
                        DataRepresentation rep(entry.path().string()); 
                        if (rep.readFile()) { 
                            vector<double> descriptors = rep.getData();  
                            string currentType = rep.getRepresentationType();  

                            int label = extractLabelFromFilename(entry.path().filename().string()); 

                        
                            Image img(descriptors, label, currentType, entry.path().string());
                            if (!addDatapoint(img)) {
                                cerr << "Erreur lors de l'ajout de l'image : " << entry.path() << endl;
                            }
                        } else {
                            cerr << "Erreur lors de la lecture des descripteurs à partir du fichier : " << entry.path() << endl;
                        }
                    } catch (const runtime_error& e) {
                        cerr << "Erreur lors du traitement de l'image : " << e.what() << endl;
                    }
                } else {
                    cerr << "Extension de fichier inattendue : " << extension << endl;
                }
            }
        }
        return true; 
    }

    // Affichage du dataset
    void printDataset() const {
        for (const auto& entry : dataset) {
            cout << entry.first.toString() << endl; 
        }
    }
};
