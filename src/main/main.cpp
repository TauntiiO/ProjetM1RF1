#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/DataRepresentation.h"
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/Image.h"
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/DataCollection.h"
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/classifier/KNNClassifier.h"

using namespace std;

int main() {
    string representationsDir = "/home/user/Documents/M1/s1/ProjetM1RF1/data/=Signatures"; 

    // Charger le dataset
    DataCollection dataset;
    dataset.loadDatasetFromDirectory(representationsDir); 
    dataset.printDataset(); 

    // Créer un vecteur contenant les images du dataset
    vector<Image> images = dataset.getImages(); 

    // conserve que celles dont le label est compris entre 1 et 10
    vector<Image> filteredImages;
    for (const auto& img : images) {
        int label = img.getLabel();
        if (label >= 1 && label <= 10) {
            filteredImages.push_back(img);
        }
    }
    //k=4
    KNNClassifier knn(filteredImages, 4, "euclidean");

    int correctPredictions = 0;
    map<int, int> labelFrequency; 

    // Boucle pour tester toutes les images filtrées
    for (const auto& img : filteredImages) {
        int predictedLabel = knn.predictLabel(img);
        int realLabel = img.getLabel();

        if (predictedLabel == realLabel) {
            correctPredictions++;
        }

        labelFrequency[predictedLabel]++;
    }

    cout << "Prédictions correctes : " << correctPredictions << " sur " << filteredImages.size() << endl;

    int mostRecognizedLabel = -1;
    int maxCount = 0;
    for (const auto& entry : labelFrequency) {
        if (entry.second > maxCount) {
            mostRecognizedLabel = entry.first;
            maxCount = entry.second;
        }
    }
    
    cout << "Le label le plus souvent reconnu est : " << mostRecognizedLabel << " avec " << maxCount << " prédictions." << endl;

    return 0;
}
