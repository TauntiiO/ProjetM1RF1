#ifndef CONFUSIONMATRIX_H
#define CONFUSIONMATRIX_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>


class ConfusionMatrix {
private:
    int numClasses; 
    std::vector<std::vector<int>> matrix; 

public:
    /**
     * Constructeur
     * Entrée :
     *   - numClasses (int) : Nombre total de classes.
     * Sortie : Une instance de `ConfusionMatrix` initialisée avec une matrice vide.
     */
    ConfusionMatrix(int numClasses);


    void addPrediction(int trueLabel, int predictedLabel);
    void printMatrix() const;
    const std::vector<std::vector<int>>& getMatrix() const;

    /**
     * @brief Sauvegarde la matrice de confusion au format CSV.
     * Entrée :
     *   - filename (std::string) : Chemin du fichier.
     * Sortie : Fichier CSV est créé ou mis à jour.
     */
    void saveToCSV(const std::string& filename) const;
};

#endif 
