#ifndef DATACOLLECTION_H
#define DATACOLLECTION_H

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <filesystem>
#include "../dataRepo/Image.h"
#include "../dataRepo/DataRepresentation.h"

class DataCollection {
    private:
        std::map<Image, int> dataset; 
        std::string representationType; 
        std::unordered_map<int, int> sampleCounts;
        std::vector<double> minValues; // Minimum descripteurs pour la normalisation
        std::vector<double> maxValues; // Maximum descripteurs pour la normalisation


        /**
         *  Extrait le label d'une image du nom de fichier.
         * Entrée :
         *   - filename (std::string) : Nom Fichier.
         * Sortie (int) : Label de l'image extrait.
         */
        int extractLabelFromFilename(const std::string& filename);

public:
    /**
     * Entrée : Aucune.
     * Sortie : Une instance vide de `DataCollection`.
     */
    DataCollection();

    /**
     * Ajoute un objet `Image` au dataset.
     * Entrée :
     *   - img (Image&) : L'image à ajouter.
     * Sortie (bool) :
     *   - true si l'ajout réussit.
     *   - false si l'image existe déjà.
     */
    bool addDatapoint(const Image& img);

    /**
     * Charge un dataset le répertoire.
     * Parcourt un dossier contenant les fichiers de représentation et remplit le dataset.
     * Entrée :
     *   - dirPath (std::string) : Chemin du répertoire à charger.
     * Sortie (bool) :
     *   - true si le chargement réussit.
     *   - false sinon.
     */
    bool loadDatasetFromDirectory(const std::string& dirPath);

    /**
     * Affiche les informations du data.
     * Entrée : Aucune.
     * Sortie : affichage .
     */
    void printDataset() const;


    std::vector<Image> getImages() const;

    /**
     * @brief Divise un dataset en ensembles d'entraînement et de test.
     * Entrée :
     *   - dataset (std::vector<Image>&) : Dataset à diviser.
     *   - trainSet (std::vector<Image>&) : Référence pour l'ensemble d'entraînement.
     *   - testSet (std::vector<Image>&) : Référence pour l'ensemble de test.
     *   - trainRatio (float) : Proportion des données pour l'entraînement (par défaut 0.8).
     * Sortie : Modifie les ensembles passés par référence.
     */
    void splitDataset(const std::vector<Image>& dataset, std::vector<Image>& trainSet, std::vector<Image>& testSet, float trainRatio = 0.8);

    /**
     * @brief Groupe les images par type de représentation.
     * Entrée :
     *   - images (std::vector<Image>&) : Liste des images à regrouper.
     * Sortie (std::unordered_map<std::string, std::vector<Image>>) :
     *   Une map associant chaque type de représentation à une liste d'images.
     */
    std::unordered_map<std::string, std::vector<Image>> groupImagesByRepresentation(const std::vector<Image>& images) const;
    bool loadTrainTestDatasets(const std::string& trainDir, const std::string& testDir);

    /**
     * @brief Normalise les descripteurs pour les ensembles d'entraînement et de test.
     */
    void normalizeDescriptors();
    const std::vector<Image>& getTrainImages() const;


    const std::vector<Image>& getTestImages() const;
    void computeNormalizationBounds(const std::vector<Image>& images);
    void normalizeDataset(std::vector<Image>& images);
    static void savePRData(const std::string& filename, const std::vector<int>& trueLabels, const std::vector<double>& confidenceScores);
};

#endif 
