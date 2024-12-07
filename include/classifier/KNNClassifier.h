#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include <vector>
#include <string>
#include <utility> 
#include "dataRepo/Image.h"
#include <unordered_map>


class KNNClassifier {
protected:
    std::vector<Image> dataset; 
    int k;                       
    std::string distanceType;    
    std::unordered_map<std::string, std::unordered_map<int, std::vector<std::pair<std::string, double>>>> distancesByRepresentationAndLabel;

public:
    /**
     * Constructeur de KNN.
     * Entrée :
     *   - data (std::vector<Image>&) : Dataset d'entraînement.
     *   - kValue (int) : Nombre de voisins à considérer.
     *   - distType (std::string) : Type de distance utilisé.
     * Sortie : Une instance initialisée de `KNNClassifier`.
     */
    KNNClassifier(const std::vector<Image>& data, int kValue, const std::string& distType);

    /**
     * Calcule la distance entre deux images.
     * Entrée :
     *   - img1 (Image&) : Première image.
     *   - img2 (Image&) : Deuxième image.
     * Sortie (double) : Distance calculée entre les deux images.
     */
    double calculateDistance(const Image& img1, const Image& img2) const;

    /**
     * Trouve les K plus proches voisins pour une image donnée.
     * Entrée :
     *   - queryImage (Image&) : Image de requête.
     * Sortie (std::vector<std::pair<double, int>>) :
     *   - Liste des K plus proches voisins sous forme de paires (distance, label).
     */
    std::vector<std::pair<double, int>> findKNearestNeighbors(const Image& queryImage) const;

    /**
     * Prédit le label d'une image donnée.
     * Entrée :
     *   - queryImage (Image&) : Image de requête.
     * Sortie (int) : Label prédit pour l'image.
     */
    int predictLabel(const Image& queryImage) const;

    /**
     * Prédit le label d'une image avec un score de confiance.
     * Entrée :
     *   - queryImage (Image&) : Image de requête.
     * Sortie (std::pair<int, double>) :
     *   - Pair contenant le label prédit et le score de confiance associé.
     */
    std::pair<int, double> predictLabelWithConfidence(const Image& queryImage) const;


    void setK(int kValue);
    void printDatasetInfo() const;

    /**
     * Vérifie l'équilibre des classes dans le dataset.
     * Entrée :
     *   - data (std::vector<Image>&) : Ensemble d'images à vérifier.
     * Sortie : Affichage des statistiques.
     */
    void checkClassBalance(const std::vector<Image>& data);

    /**
     * Calcule et stocke les distances entre les images pour optimisation.
     * Entrée : Aucune.
     * Sortie : Aucune.
     */
    void calculateAndStoreDistances();

    /**
     * Affiche les distances stockées pour débogage ou vérification.
     * Entrée : Aucune.
     * Sortie : Affichage .
     */
    void printStoredDistances() const;
};

/**
 * Trouve la valeur optimale de K pour le classifieur KNN à l'aide de la validation croisée.
 * Entrée :
 *   - data (std::vector<Image>&) : Dataset à utiliser pour l'évaluation.
 *   - distanceType (std::string) : Type de distance utilisé.
 *   - maxK (int) : Valeur maximale de K à tester.
 *   - numFolds (int) : Nombre de plis pour la validation croisée (par défaut 5).
 * Sortie (int) : Valeur optimale pour K.
 */
int findOptimalKWithCrossValidation(const std::vector<Image>& data, const std::string& distanceType, int maxK, int numFolds = 5);

#endif
