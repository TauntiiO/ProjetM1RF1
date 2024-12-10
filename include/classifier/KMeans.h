#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <string>
#include <utility>
#include "dataRepo/Image.h"
#include <unordered_map>

class KMeans {
public:
    /**
     * Constructeur de KMeans.
     * Entrée :
     *   - numClusters (int) : Nombre de clusters à former.
     *   - numFeatures (int) : Nombre de dimensions dans les descripteurs.
     *   - maxIterations (int) : Nombre maximal d'itérations pour l'algorithme (par défaut 100).
     *   - tolerance (double) : Tolérance pour la convergence (par défaut 1e-4).
     * Sortie : Une instance initialisée de `KMeans`.
     */
    KMeans(int numClusters, int numFeatures, int maxIterations = 100, double tolerance = 1e-4);

    /**
     * Entraîne le modèle KMeans sur un ensemble d'images.
     * Entrée :
     *   - images (std::vector<Image>&) : Ensemble d'images utilisées pour l'entraînement.
     * Sortie : Aucune (met à jour les centroids et leurs labels associés).
     */
    void fit(const std::vector<Image>& images);

    /**
     * Prédit le label d'une image donnée avec un score de confiance.
     * Entrée :
     *   - image (Image&) : Image pour laquelle prédire un label.
     * Sortie (std::pair<int, double>) :
     *   - Pair contenant le label prédit et le score de confiance associé.
     */
    std::pair<int, double> predictLabelWithConfidence(const Image& image) const;

private:
    int numClusters;             // Nombre de clusters (classes) à former.
    int numFeatures;             // Nombre de dimensions dans les descripteurs des images.
    int maxIterations;           // Nombre maximal d'itérations autorisées pour la convergence.
    double tolerance;            // Seuil de tolérance pour considérer que les centroids ont convergé.

    /**
     * Centroids calculés pour chaque représentation (type de descripteur).
     * Structure : {nom de la représentation -> vecteurs des centroids}.
     */
    std::unordered_map<std::string, std::vector<std::vector<double>>> centroidsByRepresentation;

    /**
     * Labels associés aux centroids pour chaque représentation.
     * Structure : {nom de la représentation -> labels associés}.
     */
    std::unordered_map<std::string, std::vector<int>> centroidLabelsByRepresentation;

    /**
     * Calcule la distance entre deux vecteurs.
     * Entrée :
     *   - a (std::vector<double>&) : Premier vecteur.
     *   - b (std::vector<double>&) : Second vecteur.
     * Sortie (double) : Distance calculée entre les deux vecteurs.
     */
    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const;

    /**
     * Calcule un score de confiance pour une prédiction.
     * Entrée :
     *   - features (std::vector<double>&) : Descripteur de l'image.
     *   - closestCluster (int) : Index du cluster le plus proche.
     * Sortie (double) : Score de confiance pour la prédiction.
     */
    double calculateConfidence(const std::vector<double>& features, int closestCluster) const;

    /**
     * Associe des labels aux centroids à partir des données d'entraînement.
     * Entrée :
     *   - images (std::vector<Image>&) : Ensemble des images utilisées pour l'entraînement.
     *   - assignments (std::vector<int>&) : Assignations des images aux clusters.
     *   - centroids (std::vector<std::vector<double>>&) : Centroids calculés pendant l'entraînement.
     * Sortie : Aucune (met à jour les labels des centroids).
     */
    void associateLabelsToCentroids(const std::vector<Image>& images, const std::vector<int>& assignments, const std::vector<std::vector<double>>& centroids);
};

#endif // KMEANS_H
