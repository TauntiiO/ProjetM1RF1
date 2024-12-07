#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>

class Metrics {
public:

    /**
     * Calcule accuracy à partir de la matrice de confusion.
     * Entrée :
     *   - confusionMatrix (const std::vector<std::vector<int>>&) : Matrice de confusion.
     * Sortie (double) : 
     *   - Accuracy.
     */
    static double accuracy(const std::vector<std::vector<int>>& confusionMatrix);

    /**
     *  Calcule la précision pour chaque classe.
     * Entrée :
     *   - confusionMatrix (const std::vector<std::vector<int>>&) : Matrice de confusion.
     * Sortie (std::vector<double>) : 
     *   - Vecteur contenant les précisions par classe.
     */
    static std::vector<double> precision(const std::vector<std::vector<int>>& confusionMatrix);

    /**
     * Calcule le rappel (recall) pour chaque classe.
     * Entrée :
     *   - confusionMatrix (const std::vector<std::vector<int>>&) : Matrice de confusion.
     * Sortie (std::vector<double>) : 
     *   - Vecteur rappels par classe.
     */
    static std::vector<double> recall(const std::vector<std::vector<int>>& confusionMatrix);

    /**
     *  Calcule le score F1 pour chaque classe.
     * Entrée :
     *   - confusionMatrix (const std::vector<std::vector<int>>&) : Matrice de confusion.
     * Sortie (std::vector<double>) : 
     *   - Vecteur des scores F1 par classe.
     */
    static std::vector<double> f1Score(const std::vector<std::vector<int>>& confusionMatrix);

    static void printMetrics(const std::vector<std::vector<int>>& confusionMatrix);
    static void saveMetricsToCSV(const std::vector<std::vector<int>>& confusionMatrix, const std::string& filename);

    /**
     *  Calcule les métriques à partir d'un fichier CSV contenant une matrice de confusion.
     * Entrée :
     *   - inputCSV (const std::string&) : Chemin du fichier CSV d'entrée.
     *   - outputCSV (const std::string&) : Chemin du fichier CSV où sauvegarder les métriques.
     * Sortie : fichier CSV .
     */
    static void calculateMetricsFromCSV(const std::string& inputCSV, const std::string& outputCSV);
};
#endif