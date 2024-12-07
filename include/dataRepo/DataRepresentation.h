#ifndef DATAREPRESENTATION_H
#define DATAREPRESENTATION_H

#include <string>
#include <vector>
#include <unordered_map>

class Image; 

class DataRepresentation {
protected:
    std::string filePath;  
    std::vector<double> data;  
    std::string representationType; 

public:
    /**
     * Entrée :
     *   - path (std::string) : Chemin d'accès au fichier.
     * Sortie : Un objet `DataRepresentation` initialisé avec le chemin spécifié.
     */
    DataRepresentation(const std::string& path);

    /**
     * Lit le fichier et extrait les descripteurs.
     * Entrée : Aucune.
     * Sortie (bool) :
     *   - true si la lecture et l'extraction est réussit.
     *   - false en cas d'erreur.
     */
    bool readFile();

    const std::vector<double>& getData() const;
    const std::string& getRepresentationType() const;

    /**
     * Charge les données à partir d'un répertoire et crée une liste d'objets `Image`.
     * Entrée :
     *   - dirPath (std::string) : Chemin d'accès au répertoire contenant les fichiers de descripteurs.
     *   - pgmDir (std::string) : Répertoire des fichiers PGM associés .
     *   - images (std::vector<Image>&) : Vecteur à remplir avec les objets `Image` créés.
     * Sortie (bool) :
     *   - true si les données sont chargées.
     *   - false sinon.
     */
    bool loadFromDirectory(const std::string& dirPath, const std::string& pgmDir, std::vector<Image>& images);

private:

    /**
     *  Détermine le type de représentation à partir du fichier.
     * Entrée : Aucune.
     * Sortie : modifie `representationType`.
     */
    void determineRepresentationType();

    /**
     * Extrait le label d'une image à partir de son nom de fichier.
     * Entrée :
     *   - filename (std::string) : Nom du fichier.
     * Sortie (int) : Label de l'image extrait du nom du fichier.
     */
    int extractLabelFromFilename(const std::string& filename);
};

#endif
