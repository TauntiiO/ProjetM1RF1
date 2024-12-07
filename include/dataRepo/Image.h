#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>

class Image {
    private:
    std::vector<double> descripteurs;
    int label;
    std::string representationType;
    std::string imagePath;

public:

    Image(); 

    /**
     * Entrée : 
     *   - descripteurs (std::vector<double>) : Liste des descripteurs.
     *   - label (int) : Classe associée.
     *   - type (std::string) : Type de représentation (GFD, ART etc).
     *   - path (std::string) : Accèe à l'image.
     * Sortie : Un objet `Image` initialisé avec les valeurs fournies.
     */
    Image(const std::vector<double>& descripteurs, int label, const std::string& type, const std::string& path);

    /**
     *  Accède aux descripteurs de l'image.
     * Entrée : Aucune.
     * Sortie (std::vector<double>): Référence vers le vecteur de descripteurs .
     */
    const std::vector<double>& getDescripteurs() const;

    /**
     * Entrée :
     *   - newDescripteurs (std::vector<double>) : Nouveau vecteur de descripteurs.
     * Sortie : Aucune.
     */
    void setDescripteurs(const std::vector<double>& newDescripteurs);

    int getLabel() const;
    const std::string& getRepresentationType() const;
    const std::string& getImagePath() const;

    /**
     * Vérifie si le type de représentation de l'image correspond à celui attendu.
     * Entrée : 
     *   - expectedType (std::string) : Type de représentation attendu.
     * Sortie (bool) :
     *   - true si le type correspond.
     *   - false sinon.
     */
    bool isValidRepresentation(const std::string& expectedType) const;

    /**
     * Vérifie si la taille des descripteurs correspond à celle attendue.
     * Entrée : 
     *   - expectedSize (int) : Taille attendue.
     * Sortie (bool) :
     *   - true si la taille correspond.
     *   - false sinon.
     */
    bool validateDescriptors(int expectedSize) const;

    /**
     * Valide les descripteurs en fonction du type de représentation de l'image.
     * Entrée : Aucune.
     * Sortie (bool):
     *   - true si les descripteurs correspondent au type.
     *   - false sinon.
     */
    bool validateDescriptorsForType() const;

    /**
     * Vérifie si le label est compris dans un intervalle.
     * Entrée :
     *   - minLabel (int) : Borne inférieure incluse.
     *   - maxLabel (int) : Borne supérieure incluse.
     * Sortie (bool):
     *   - true si le label est dans l'intervalle.
     *   - false sinon.
     */
    bool validateLabel(int minLabel, int maxLabel) const;

    /**
     * Compare cette image avec une autre.
     * Entrée :
     *   - other (Image) : L'image à comparer.
     * Sortie (bool):
     *   - true si cette image est inférieure à l'autre.
     *   - false sinon.
     */
    bool operator<(const Image& other) const;

    /**
     * fait un affichage textuelle de l'image.
     * Entrée : Aucune.
     * Sortie (std::string): Chaîne décrivant l'image 
     */
    std::string toString() const;
};


#endif