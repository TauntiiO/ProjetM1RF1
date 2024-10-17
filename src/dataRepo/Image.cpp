
#include "Image.h"
#include <sstream>
#include <iostream>
#include <iomanip> 

using namespace std;

/**
 * Classe Image
 * 
 * Cette classe représente une image avec :
 * - des descripteurs,
 * - un label,
 * - un type de représentation (GFD, ART, Yang, etc.),
 * - et un chemin d'accès à l'image.
 * 
 * Elle fournit des méthodes pour :
 * - accéder aux descripteurs, au label, au type de représentation et le chemin de l'image,
 * - valider que les descripteurs correspondent au type attendu,
 * - comparer deux images entre elles,
 */


Image::Image(const vector<double>& d, int l, const string& type, const string& path)
    : descripteurs(d), label(l), representationType(type), imagePath(path) {}

//Pour obtenir le descripteurs
const vector<double>& Image::getDescripteurs() const {
    return descripteurs;
}

//Pour obtenir le label
int Image::getLabel() const {
    return label;
}

//Pour obtenir le type de rreprésentation
const string& Image::getRepresentationType() const {
    return representationType;
}

//pour obtenir le chemin de l'image
const string& Image::getImagePath() const {
    return imagePath;
}

//Verifie si le type de l'image correspond au type attendu
bool Image::isValidRepresentation(const string& expectedType) const {
    return representationType == expectedType;
}

//Vérifie si la taille du vecteur des descripteurs est correcte en comparant sa taille à celle qui est attendue
bool Image::validateDescriptors(int expectedSize) const {
    return descripteurs.size() == expectedSize;
}

//Verifie que le nombre de descripteurs(la taille) correspond au type de descripteur
bool Image::validateDescriptorsForType() const {
    if (representationType == "GFD" && descripteurs.size() != 100) {
        return false;
    } else if (representationType == "Yang" && descripteurs.size() != 29) {
        return false;
    } else if (representationType == "Zernike7" && descripteurs.size() != 18) {
        return false;
    } else if (representationType == "ART" && descripteurs.size() != 36) {
        return false;
    }
    return true;
}

//Pour comparer deux images, d'abord selon leur label puis de leurs descripteurs, ensuite chemin
bool Image::operator<(const Image& other) const {
    if (label != other.label) {
        return label < other.label;
    } 
    if (descripteurs != other.descripteurs) {
        return descripteurs < other.descripteurs;
    }
    return imagePath < other.imagePath;
}


/**
 * Vérifie que le label d'une image se trouve bien dans la plage donnée
 * Compare le label de l'image avec les valeurs min et max.
 * Si le label est comprie entre ces deux valeurs il renvoie true sinon false
 */
bool Image::validateLabel(int minLabel, int maxLabel) const {
    return label >= minLabel && label <= maxLabel;
}


string Image::toString() const {
    ostringstream oss;
    oss << "Label : " << label 
        << "\nType de descripteur : " << representationType 
        << "\nChemin de l'image : " << imagePath 
        << "\nNombre de descripteurs : " << descripteurs.size()
        << "\nDescripteurs : ";
    
    oss << fixed << setprecision(4); 
    for (size_t i = 0; i < descripteurs.size(); ++i) {
        oss << descripteurs[i] << " ";
        if ((i + 1) % 10 == 0) oss << "\n";  
    }
    return oss.str();
}

