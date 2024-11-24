
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

Image::Image() 
    : descripteurs{}, label(0), representationType(""), imagePath("") {}

Image::Image(const std::vector<double>& d, int l, const std::string& type, const std::string& path)
    : descripteurs(d), label(l), representationType(type), imagePath(path) {}

const vector<double>& Image::getDescripteurs() const {
    return descripteurs;
}

int Image::getLabel() const {
    return label;
}

const string& Image::getRepresentationType() const {
    return representationType;
}

const string& Image::getImagePath() const {
    return imagePath;
}

//Verifie si le type de l'image correspond au type 
bool Image::isValidRepresentation(const string& expectedType) const {
    return representationType == expectedType;
}

//Vérifie si la taille du vecteur des descripteurs est correcte en comparant sa taille à celle qui est attendue
bool Image::validateDescriptors(int expectedSize) const {
    return descripteurs.size() == expectedSize;
}

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
        << ", Type : " << representationType 
        << ", Descripteurs : ";

    for (size_t i = 0; i < min(descripteurs.size(), size_t(5)); ++i) {
        oss << descripteurs[i] << " ";
    }
    if (descripteurs.size() > 5) {
        oss << "... (" << descripteurs.size() << " au total)";
    }
    return oss.str();
}



