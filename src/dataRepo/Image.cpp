#include "dataRepo/Image.h"
#include <sstream>
#include <iostream>
#include <iomanip> 

using namespace std;

Image::Image() 
    : descripteurs{}, label(0), representationType(""), imagePath("") {}

Image::Image(const std::vector<double>& d, int l, const std::string& type, const std::string& path)
    : descripteurs(d), label(l), representationType(type), imagePath(path) {}

const vector<double>& Image::getDescripteurs() const {
    return descripteurs;
}

void Image::setDescripteurs(const std::vector<double>& newDescripteurs) {
    descripteurs = newDescripteurs;
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

bool Image::isValidRepresentation(const string& expectedType) const {
    return representationType == expectedType;
}


bool Image::validateDescriptors(int expectedSize) const {
    return descripteurs.size() == static_cast<std::size_t>(expectedSize);
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



