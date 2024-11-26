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
    Image(const std::vector<double>& descripteurs, int label, const std::string& type, const std::string& path);

    const std::vector<double>& getDescripteurs() const;
    void setDescripteurs(const std::vector<double>& newDescripteurs);

    int getLabel() const;
    const std::string& getRepresentationType() const;
    const std::string& getImagePath() const;

    bool isValidRepresentation(const std::string& expectedType) const;
    bool validateDescriptors(int expectedSize) const;
    bool validateDescriptorsForType() const;
    bool validateLabel(int minLabel, int maxLabel) const;

    bool operator<(const Image& other) const;


    std::string toString() const;
};


#endif
