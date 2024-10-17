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
    DataRepresentation(const std::string& path);

    bool readFile();
    const std::vector<double>& getData() const;
    const std::string& getRepresentationType() const;
    bool loadFromDirectory(const std::string& dirPath, const std::string& pgmDir, std::vector<Image>& images);

private:
    void determineRepresentationType();
    int extractLabelFromFilename(const std::string& filename);
};

#endif
