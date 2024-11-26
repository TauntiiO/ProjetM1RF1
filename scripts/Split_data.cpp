#include "DataCollection.h"
#include <iostream>
#include <filesystem>
#include <regex>
#include <random>
#include <unordered_set>

namespace fs = std::filesystem;

bool isFileInRange(const std::string& fileName, const std::string& minFile, const std::string& maxFile) {
    std::regex filePattern(R"(s(\d{2})n(\d{3})\.zrk\.txt)");
    std::smatch match;

    if (std::regex_match(fileName, match, filePattern)) {
        int classNumber = std::stoi(match[1].str());
        int imageNumber = std::stoi(match[2].str());

        std::smatch minMatch, maxMatch;
        if (std::regex_match(minFile, minMatch, filePattern) &&
            std::regex_match(maxFile, maxMatch, filePattern)) {
            int minClass = std::stoi(minMatch[1].str());
            int minImage = std::stoi(minMatch[2].str());
            int maxClass = std::stoi(maxMatch[1].str());
            int maxImage = std::stoi(maxMatch[2].str());

            if (classNumber < minClass || classNumber > maxClass) return false;
            if (classNumber == minClass && imageNumber < minImage) return false;
            if (classNumber == maxClass && imageNumber > maxImage) return false;

            return true;
        }
    }
    return false;
}

void createTrainTestSplit(const std::vector<Image>& images, float trainRatio, const std::string& trainDir, const std::string& testDir) {
    std::unordered_set<std::string> addedFiles;
    std::vector<Image> filteredImages;

    const std::string minFile = "s01n001.zrk.txt";
    const std::string maxFile = "s11n001.zrk.txt";

    std::cout << "Détection des fichiers éligibles...\n";

    for (const auto& img : images) {
        std::string filePath = img.getImagePath();
        std::string fileName = fs::path(filePath).filename().string();

        std::cout << "Fichier : " << filePath << " | Label : " << img.getLabel()
                  << " | Type : " << img.getRepresentationType() << std::endl;

        if (img.getLabel() >= 1 && img.getLabel() <= 15 && img.getRepresentationType() == "Zernike7" &&
            isFileInRange(fileName, minFile, maxFile)) {
            if (!fs::exists(filePath)) {
                std::cerr << "Fichier introuvable, ignoré : " << filePath << std::endl;
                continue;
            }
            if (addedFiles.find(filePath) == addedFiles.end()) {
                filteredImages.push_back(img);
                addedFiles.insert(filePath);
                std::cout << "Ajouté : " << filePath << " (Label : " << img.getLabel() << ")\n";
            } else {
                std::cerr << "Fichier déjà ajouté, ignoré : " << filePath << std::endl;
            }
        } else {
            std::cout << "Ignoré (hors critère) : " << filePath << "\n";
        }
    }

    std::cout << "Nombre total de fichiers filtrés pour traitement : " << filteredImages.size() << std::endl;

    if (filteredImages.empty()) {
        std::cerr << "Erreur : Aucun fichier ne correspond aux critères.\n";
        return;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filteredImages.begin(), filteredImages.end(), g);

    size_t trainSize = static_cast<size_t>(filteredImages.size() * trainRatio);
    std::vector<Image> trainSet(filteredImages.begin(), filteredImages.begin() + trainSize);
    std::vector<Image> testSet(filteredImages.begin() + trainSize, filteredImages.end());

    if (fs::exists(trainDir)) fs::remove_all(trainDir);
    if (fs::exists(testDir)) fs::remove_all(testDir);
    fs::create_directories(trainDir);
    fs::create_directories(testDir);

    for (const auto& img : trainSet) {
        if (!fs::exists(img.getImagePath())) {
            std::cerr << "Fichier introuvable dans Train : " << img.getImagePath() << std::endl;
            continue;
        }
        try {
            fs::copy(img.getImagePath(), fs::path(trainDir) / fs::path(img.getImagePath()).filename(),
                     fs::copy_options::skip_existing);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Erreur dans Train : " << e.what() << std::endl;
        }
    }

    for (const auto& img : testSet) {
        if (!fs::exists(img.getImagePath())) {
            std::cerr << "Fichier introuvable dans Test : " << img.getImagePath() << std::endl;
            continue;
        }
        try {
            fs::copy(img.getImagePath(), fs::path(testDir) / fs::path(img.getImagePath()).filename(),
                     fs::copy_options::skip_existing);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Erreur dans Test : " << e.what() << std::endl;
        }
    }

    size_t trainCount = std::distance(fs::directory_iterator(trainDir), fs::directory_iterator{});
    size_t testCount = std::distance(fs::directory_iterator(testDir), fs::directory_iterator{});

    std::cout << "\n=== Résumé final ===\n";
    std::cout << "Train : " << trainCount << " fichiers\n";
    std::cout << "Test : " << testCount << " fichiers\n";

    if (trainCount + testCount != filteredImages.size()) {
        std::cerr << "Avertissement : Le total des fichiers (Train + Test) ne correspond pas au nombre filtré.\n";
    }
}

int main() {
    DataCollection dataset;
    dataset.loadDatasetFromDirectory("./data/=Signatures");

    std::string trainDir = "./data/=Signatures/=Zernike7/train";
    std::string testDir = "./data/=Signatures/=Zernike7/test";

    createTrainTestSplit(dataset.getImages(), 0.8, trainDir, testDir);

    return 0;
}
