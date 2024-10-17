#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/DataRepresentation.cpp"
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/Image.cpp"
#include "/home/user/Documents/M1/s1/ProjetM1RF1/src/dataRepo/DataCollection.cpp"
using namespace std;

int main() {
    string representationsDir = "/home/user/Documents/M1/s1/ProjetM1RF1/data/=Signatures"; 

    DataCollection dataset;

    dataset.loadDatasetFromDirectory(representationsDir); 
    // Affichez le dataset
    dataset.printDataset(); 

    return 0;
}