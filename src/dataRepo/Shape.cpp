#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

/**
 * Classe Shape
 * Cette classe représente une forme avec ses descripteurs et un label associé.
 * Elle permet d'accéder aux descripteurs, au label, et de générer une chaîne de caractères.
 */

class Shape {
    protected:
        vector<double> descripteurs;
        int label;

    public:
        Shape(const vector<double>& d, int l) : descripteurs(d), label(l) {}

        const vector<double>& getDescripteurs() const {
            return descripteurs;
        }

        int getLabel() const {
            return label;
        }

        string toString() const {
            ostringstream oss;
            oss << "Label : " << label << "\nDescripteurs : ";
            for (const auto& d : descripteurs) {
                oss << d << " ";
            }
            return oss.str();
        }
};