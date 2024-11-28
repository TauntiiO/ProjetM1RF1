import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def sanitize_filename(filename):
    """Nettoie un nom de fichier en supprimant ou remplaçant les caractères spéciaux."""
    return filename.replace("/", "_").replace("=", "_").replace(" ", "_").replace(":", "_").replace("?", "").replace("===", "").strip()

# Fonction pour lire le fichier texte et extraire les données
def parse_results(file_path):
    representations = {}
    current_representation = None
    current_algorithm = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Identifier le début d'une nouvelle représentation
            if "Évaluation pour la représentation" in line:
                current_representation = line.split(":")[-1].strip()
                representations[current_representation] = {"KNN": {"y_true": [], "y_pred": []}, 
                                                           "KMeans": {"y_true": [], "y_pred": []}}

            # Identifier l'algorithme (KNN ou K-Means)
            elif "Classification avec KNN" in line:
                current_algorithm = "KNN"
            elif "Clustering avec K-Means" in line:
                current_algorithm = "KMeans"

            # Extraire les prédictions
            elif "Prédiction finale" in line and current_algorithm:
                parts = line.split(", ")
                predicted_label = int(parts[0].split(":")[-1].strip())
                confidence = float(parts[1].split(":")[-1].strip().replace("%", "")) / 100

                # Simulez le label correct (si présent)
                true_label = int(predicted_label)  # À adapter si les vérités terrain sont ailleurs
                representations[current_representation][current_algorithm]["y_true"].append(true_label)
                representations[current_representation][current_algorithm]["y_pred"].append(predicted_label)

    return representations

# Fonction pour afficher et sauvegarder une matrice de confusion
def plot_confusion_matrix(y_true, y_pred, representation, algorithm, save_dir="results/confusion_matrices"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))

    # Nettoyer les noms
    sanitized_representation = sanitize_filename(representation)
    sanitized_algorithm = sanitize_filename(algorithm)

    # Création de la matrice de confusion
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Matrice de confusion {sanitized_algorithm} pour {sanitized_representation}")
    output_path = os.path.join(save_dir, f"{sanitized_representation}_{sanitized_algorithm}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

# Main
def main():
    file_path = "./in.txt"  # Chemin vers votre fichier texte
    results = parse_results(file_path)

    for representation, algorithms in results.items():  # Pour chaque représentation
        for algorithm, data in algorithms.items():  # Pour chaque algorithme (KNN, KMeans)
            y_true = data["y_true"]
            y_pred = data["y_pred"]

            if len(y_true) > 0 and len(y_pred) > 0:
                plot_confusion_matrix(y_true, y_pred, representation, algorithm)  # Passez 'algorithm'
                print(f"Matrice de confusion générée pour {representation} avec {algorithm}.")
            else:
                print(f"Pas de données pour {representation} avec {algorithm}.")

if __name__ == "__main__":
    main()
