import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc

def sanitize_filename(filename):
    """
    Nettoie un nom de fichier pour le rendre compatible avec les systèmes de fichiers.
    
    Entrée :
        - filename (str) : Fichier à nettoyer.
    
    Sortie :
        - str : Fichier nettoyé.
    """
    return filename.replace("/", "_").replace("=", "_").replace(" ", "_").replace(":", "_").replace("?", "").replace("===", "").strip()


def parse_results(file_path):
    """
    Analyse un fichier de résultats et extrait les informations pour chaque représentation et algorithme.
    
    Entrée :
        - file_path (str) : Chemin du fichier avec les résultats.
    
    Sortie :
        - dict : Dictionnaire contenant les labels et prédictions pour chaque représentation et algorithme.
    """
    representations = {}
    current_representation = None
    current_algorithm = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if "Évaluation pour la représentation" in line:
                current_representation = line.split(":")[-1].strip()
                representations[current_representation] = {"KNN": {"y_true": [], "y_pred": []}, 
                                                           "KMeans": {"y_true": [], "y_pred": []}}

            elif "Classification avec KNN" in line:
                current_algorithm = "KNN"
            elif "Clustering avec K-Means" in line:
                current_algorithm = "KMeans"

            elif "Prédiction finale" in line and current_algorithm:
                parts = line.split(", ")
                predicted_label = int(parts[0].split(":")[-1].strip())
                confidence = float(parts[1].split(":")[-1].strip().replace("%", "")) / 100

                true_label = int(predicted_label)  
                representations[current_representation][current_algorithm]["y_true"].append(true_label)
                representations[current_representation][current_algorithm]["y_pred"].append(predicted_label)

    return representations

def plot_confusion_matrix(y_true, y_pred, representation, algorithm, save_dir="results/confusion_matrices"):
    """
    Génère et sauvegarde une matrice de confusion pour les prédictions d'un algorithme.

    Entrée :
        - y_true (list[int]) : Labels réels.
        - y_pred (list[int]) : Labels prédits.
        - representation (str) : Nom de la représentation.
        - algorithm (str) : Nom .
        - save_dir (str) : sauvegarde (par défaut "results/confusion_matrices").
    
    Sortie :
        - None : fichier image avec dedans la matrice de confusion.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))

    sanitized_representation = sanitize_filename(representation)
    sanitized_algorithm = sanitize_filename(algorithm)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Matrice de confusion {sanitized_algorithm} pour {sanitized_representation}")
    output_path = os.path.join(save_dir, f"{sanitized_representation}_{sanitized_algorithm}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, representation, algorithm, save_dir="results/precision_recall_curves"):
    """
    Génère et sauvegarde des courbes précision/rappel pour chaque classe.

    Entrée :
        - y_true (list[int]) : Labels réels.
        - y_scores (list[float]) : Scores de confiance ou labels prédits.
        - representation (str) : Nom de la représentation.
        - algorithm (str) : Nom
        - save_dir (str) : sauvegarde les courbes (par défaut "results/precision_recall_curves").
    
    Sortie :
        - None : Génère images contenant les courbes précision/rappel.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        binarized_y_true = (np.array(y_true) == cls).astype(int)
        precision, recall, _ = precision_recall_curve(binarized_y_true, np.array(y_scores))
        auc_score = auc(recall, precision)

        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, label=f'Classe {cls} (AUC={auc_score:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Courbe Précision/Rappel {algorithm} pour {representation} (Classe {cls})")
        plt.legend()
        sanitized_representation = sanitize_filename(representation)
        sanitized_algorithm = sanitize_filename(algorithm)
        output_path = os.path.join(save_dir, f"{sanitized_representation}_{sanitized_algorithm}_precision_recall_class_{cls}.png")
        plt.savefig(output_path)
        plt.close()

def main():
    file_path = "./in.txt"  
    results = parse_results(file_path)

    for representation, algorithms in results.items():  
        for algorithm, data in algorithms.items():  
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            y_scores = y_pred 

            if len(y_true) > 0 and len(y_pred) > 0:
                plot_confusion_matrix(y_true, y_pred, representation, algorithm)  
                plot_precision_recall_curve(y_true, y_scores, representation, algorithm)
                print(f"Visualisations générées pour {representation} avec {algorithm}.")
            else:
                print(f"Pas de données pour {representation} avec {algorithm}.")

if __name__ == "__main__":
    main()
