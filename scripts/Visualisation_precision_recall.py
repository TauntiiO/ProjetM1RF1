import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

base_dir = os.path.dirname(os.path.abspath(__file__))
input_base_dir = os.path.join(base_dir, "../results")  
output_base_dir = os.path.join(base_dir, "../results/Visualisations_PR")

os.makedirs(output_base_dir, exist_ok=True)

def plot_pr_curve_grouped(file_path, output_path):
    """
    Génère une courbe précision/rappel pour les classes avec les meilleurs et pires AUC.

    Entrée :
        - file_path (str) : Chemin du fichier CSV.
        - output_path (str) : Pour sauvegarder la courbe générée.
    
    Sortie :
        - None : Courbe sauvegardée.
    """
    data = pd.read_csv(file_path)
    true_labels = data['TrueLabel']
    confidence_scores = data['ConfidenceScore']
    classes = sorted(true_labels.unique())
    auc_values = {}

    for c in classes:
        binary_true_labels = (true_labels == c).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true_labels, confidence_scores)
        auc_values[c] = auc(recall, precision)

    sorted_classes = sorted(auc_values.items(), key=lambda x: x[1])

    top_classes = sorted_classes[-3:]  # Meilleures classes (3 dernières)
    worst_classes = sorted_classes[:3]  # Pires classes (3 premières)

    plt.figure(figsize=(12, 8))
    for c, auc_value in top_classes + worst_classes:
        binary_true_labels = (true_labels == c).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true_labels, confidence_scores)
        plt.plot(recall, precision, label=f"Class {c} (AUC={auc_value:.2f})")

    plt.title("Précision/Rappel - Classes avec AUC Extrêmes")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.legend(loc="best")
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()
    print(f"Courbe PR sauvegardée dans : {output_path}")

for sub_dir in os.listdir(input_base_dir):
    input_dir = os.path.join(input_base_dir, sub_dir, "precision_recall_data")
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        output_dir = os.path.join(output_base_dir, sub_dir, "precision_recall_data")
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir(input_dir):
            if file_name.endswith("_pr_data.csv"):
                input_file_path = os.path.join(input_dir, file_name)
                output_file_path = os.path.join(output_dir, file_name.replace("_pr_data.csv", "_pr_curve_grouped.png"))
                
                plot_pr_curve_grouped(input_file_path, output_file_path)

print("Toutes les courbes PR groupées ont été générées.")
