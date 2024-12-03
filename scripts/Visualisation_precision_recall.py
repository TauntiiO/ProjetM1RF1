import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

input_dir = "../results/precision_recall_data"  # Dossier des fichiers PR
output_dir = "../results/Visualisations_PR"    # Dossier des visualisations

os.makedirs(output_dir, exist_ok=True)

def plot_pr_curve_grouped(file_path, output_path):
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

for file_name in os.listdir(input_dir):
    if file_name.endswith("_pr_data.csv"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace("_pr_data.csv", "_pr_curve_grouped.png"))
        
        plot_pr_curve_grouped(input_file_path, output_file_path)

print("Toutes les courbes PR groupées ont été générées.")
