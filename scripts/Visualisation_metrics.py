import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script actuel
input_base_dir = os.path.join(base_dir, "../results")  # Répertoire parent des sous-dossiers
output_base_dir = os.path.join(base_dir, "../results/Visualisations")

os.makedirs(output_base_dir, exist_ok=True)

def visualize_metrics(file_path, output_path):
    """
    Fabrique un graphique des métriques (précision, rappel, F1-score) pour chaque classe.

    Entrée :
        - file_path (str) : Chemin CSV avec les métriques.
        - output_path (str) : chemin pour sauvegarder le graphique.
    
    Sortie :
        - None : Le graphique est sauvegardé.
    """
    metrics_df = pd.read_csv(file_path)
    if "Global" in metrics_df["Class"].values:
        metrics_df = metrics_df[metrics_df["Class"] != "Global"]
    metrics_df["Precision"] = metrics_df["Precision"].str.rstrip('%').astype(float)
    metrics_df["Recall"] = metrics_df["Recall"].str.rstrip('%').astype(float)
    metrics_df["F1-Score"] = metrics_df["F1-Score"].str.rstrip('%').astype(float)

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["Class"], metrics_df["Precision"], label="Precision", marker='o')
    plt.plot(metrics_df["Class"], metrics_df["Recall"], label="Recall", marker='s')
    plt.plot(metrics_df["Class"], metrics_df["F1-Score"], label="F1-Score", marker='d')

    plt.title(f'Metrics Visualization for {os.path.basename(file_path).replace("_metrics.csv", "")}')
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.ylim(0, 110)
    plt.legend(loc="lower right")
    plt.grid(axis='y')

    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé dans : {output_path}")


for sub_dir in os.listdir(input_base_dir):
    input_dir = os.path.join(input_base_dir, sub_dir, "metrics")
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        output_dir = os.path.join(output_base_dir, sub_dir, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(input_dir):
            if file_name.endswith("_metrics.csv"):
                input_file_path = os.path.join(input_dir, file_name)
                output_file_path = os.path.join(output_dir, file_name.replace("_metrics.csv", "_metrics.png"))
                
                visualize_metrics(input_file_path, output_file_path)

print("Tous les graphiques des métriques ont été générés.")
