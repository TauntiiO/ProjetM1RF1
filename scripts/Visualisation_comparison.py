import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  
input_base_dir = os.path.join(base_dir, "../results")  
output_dir = os.path.join(input_base_dir, "comparisons_visualizations") 
class_dirs = ["10_classes", "18_classes"]

os.makedirs(output_dir, exist_ok=True)

metrics_data = []

for class_dir in class_dirs:
    metrics_dir = os.path.join(input_base_dir, class_dir, "metrics")
    for file_name in os.listdir(metrics_dir):
        if file_name.endswith("_metrics.csv"):
            algo = "KMeans" if "KMeans" in file_name else "KNN"  

            # Identifie la méthode de représentation
            if "ART" in file_name:
                representation = "ART"
            elif "GFD" in file_name:
                representation = "GFD"
            elif "Yang" in file_name:
                representation = "Yang"
            elif "Zernike7" in file_name:
                representation = "Zernike7"
            else:
                representation = "Unknown"

            file_path = os.path.join(metrics_dir, file_name)

            metrics_df = pd.read_csv(file_path)
            
            if "Global" in metrics_df["Class"].values:
                metrics_df = metrics_df[metrics_df["Class"] != "Global"]

            metrics_df["Algorithm"] = algo
            metrics_df["ClassCount"] = int(class_dir.split("_")[0])  # 10 ou 18
            metrics_df["Representation"] = representation
            metrics_data.append(metrics_df)

metrics_combined_df = pd.concat(metrics_data, ignore_index=True)

metrics_combined_df["Precision"] = metrics_combined_df["Precision"].str.rstrip('%').astype(float)
metrics_combined_df["Recall"] = metrics_combined_df["Recall"].str.rstrip('%').astype(float)
metrics_combined_df["F1-Score"] = metrics_combined_df["F1-Score"].str.rstrip('%').astype(float)

metrics_list = ["Precision", "Recall", "F1-Score"]

for metric in metrics_list:
    plt.figure(figsize=(16, 10))
    sns.barplot(
        data=metrics_combined_df[metrics_combined_df["ClassCount"] == 10],
        x="ClassCount",
        y=metric,
        hue="Representation",
        palette="tab10"
    )
    plt.title(f"Comparaison de {metric} par méthode de représentation (10 classes)")
    plt.xlabel("Nombre de Classes")
    plt.ylabel(f"{metric} (%)")
    plt.legend(title="Méthode de Représentation", loc="lower right")
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"comparison_{metric.lower()}_representation_10_classes.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Graphique pour {metric} (méthodes de représentation) sauvegardé dans : {output_path}")

for metric in metrics_list:
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=metrics_combined_df,
        x="ClassCount",
        y=metric,
        hue="Algorithm",
        ci=None  # Supprime l'écart type
    )
    plt.title(f"Comparaison de {metric} par algorithme et nombre de classes")
    plt.xlabel("Nombre de Classes")
    plt.ylabel(f"{metric} (%)")
    plt.legend(title="Algorithme", loc="lower right")
    plt.grid(axis="y")
    plt.tight_layout()

    # Sauvegarde le graphique
    output_path = os.path.join(output_dir, f"comparison_{metric.lower()}_algorithms.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Graphique pour {metric} (algorithmes) sauvegardé dans : {output_path}")

print("Tous les graphiques de comparaison ont été générés.")
