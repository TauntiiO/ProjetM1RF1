import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Répertoires des résultats
base_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script actuel
input_base_dir = os.path.join(base_dir, "../results")  # Répertoire parent des sous-dossiers
output_dir = os.path.join(input_base_dir, "comparisons_visualizations")  # Dossier pour les graphiques
class_dirs = ["10_classes", "18_classes"]

# Créer le dossier de sortie
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire pour stocker les données des métriques
metrics_data = []

# Parcours des répertoires
for class_dir in class_dirs:
    metrics_dir = os.path.join(input_base_dir, class_dir, "metrics")
    for file_name in os.listdir(metrics_dir):
        if file_name.endswith("_metrics.csv"):
            algo = "KMeans" if "KMeans" in file_name else "KNN"  # Déterminer l'algorithme
            file_path = os.path.join(metrics_dir, file_name)

            # Charger les données de métriques
            metrics_df = pd.read_csv(file_path)
            
            # Supprimer les lignes globales si elles existent
            if "Global" in metrics_df["Class"].values:
                metrics_df = metrics_df[metrics_df["Class"] != "Global"]
            
            # Ajouter les colonnes pour le contexte
            metrics_df["Algorithm"] = algo
            metrics_df["ClassCount"] = int(class_dir.split("_")[0])  # 10 ou 18
            metrics_data.append(metrics_df)

# Concaténer toutes les données en un seul DataFrame
metrics_combined_df = pd.concat(metrics_data, ignore_index=True)

# Transformation des colonnes en pourcentages pour cohérence
metrics_combined_df["Precision"] = metrics_combined_df["Precision"].str.rstrip('%').astype(float)
metrics_combined_df["Recall"] = metrics_combined_df["Recall"].str.rstrip('%').astype(float)
metrics_combined_df["F1-Score"] = metrics_combined_df["F1-Score"].str.rstrip('%').astype(float)

# Liste des métriques à visualiser
metrics_list = ["Precision", "Recall", "F1-Score"]

# Générer un graphique par métrique
for metric in metrics_list:
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=metrics_combined_df,
        x="ClassCount",
        y=metric,
        hue="Algorithm",
        ci="sd"
    )
    plt.title(f"Comparaison de {metric} par algorithme et nombre de classes")
    plt.xlabel("Nombre de Classes")
    plt.ylabel(f"{metric} (%)")
    plt.legend(title="Algorithme", loc="lower right")
    plt.grid(axis="y")
    plt.tight_layout()

    # Sauvegarder le graphique
    output_path = os.path.join(output_dir, f"comparison_{metric.lower()}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Graphique pour {metric} sauvegardé dans : {output_path}")

print("Tous les graphiques de comparaison ont été générés.")
