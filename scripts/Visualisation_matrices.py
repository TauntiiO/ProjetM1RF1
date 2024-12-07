import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_dir = "../results/confusion_matrices"  
output_dir = "../results/Visualisations"   

os.makedirs(output_dir, exist_ok=True)

def visualize_confusion_matrix(file_path, output_path):
    """
    Génère une heatmap pour une matrice de confusion à partir de CSV.

    Entrée :
        - file_path (str) : chemin CSV qui possede la matrice de confusion.
        - output_path (str) : Chemin pour sauvegarder la heatmap.
    
    Sortie :
        - None : La heatmap est sauvegardée.
    """
    cm = pd.read_csv(file_path, index_col=0)
    
    representation_name = os.path.basename(file_path).replace("_confusion_matrix.csv", "")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {representation_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap sauvegardée dans : {output_path}")

for file_name in os.listdir(input_dir):
    if file_name.endswith("_confusion_matrix.csv"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace(".csv", ".png"))
        
        visualize_confusion_matrix(input_file_path, output_file_path)

print("Toutes les visualisations ont été générées.")
