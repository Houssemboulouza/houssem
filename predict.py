import sys
import os
import mlflow.pyfunc
import random
import pandas as pd

# Ajouter le dossier 'lib' au chemin des modules Python
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib'))
if lib_path not in sys.path:
    sys.path.append(lib_path)

from lib.e2sm_rc_module import e2sm_rc_module  # Importer le module depuis le dossier 'lib'

# Fonction pour charger le modèle depuis le Model Registry
def load_model_from_registry():
    model_uri = "models:/exemple_de_modele/1"  # Remplacez par le nom et la version corrects du modèle
    model = mlflow.pyfunc.load_model(model_uri)
    print("Modèle chargé depuis le registre MLflow.")
    return model

# Fonction pour effectuer une prédiction
def predict_with_model(model, data):
    prediction = model.predict(data)
    return prediction

# Fonction principale pour valider le modèle et prendre des décisions
def main():
    # Charger le modèle
    model = load_model_from_registry()

    # Simuler des données aléatoires pour valider le modèle
    sample_data = pd.DataFrame({
        "feature1": [random.uniform(0, 1) for _ in range(5)],
        "feature2": [random.uniform(0, 1) for _ in range(5)],
        "feature3": [random.uniform(0, 1) for _ in range(5)],
        "feature4": [random.uniform(0, 1) for _ in range(5)],
        "feature5": [random.uniform(0, 1) for _ in range(5)]
    })

    print("Données utilisées pour la prédiction :\n", sample_data)

    # Effectuer des prédictions
    predictions = predict_with_model(model, sample_data)
    print("Prédictions :", predictions)

    # Initialiser le module e2sm_rc_module pour envoyer des commandes RIC
    e2sm_rc = e2sm_rc_module(parent=None)  # Remplacez `parent=None` par une instance valide si nécessaire

    # Prendre des décisions en fonction des prédictions
    for idx, prediction in enumerate(predictions):
        if prediction == 0:
            print(f"Prédiction pour l'entrée {idx}: 0 - Action: Désactiver le noeud E2 ou changer la connectivité.")
            e2sm_rc.send_control_request_style_2_action_6(
                e2_node_id="E2Node-1",  # Remplacez par un ID valide
                ue_id=random.randint(1000, 9999),  # Simule un ID d'UE
                min_prb_ratio=10,
                max_prb_ratio=20,
                dedicated_prb_ratio=15
            )
        elif prediction == 1:
            print(f"Prédiction pour l'entrée {idx}: 1 - Action: Allouer plus de ressources pour le noeud E2.")
            e2sm_rc.send_control_request_style_2_action_6(
                e2_node_id="E2Node-2",  # Remplacez par un ID valide
                ue_id=random.randint(1000, 9999),
                min_prb_ratio=20,
                max_prb_ratio=50,
                dedicated_prb_ratio=30
            )
        else:
            print(f"Prédiction pour l'entrée {idx}: {prediction} - Aucune action définie.")

if __name__ == "__main__":
    main()
