import os
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Configuration d'InfluxDB
INFLUXDB_URL = "http://10.180.113.115:32086"
INFLUXDB_TOKEN = "FUexw0jlTRCJ6pp861SW6dW_dsa1mctRcqr4DRune7L9_ThhUpEq9DA0KyD-LaGJOg32_e4oOhnz_ff-7xaNxg=="
INFLUXDB_ORG = "oran-lab"
INFLUXDB_BUCKET = "e2data_houssem"

# Configuration de MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = 'user'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'sr9TvkIjaj'
mlflow.set_tracking_uri("http://10.180.113.115:32256/")
mlflow.set_experiment("model_houssem")

# Fonction pour récupérer les données depuis InfluxDB
def fetch_data_from_influxdb():
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        query_api = client.query_api()
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -10d)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = query_api.query_data_frame(query)
        client.close()
        if isinstance(result, list):
            result = pd.concat(result, ignore_index=True)
        return result
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None

# Prétraitement des données
def preprocess_data(data):
    if data is not None:
        print("Colonnes disponibles :", data.columns)
        data = data.dropna(subset=["RF.serving.RSSINR", "RF.serving.RSRP", "RF.serving.RSRQ", "TargetTput", "RRU.PrbUsedDl"])
        X = data[["RF.serving.RSSINR", "RF.serving.RSRP", "RF.serving.RSRQ", "RRU.PrbUsedDl"]]
        y = data["TargetTput"]
        return X, y, data
    else:
        print("Aucune donnée disponible.")
        return None, None, None

# Entraînement et enregistrement du modèle
def train_and_register_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluer le modèle
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Erreur quadratique moyenne (MSE) : {mse}")

    # Enregistrer le modèle dans MLflow
    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print("Modèle enregistré dans MLflow.")
    return model

# Prise de décision basée sur les prédictions
def make_decisions(data, X, model):
    if model is not None:
        predictions = model.predict(X)
        for i, pred in enumerate(predictions):
            user_info = data.iloc[i]
            user_id = user_info.get("du_id", "Inconnu")  # Récupérer `du_id` ou utiliser "Inconnu" par défaut

            print(f"\nUtilisateur {user_id} - Prediction: {pred:.2f} (TargetTput)")

            # Décision basée sur le débit prédit
            if pred < 0.2:  # Seuil pour faible débit
                print(f"  Action : Débit faible détecté pour l'utilisateur {user_id}.")
                if user_info.get("RF.serving.RSSINR", 0) < 15:  # Si RSSINR est bas
                    print(f"  Décision : Handover vers une autre cellule recommandé.")
                else:
                    print("  Décision : Augmenter la puissance d'émission ou allouer plus de PRB.")
            else:
                print(f"  QoS satisfaisante pour l'utilisateur {user_id}.")
    else:
        print("Modèle non disponible pour la prise de décision.")


# Exécution principale
if __name__ == "__main__":
    print("--- Début du processus ---")
    
    # Récupération des données
    data = fetch_data_from_influxdb()
    if data is not None:
        print("\nDonnées récupérées :")
        print(data.head(30))

        # Prétraitement
        X, y, preprocessed_data = preprocess_data(data)
        if X is not None and y is not None:
            print("\nDonnées après prétraitement :")
            print(X.head())

            # Entraînement et enregistrement du modèle
            model = train_and_register_model(X, y)

            # Prise de décision
            make_decisions(preprocessed_data, X, model)
    else:
        print("Échec de la récupération des données.")

    print("--- Fin du processus ---")
