import asyncio
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from influxdb_client import InfluxDBClient
from datetime import datetime

# Configuration InfluxDB
INFLUXDB_URL = "http://10.180.113.115:32086"
INFLUXDB_TOKEN = "FUexw0jlTRCJ6pp861SW6dW_dsa1mctRcqr4DRune7L9_ThhUpEq9DA0KyD-LaGJOg32_e4oOhnz_ff-7xaNxg=="
INFLUXDB_ORG = "oran-lab"
INFLUXDB_BUCKET = "e2data_houssem"

# Fonction pour lire les données d'InfluxDB
async def fetch_data_from_influx():
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: -1h)
    |> filter(fn: (r) => r._measurement == "ue_metrics")
    |> filter(fn: (r) => r._field == "RSRP" or r._field == "TargetTput")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["RSRP", "TargetTput"])
    '''
    tables = query_api.query(query)
    data = []

    for table in tables:
        for record in table.records:
            data.append([record["RSRP"], record["TargetTput"]])

    client.close()
    return np.array(data)

# Fonction principale
async def main():
    # Charger les données
    print("Lecture des données depuis InfluxDB...")
    data = await fetch_data_from_influx()

    if data.size == 0:
        print("Aucune donnée trouvée dans InfluxDB.")
        return

    # Préparer les données pour le modèle
    print("Préparation des données et entraînement du modèle...")
    X = data[:, :-1]  # Variables indépendantes
    y = (data[:, -1] > 0.5).astype(int)  # Label binaire basé sur une condition

    # Vérification de la distribution des classes
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Distribution des classes : {class_distribution}")

    # Ajouter des échantillons synthétiques si une seule classe est présente
    if len(unique) == 1:
        print("Une seule classe détectée, ajout d'échantillons synthétiques...")
        X_synthetic = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(10, X.shape[1]))
        y_synthetic = 1 - unique[0]
        X = np.vstack([X, X_synthetic])
        y = np.hstack([y, [y_synthetic] * len(X_synthetic)])

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Évaluer le modèle
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Exécuter la fonction principale
if __name__ == "__main__":
    asyncio.run(main())
