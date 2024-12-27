import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

# Configuration d'InfluxDB
INFLUXDB_URL = "http://10.180.113.115:32086"  # URL du serveur
INFLUXDB_TOKEN = "FUexw0jlTRCJ6pp861SW6dW_dsa1mctRcqr4DRune7L9_ThhUpEq9DA0KyD-LaGJOg32_e4oOhnz_ff-7xaNxg=="
INFLUXDB_ORG = "oran-lab"
INFLUXDB_BUCKET = "e2data_houssem"

# Initialisation du client InfluxDB
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Limite de traitement (nombre de lignes à traiter)
LIMIT = 100  # Ajustez cette valeur selon vos besoins

# Fonction pour importer les données UE
def import_ue_data(file_path):
    print(f"Lecture des données des utilisateurs depuis {file_path}...")
    data = pd.read_csv(file_path)
    print(f"{len(data)} lignes chargées avec succès.")

    for i, row in enumerate(data.iterrows()):
        if i >= LIMIT:  # Arrêter après la limite
            print(f"Limite atteinte : {LIMIT} lignes traitées.")
            break

        adjusted_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        try:
            point = Point("ue_metrics") \
                .tag("du_id", row[1]["du-id"]) \
                .tag("nrCellIdentity", row[1]["nrCellIdentity"]) \
                .field("RRU.PrbUsedDl", row[1]["RRU.PrbUsedDl"]) \
                .field("RF.serving.RSRP", row[1]["RF.serving.RSRP"]) \
                .field("RF.serving.RSRQ", row[1]["RF.serving.RSRQ"]) \
                .field("RF.serving.RSSINR", row[1]["RF.serving.RSSINR"]) \
                .field("TargetTput", row[1]["targetTput"]) \
                .time(adjusted_time, WritePrecision.NS)

            write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            print(f"Point UE envoyé : {point.to_line_protocol()}")
        except Exception as e:
            print(f"Erreur lors de l'envoi du point UE : {e}")

# Fonction pour importer les données des cellules
def import_cells_data(file_path):
    print(f"Lecture des données des cellules depuis {file_path}...")
    data = pd.read_csv(file_path)
    print(f"{len(data)} lignes chargées avec succès.")

    for i, row in enumerate(data.iterrows()):
        if i >= LIMIT:  # Arrêter après la limite
            print(f"Limite atteinte : {LIMIT} lignes traitées.")
            break

        adjusted_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        try:
            point = Point("cell_metrics") \
                .tag("du_id", row[1]["du-id"]) \
                .tag("nrCellIdentity", row[1]["nrCellIdentity"]) \
                .field("throughput", row[1]["throughput"]) \
                .field("availPrbDl", row[1]["availPrbDl"]) \
                .field("availPrbUl", row[1]["availPrbUl"]) \
                .field("pdcpBytesDl", row[1]["pdcpBytesDl"]) \
                .time(adjusted_time, WritePrecision.NS)

            write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            print(f"Point Cell envoyé : {point.to_line_protocol()}")
        except Exception as e:
            print(f"Erreur lors de l'envoi du point Cell : {e}")

# Exécution principale
if __name__ == "__main__":
    print("--- Début de l'importation des données ---")
    import_ue_data("./data/ue.csv")
    import_cells_data("./data/cells.csv")
    print("--- Fin de l'importation ---")

    # Fermeture du client
    client.close()
