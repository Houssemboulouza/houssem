import asyncio
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import os
from lib.e2sm_rc_module import e2sm_rc_module
from lib.xAppBase import xAppBase

# Configuration d'InfluxDB et MLflow
INFLUXDB_URL = "http://10.180.113.115:32086"
INFLUXDB_TOKEN = "FUexw0jlTRCJ6pp861SW6dW_dsa1mctRcqr4DRune7L9_ThhUpEq9DA0KyD-LaGJOg32_e4oOhnz_ff-7xaNxg=="
INFLUXDB_ORG = "oran-lab"
INFLUXDB_BUCKET = "e2data_houssem"
MLFLOW_TRACKING_URI = "http://10.180.113.115:32256/"

# File for saving the model
MODEL_FILE = "rf_model.pkl"

class AdaptiveRANController(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        super(AdaptiveRANController, self).__init__(config, http_server_port, rmr_port)
        self.e2_module = e2sm_rc_module(self)
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)
            print("Existing model loaded.")
        else:
            print("No model found. Creating a new one.")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.save_model()

    def save_model(self):
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.model, f)
        print("Model successfully saved.")

    async def start(self):
        while self.running:
            await asyncio.sleep(60)
            await self.main()
            print("xApp is running. Sleeping for 60 seconds...")

    async def main(self):
        data = await self.fetch_data_from_influx()
        if data is not None and not data.empty:
            X, y, _ = self.preprocess_data(data)
            self.train_and_register_model(X, y)
            self.make_decisions(data, X, self.model)
        else:
            print("No data found in InfluxDB.")

    async def fetch_data_from_influx(self):
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        query_api = client.query_api()
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -10d)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = query_api.query_data_frame(query)
        client.close()
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def preprocess_data(self, data):
        data = data.dropna(subset=["RF.serving.RSSINR", "RF.serving.RSRP", "RF.serving.RSRQ", "TargetTput", "RRU.PrbUsedDl"])
        X = data[["RF.serving.RSSINR", "RF.serving.RSRP", "RF.serving.RSRQ", "RRU.PrbUsedDl"]]
        y = data["TargetTput"]
        return X, y, data

    def train_and_register_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run():
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(self.model, "model")
        print("Model trained and registered in MLflow.")

    def make_decisions(self, data, X, model):
        predictions = model.predict(X)
        for i, pred in enumerate(predictions):
            user_info = data.iloc[i]
            user_id = user_info.get("du_id", "Unknown")
            print(f"\nUser {user_id} - Prediction: {pred:.2f} (TargetTput)")
            if pred < 0.2:
                print(f"  Action: Low throughput detected for user {user_id}.")
                if user_info.get("RF.serving.RSSINR", 0) < 15:
                    print("  Decision: Handover to another cell recommended.")
                else:
                    print("  Decision: Increase transmission power or allocate more PRBs.")
            else:
                print(f"  QoS satisfactory for user {user_id}.")

if __name__ == "__main__":
    config = {}
    http_server_port = 8080
    rmr_port = 4560
    app = AdaptiveRANController(config, http_server_port, rmr_port)
    asyncio.run(app.start())
