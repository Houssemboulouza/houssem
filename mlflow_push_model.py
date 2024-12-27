import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Étape 1 : Créez des données factices pour l'entraînement
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 2 : Entraînez un modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Étape 3 : Évaluez le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision : {accuracy:.2f}")

# Étape 4 : Configurez MLflow
mlflow.set_tracking_uri("http://10.180.113.115:32256/")  # Remplacez par l'URI de votre serveur MLflow
experiment_name = "model_registry_example"  # Nom de l'expérience
mlflow.set_experiment(experiment_name)

# Étape 5 : Enregistrez le modèle dans le Model Registry
registered_model_name = "logistic_regression_model"

with mlflow.start_run():
    # Enregistrez les métriques et hyperparamètres
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("model_type", "LogisticRegression")

    # Enregistrez le modèle
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model",
        registered_model_name=registered_model_name
    )
    print(f"Modèle enregistré avec succès sous le nom '{registered_model_name}'.")

# Vérification et validation
client = mlflow.tracking.MlflowClient()
try:
    model_versions = client.get_latest_versions(registered_model_name)
    print(f"Versions disponibles pour le modèle '{registered_model_name}': {[mv.version for mv in model_versions]}")
except Exception as e:
    print(f"Erreur lors de la vérification des versions du modèle : {e}")
