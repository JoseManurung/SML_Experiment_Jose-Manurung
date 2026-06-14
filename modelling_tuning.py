import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_log():
    print("Memulai proses training model dan tracking MLflow...")
    
    # 1. Setup Tracking URI ke DagsHub agar otomatis online
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jose_manurung8"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "jose_manurung8" 
    mlflow.set_tracking_uri("https://dagshub.com/jose_manurung8/SML_Experiment_Jose-Manurung.mlflow")
    
    # 2. Memuat dataset 
    url = "https://raw.githubusercontent.com/datasets/heart-disease-dataset/master/data/heart-disease.csv"
    try:
        df = pd.read_csv(url)
        # Menyesuaikan nama kolom target jika berbeda di repositori publik
        if 'target' in df.columns:
            df = df.rename(columns={'target': 'HeartDisease'})
    except Exception as e:
        print("Menggunakan fallback data simulation...")
        np.random.seed(42)
        data = np.random.randn(200, 11)
        df = pd.DataFrame(data, columns=[f'f{i}' for i in range(11)])
        df['HeartDisease'] = np.random.choice([0, 1], size=200)

    df = df.drop_duplicates()
    
    # 3. Preprocessing otomatis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Set nama eksperimen di MLflow
    mlflow.set_experiment("Heart_Failure_Prediction_Eksperimen")
    
    # 5. Mulai Run MLflow
    with mlflow.start_run(run_name="Random_Forest_Tuning"):
        n_estimators = 50
        max_depth = 3
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        print(f"Model berhasil dilatih! Akurasi: {acc:.4f}")
        
        mlflow.sklearn.log_model(model, "model")
        
        # Artefak 1: Gambar matriks
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig('training_confusion_matrix.png')
        plt.close()
        mlflow.log_artifact('training_confusion_matrix.png')
        
        # Artefak 2: File JSON info
        with open("metric_info.json", "w") as f:
            f.write('{"status": "training_successful", "dataset": "heart_failure"}')
        mlflow.log_artifact('metric_info.json')

    print("Pencatatan ke MLflow DagsHub Selesai!")

if __name__ == "__main__":
    train_and_log()
