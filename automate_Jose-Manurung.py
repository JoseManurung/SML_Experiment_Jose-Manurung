import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing():
    print("Memulai otomatisasi preprocessing data...")
    
    # 1. Memuat dataset
    df = pd.read_csv('heart.csv')
    
    # 2. Menghapus duplikasi
    df = df.drop_duplicates()
    
    # 3. Label Encoding untuk kolom teks/kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    # 4. Memisahkan Fitur dan Target
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    # 5. Split dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Otomatisasi Selesai! Dimensi Data Training: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()