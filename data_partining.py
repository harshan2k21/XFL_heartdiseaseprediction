import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset

def partition_data(num_clients):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    # 1. Load raw data but DO NOT clean or scale it globally
    df = pd.read_csv(url, names=columns, na_values="?")
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X_raw = df.drop('target', axis=1).values
    y_raw = df['target'].values
    
    # 2. Split the RAW indices first (Simulating 5 raw hospital databases)
    indices = np.arange(len(X_raw))
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)
    
    client_datasets = []
    
    # 3. Flaw Fixed: Localized Imputation and Scaling
    for idx in client_indices:
        X_local = X_raw[idx]
        y_local = y_raw[idx]
        
        # A. Local Imputation (Median is safer for medical outliers)
        imputer = SimpleImputer(strategy='median')
        X_local = imputer.fit_transform(X_local)
        
        # B. Local Scaling (Zero Data Leakage between hospitals)
        scaler = StandardScaler()
        X_local = scaler.fit_transform(X_local)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_local, dtype=torch.float32)
        y_tensor = torch.tensor(y_local, dtype=torch.long)
        
        # Append the fully localized dataset
        client_datasets.append(TensorDataset(X_tensor, y_tensor))
        
    return client_datasets
