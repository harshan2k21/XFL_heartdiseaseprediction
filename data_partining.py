import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, Subset

def partition_data(num_clients):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    # Load and clean
    df = pd.read_csv(url, names=columns, na_values="?")
    df = df.fillna(df.mean()) 
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Standardize (Important for MLP performance)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    partitions = np.array_split(indices, num_clients)
    
    return [Subset(dataset, idx) for idx in partitions]
