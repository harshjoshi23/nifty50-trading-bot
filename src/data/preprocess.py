import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset

def preprocess_data(stock_file, data_dir="../../data/", seq_length=60):
    # Load data
    df = pd.read_csv(f"{data_dir}/{stock_file}")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Use closing price
    data = df[['Close']].dropna()
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test (80-20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).reshape(-1, seq_length, 1)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test).reshape(-1, seq_length, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test, scaler

# Example usage
if __name__ == "__main__":
    stock = "ADANIPORTS.csv"
    X_train, y_train, X_test, y_test, scaler = preprocess_data(stock)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")