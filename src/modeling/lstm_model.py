import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.preprocess import preprocess_data

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Last time step
        return out

def train_model(stock_file, seq_length=60, epochs=100, batch_size=32):
    # Preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(stock_file, seq_length=seq_length)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"../../models/lstm_{stock_file.replace('.csv', '')}.pth")
    return model, scaler

if __name__ == "__main__":
    stock = "ADANIPORTS.csv"
    model, scaler = train_model(stock)