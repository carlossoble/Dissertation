import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import datetime


data = pd.read_csv('master_df.csv')

# Convert 'Date' column to datetime
data['date_parsed'] = pd.to_datetime(data['Date'], errors='coerce')
if data['date_parsed'].isnull().any():
    raise ValueError("Some dates could not be parsed. Check the date format in the CSV.")

# Define the split date and split the data into Train and Test
split_date = pd.to_datetime("2021-01-01")  # adjust as needed
train_data = data[data['date_parsed'] < split_date].copy()
test_data  = data[data['date_parsed'] >= split_date].copy()

# Define feature groups and target
ticker_cols = [f"ticker_data_pre{i}" for i in range(30, 0, -1)]

# 3 assets (SAN, IBE, ITX), each has 5 days => total length = 15
other_cols = (
    [f"SAN_pre{i}" for i in range(5, 0, -1)] +
    [f"IBE_pre{i}" for i in range(5, 0, -1)] +
    [f"ITX_pre{i}" for i in range(5, 0, -1)]
)

# Static features
static_cols = [f"signature_{i}" for i in range(22)] + [f"levy_area_{i}" for i in [12, 13, 14]]

# Target column
target_col = "ticker_data_target"


# Extract features function
def extract_features(df):
    X_ticker = df[ticker_cols].values  # shape: (n_samples, 30)
    X_other = df[other_cols].values    # shape: (n_samples, 15)
    X_static = df[static_cols].values  # shape: (n_samples, 25)
    y = df[target_col].values.reshape(-1, 1)
    return X_ticker, X_other, X_static, y

X_ticker_train, X_other_train, X_static_train, y_train = extract_features(train_data)
X_ticker_test,  X_other_test,  X_static_test,  y_test  = extract_features(test_data)

# Scale features
scaler_ticker = StandardScaler()
X_ticker_train_scaled = scaler_ticker.fit_transform(X_ticker_train)
X_ticker_test_scaled  = scaler_ticker.transform(X_ticker_test)

scaler_other = StandardScaler()
X_other_train_scaled = scaler_other.fit_transform(X_other_train)
X_other_test_scaled  = scaler_other.transform(X_other_test)

scaler_static = StandardScaler()
X_static_train_scaled = scaler_static.fit_transform(X_static_train)
X_static_test_scaled  = scaler_static.transform(X_static_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)


# PyTorch Dataset
class FinancialDataset(Dataset):
    def __init__(self, ticker, other, static, y):
        self.ticker = torch.tensor(ticker, dtype=torch.float32)
        self.other = torch.tensor(other, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ticker)
    
    def __getitem__(self, idx):
        return self.ticker[idx], self.other[idx], self.static[idx], self.y[idx]

train_dataset = FinancialDataset(X_ticker_train_scaled, X_other_train_scaled, X_static_train_scaled, y_train_scaled)
test_dataset  = FinancialDataset(X_ticker_test_scaled,  X_other_test_scaled,  X_static_test_scaled,  y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)


# Define the model
class VaRLSTMModel(nn.Module):
    def __init__(self, 
                 ticker_seq_len=30,   # number of days for ticker
                 lstm_hidden=10, 
                 other_lstm_hidden=4, 
                 other_seq_len=5,     # length of each other asset sequence
                 n_other_assets=3,    # number of other assets
                 dense_hidden=8,
                 static_dim=25):
        super(VaRLSTMModel, self).__init__()
        
        # Branch 1: LSTM for main ticker
        # Input shape (batch, 30, 1)
        self.lstm_ticker = nn.LSTM(
            input_size=1, 
            hidden_size=lstm_hidden, 
            batch_first=True
        )
        
        # Branch 2: Single LSTM for other assets
        # We'll treat the 3 assets as 3 features, each with 5 time steps
        # => input_size=3, sequence_length=5
        self.lstm_other = nn.LSTM(
            input_size=n_other_assets, 
            hidden_size=other_lstm_hidden, 
            batch_first=True
        )
        
        # Branch 3: Dense branch for static features
        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, dense_hidden),
            nn.ReLU()
        )
        
        # Combined dimension:
        combined_in_dim = lstm_hidden + other_lstm_hidden + dense_hidden
        
        self.combined = nn.Sequential(
            nn.Linear(combined_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # final output
        )
    
    def forward(self, ticker_seq, other_seq, static_features):
        """
        ticker_seq: (batch, 30)
        other_seq : (batch, 15) => includes 3 assets, each with 5 days => reshape to (batch, 5, 3)
        static_features: (batch, 25)
        """
        # 1. Ticker LSTM
        x_ticker = ticker_seq.unsqueeze(-1)  # (batch, 30) => (batch, 30, 1)
        _, (h_ticker, _) = self.lstm_ticker(x_ticker)
        x_ticker_out = h_ticker[-1]  # (batch, lstm_hidden)
        
        # 2. Other LSTM
        # Reshape from (batch, 15) => (batch, 5, 3)
        batch_size = other_seq.shape[0]
        x_other = other_seq.view(batch_size, 5, 3)
        
        _, (h_other, _) = self.lstm_other(x_other)
        x_other_out = h_other[-1]  # (batch, other_lstm_hidden)
        
        # 3. Static branch
        x_static = self.static_branch(static_features)  # (batch, dense_hidden)
        
        # 4. Combine
        combined = torch.cat((x_ticker_out, x_other_out, x_static), dim=1)
        out = self.combined(combined)  # (batch, 1)
        return out

# Instantiate model
model = VaRLSTMModel(
    ticker_seq_len=30, 
    lstm_hidden=8,
    other_lstm_hidden=4,
    other_seq_len=5, 
    n_other_assets=3, 
    dense_hidden=8, 
    static_dim=25
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Quantile loss
def quantile_loss(y_pred, y_true, q):
    error = y_true - y_pred
    return torch.mean(torch.max(q * error, (q - 1) * error))

quantile = 0.05  # VaR at 5%


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for ticker_seq, other_seq, static_features, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(ticker_seq, other_seq, static_features)
        loss = quantile_loss(outputs, batch_y, quantile)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    # Evaluate
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for ticker_seq, other_seq, static_features, batch_y in test_loader:
            outputs = model(ticker_seq, other_seq, static_features)
            loss = quantile_loss(outputs, batch_y, quantile)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}")


# Save the trained model
today_str = datetime.datetime.now().strftime("%Y%m%d")
model_filename = f"var_lstm_lstm_dense_model{today_str}.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_filename)
print(f"Model saved to {model_filename}")


# Inference on entire dataset
X_ticker_all, X_other_all, X_static_all, y_all = extract_features(data)
X_ticker_all_scaled = scaler_ticker.transform(X_ticker_all)
X_other_all_scaled = scaler_other.transform(X_other_all)
X_static_all_scaled = scaler_static.transform(X_static_all)

ticker_tensor = torch.tensor(X_ticker_all_scaled, dtype=torch.float32)
other_tensor  = torch.tensor(X_other_all_scaled,  dtype=torch.float32)
static_tensor = torch.tensor(X_static_all_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    all_preds_scaled = model(ticker_tensor, other_tensor, static_tensor).numpy()

all_preds = scaler_y.inverse_transform(all_preds_scaled)


# Create results DataFrame
results_df = pd.DataFrame({
    "Ticker": data['Ticker'],
    "Date": data["Date"],
    "target": data[target_col],
    "prediction": all_preds.flatten()
})
results_df["Set"] = np.where(data["date_parsed"] < split_date, "Train", "Test")
results_df.to_csv("results_lstm_lstm_dense.csv", index=False)
print("Results saved to results_lstm_lstm_dense.csv")


