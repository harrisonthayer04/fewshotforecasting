# %%
!pip install dtw
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from dtw import accelerated_dtw
from tqdm import tqdm

import math
import os
import inspect
import csv

# %%
def load_and_combine(csv_paths):
    df_list = []
    for path in csv_paths:
        temp_df = pd.read_csv(path, parse_dates=['timestamp'])
        temp_df.sort_values(by=['Station', 'timestamp'], inplace=True)
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

csv_files = [
    "../Data/training_data_january_2023.csv",
    "../Data/training_data_february_2023.csv"
]

full_df = load_and_combine(csv_files)
N = len(full_df)
train_end = int(0.70 * N)
support_end = int(0.85 * N)

train_df = full_df.iloc[:train_end].copy().reset_index(drop=True)
support_df = full_df.iloc[train_end:support_end].copy().reset_index(drop=True)
test_df   = full_df.iloc[support_end:].copy().reset_index(drop=True)

print("Training samples:", len(train_df))
print("Support samples:", len(support_df))
print("Test samples:",    len(test_df))

# %%
def create_4day_chunks(df_single_station):
    df_single_station = df_single_station.sort_values('timestamp').copy()
    df_single_station['date'] = df_single_station['timestamp'].dt.date
    
    day_groups = []
    for day, group in df_single_station.groupby('date'):
        day_groups.append((day, group.sort_values('timestamp')))
    
    chunks = []
    for i in range(len(day_groups) - 3):
        _, df0 = day_groups[i]
        _, df1 = day_groups[i+1]
        _, df2 = day_groups[i+2]
        _, df3 = day_groups[i+3]
        
        combined_df = pd.concat([df0, df1, df2, df3], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        if len(combined_df) == 384:
            flow_4days = combined_df['Total_Flow'].to_numpy()  # shape (384,)
            chunks.append(flow_4days)
    return chunks

def create_dataset(df):
    all_chunks = []
    for sid in df['Station'].unique():
        sid_df = df[df['Station'] == sid]
        station_chunks = create_4day_chunks(sid_df)
        all_chunks.extend(station_chunks)
    return np.array(all_chunks)  # shape (N, 384)

training_dataset = create_dataset(train_df)
support_dataset  = create_dataset(support_df)
test_dataset     = create_dataset(test_df)

print("training_dataset shape:", training_dataset.shape)
print("support_dataset shape:", support_dataset.shape)
print("test_dataset shape:",    test_dataset.shape)

# %%
def find_nearest_support(example_48h, support_dataset):
    """
    example_48h: shape (192,) - the 'past'
    support_dataset: shape (N_supp, 384)
    """
    best_dist = float('inf')
    best_idx = 0
    
    for i, s_chunk in enumerate(support_dataset):
        s_past = s_chunk[:192]
        dist = np.linalg.norm(example_48h - s_past)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

# %%
class SelfAttention(nn.Module):
    """
    A minimal single-head self-attention for sequences:
    Input shape:  (B, T, hidden_dim)
    Output shape: (B, T, hidden_dim)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        returns: (batch_size, seq_len, hidden_dim)
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attn_scores = torch.bmm(Q, K.transpose(1,2)) / self.scale  # (B, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)          # (B, T, T)
        out = torch.bmm(attn_weights, V)                           # (B, T, hidden_dim)
        return out


# %%
class FewShotLSTMAttn(nn.Module):
    """
    A direct few-shot baseline:
    1) difference = test_past - support_past
    2) LSTM hidden init from difference
    3) Self-attention on LSTM outputs
    """
    def __init__(self, hidden_dim=64, num_layers=1):
        super().__init__()
        # We expect difference vector to be shape (B, 192)
        # So diff_to_hidden expects input_dim=192
        self.diff_to_hidden = nn.Linear(192, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=1,         # support_future has shape (..., 1)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.attention = SelfAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, test_past, support_past, support_future):
        """
        test_past: (B, 192)
        support_past: (B, 192)
        support_future: (B, 192, 1)
        returns: (B, 192, 1)
        """
        # 1) difference
        diff_vec = test_past - support_past        # shape (B, 192)

        # 2) init hidden/cell from difference
        #    linear -> (B, hidden_dim)
        h0 = self.diff_to_hidden(diff_vec)         # (B, hidden_dim)
        c0 = torch.zeros_like(h0)                  # (B, hidden_dim)

        # LSTM expects (num_layers, B, hidden_dim)
        h0 = h0.unsqueeze(0)  # => (1, B, hidden_dim)
        c0 = c0.unsqueeze(0)  # => (1, B, hidden_dim)

        # 3) LSTM on support_future
        #    support_future: (B, 192, 1)
        lstm_out, (hn, cn) = self.lstm(support_future, (h0, c0))
        # => lstm_out: (B, 192, hidden_dim)

        # 4) self-attention over the LSTM outputs
        attn_out = self.attention(lstm_out)   # => (B, 192, hidden_dim)

        # 5) final linear at each time step
        pred = self.fc_out(attn_out)         # => (B, 192, 1)
        return pred


# %%
# Example:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

model = FewShotLSTMAttn(hidden_dim=64, num_layers=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()

def get_batch(dataset, batch_size):
    """
    Returns:
      x_test_tensor: (B, 192) -> test_past
      y_test_tensor: (B, 192, 1) -> test_future
      x_support_tensor: (B, 192) -> support_past
      y_support_tensor: (B, 192, 1) -> support_future
    """
    idxs = np.random.choice(len(dataset), batch_size, replace=False)
    
    x_test_list = []
    y_test_list = []
    x_support_list = []
    y_support_list = []

    for idx in idxs:
        chunk = dataset[idx]        # shape (384,)
        test_past = chunk[:192]     # shape (192,)
        test_future = chunk[192:]   # shape (192,)

        # find nearest
        idx_support = find_nearest_support(test_past, support_dataset)
        support_chunk = support_dataset[idx_support]
        support_past   = support_chunk[:192]
        support_future = support_chunk[192:]
        
        x_test_list.append(test_past)      # shape (192,)
        y_test_list.append(test_future)    # shape (192,)
        x_support_list.append(support_past)
        y_support_list.append(support_future)

    # Convert to Tensors
    x_test_arr = np.array(x_test_list)          # (B, 192)
    y_test_arr = np.array(y_test_list)          # (B, 192)
    x_support_arr = np.array(x_support_list)    # (B, 192)
    y_support_arr = np.array(y_support_list)    # (B, 192)

    x_test_tensor     = torch.tensor(x_test_arr,    dtype=torch.float)
    x_support_tensor  = torch.tensor(x_support_arr, dtype=torch.float)
    y_test_tensor     = torch.tensor(y_test_arr,    dtype=torch.float).unsqueeze(-1)
    y_support_tensor  = torch.tensor(y_support_arr, dtype=torch.float).unsqueeze(-1)

    return x_test_tensor, y_test_tensor, x_support_tensor, y_support_tensor

csv_filename = "few_shot_lstm_attn_log.csv"
with open(csv_filename, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "MAE", "MSE"])

for epoch in range(EPOCHS):
    total_mae = 0.0
    total_mse = 0.0

    # Let's define # of steps per epoch, e.g., 100
    for step in range(100):
        x_test, y_test, x_support, y_support = get_batch(training_dataset, BATCH_SIZE)
        
        x_test      = x_test.to(device)      # (B, 192)
        y_test      = y_test.to(device)      # (B, 192, 1)
        x_support   = x_support.to(device)   # (B, 192)
        y_support   = y_support.to(device)   # (B, 192, 1)
        
        optimizer.zero_grad()

        # forward
        y_pred = model(x_test, x_support, y_support)  # (B, 192, 1)

        # compute loss
        loss_mae = criterion_mae(y_pred, y_test)
        loss_mse = criterion_mse(y_pred, y_test)

        loss_mae.backward()  # or could do a combination of MAE+MSE
        optimizer.step()

        total_mae += loss_mae.item()
        total_mse += loss_mse.item()

    avg_mae = total_mae / (step + 1)
    avg_mse = total_mse / (step + 1)
    print(f"Epoch {epoch+1}/{EPOCHS} - MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}")

    # write to CSV
    with open(csv_filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_mae, avg_mse])

    # (optional) save model
    torch.save(model.state_dict(), f"few_shot_lstm_attn_epoch_{epoch+1}.pth")


# %%
