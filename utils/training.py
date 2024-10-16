# utils/training.py

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
from models.player_model import PlayerModel
from config import device, player_models, logger
import os

def train_player_model(player, gameplay_data: List[Dict[str, Any]], device: torch.device):
    from classes.player import Player  # Import inside the function to avoid circular import
    
    if not gameplay_data:
        return

    df = pd.DataFrame(gameplay_data)
    if 'outcome' not in df.columns:
        df['outcome'] = 0

    X_train = pd.DataFrame(df['features'].tolist())
    y_train = df['outcome']

    y_train = pd.to_numeric(y_train, errors='coerce')

    valid_indices = (~y_train.isna()) & (~y_train.isin([np.inf, -np.inf])) & (y_train.abs() <= 1e10)
    X_train_clean = X_train[valid_indices]
    y_train_clean = y_train[valid_indices]

    if y_train_clean.empty:
        return

    y_train_clean = y_train_clean.astype(float)

    X_train_tensor = torch.tensor(X_train_clean.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_clean.values, dtype=torch.float32).to(device).unsqueeze(1)

    mean = X_train_tensor.mean(dim=0)
    std = X_train_tensor.std(dim=0)
    std[std == 0] = 1.0  # Prevent division by zero

    X_train_scaled = (X_train_tensor - mean) / std

    try:
        input_size = X_train_scaled.shape[1]
        model = PlayerModel(num_players=player.num_players, input_size=input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(X_train_scaled, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        os.makedirs("models_saved", exist_ok=True)
        torch.save(model.state_dict(), f"models_saved/pytorch_player_{player.player_id}.pt")
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, f"models_saved/scaler_player_{player.player_id}.pt")
        player.model = model
        player_models[player.player_id] = model

    except Exception as e:
        logger.error(f"Failed to train model for Player {player.player_id}: {e}")
