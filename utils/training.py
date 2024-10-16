import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
from models.player_model import PlayerModel
from config import device, player_models, logger
import os
from torch.cuda.amp import autocast, GradScaler  # For mixed precision

def train_player_model(player, gameplay_data: List[Dict[str, Any]], device: torch.device):
    from classes.player import Player  # Import inside the function to avoid circular import
    
    if not gameplay_data:
        logger.warning(f"No gameplay data available for Player {player.player_id}. Skipping training.")
        return
    
    # Convert gameplay data to DataFrame
    df = pd.DataFrame(gameplay_data)
    
    # Ensure 'outcome' column exists
    if 'outcome' not in df.columns:
        df['outcome'] = 0
    
    # Extract features and outcomes
    X_train = pd.DataFrame(df['features'].tolist())
    y_train = pd.to_numeric(df['outcome'], errors='coerce')
    
    # Filter out invalid data
    valid_indices = (~y_train.isna()) & (~y_train.isin([np.inf, -np.inf])) & (y_train.abs() <= 1e10)
    X_train_clean = X_train[valid_indices].reset_index(drop=True)
    y_train_clean = y_train[valid_indices].reset_index(drop=True)
    
    if y_train_clean.empty:
        logger.warning(f"All gameplay data for Player {player.player_id} is invalid. Skipping training.")
        return
    
    y_train_clean = y_train_clean.astype(float)
    
    # Convert to torch tensors on CPU
    X_train_tensor = torch.tensor(X_train_clean.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_clean.values, dtype=torch.float32).unsqueeze(1)
    
    # Compute mean and std on CPU for scaling
    mean = X_train_tensor.mean(dim=0)
    std = X_train_tensor.std(dim=0)
    std[std == 0] = 1.0  # Prevent division by zero
    
    # Scale features
    X_train_scaled = (X_train_tensor - mean) / std
    
    # Create TensorDataset and DataLoader with pin_memory=True for faster transfers
    dataset = TensorDataset(X_train_scaled, y_train_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,         # Adjust based on your CPU cores
        pin_memory=True,       # Speeds up data transfer to GPU
        prefetch_factor=2      # Number of batches to prefetch
    )
    
    try:
        input_size = X_train_scaled.shape[1]
        model = PlayerModel(num_players=player.num_players, input_size=input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # Switched to AdamW for better regularization
        
        # Initialize GradScaler for mixed precision
        scaler = GradScaler()
        
        model.train()
        for epoch in range(20):  # Increased epochs to 20 for better convergence
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                # Move data to GPU asynchronously
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Use autocast for mixed precision
                with autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item() * batch_x.size(0)
            
            average_loss = epoch_loss / len(dataset)
            logger.info(f"Player {player.player_id} - Epoch {epoch+1}/20 - Loss: {average_loss:.4f}")
        
        # Ensure the models_saved directory exists
        os.makedirs("models_saved", exist_ok=True)
        
        # Save the trained model
        model_path = f"models_saved/pytorch_player_{player.player_id}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model for Player {player.player_id} at {model_path}")
        
        # Save scaler parameters
        scaler_path = f"models_saved/scaler_player_{player.player_id}.pt"
        torch.save({'mean': mean.cpu(), 'std': std.cpu()}, scaler_path)
        logger.info(f"Saved scaler for Player {player.player_id} at {scaler_path}")
        
        # Assign the trained model to the player instance and global dictionary
        player.model = model
        player_models[player.player_id] = model
    
    except Exception as e:
        logger.error(f"Failed to train model for Player {player.player_id}: {e}")