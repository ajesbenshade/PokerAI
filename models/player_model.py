# models/player_model.py

import torch.nn as nn

class PlayerModel(nn.Module):
    def __init__(self, num_players, input_size=None, hidden_sizes=[128, 64], output_size=1):
        super(PlayerModel, self).__init__()
        if input_size is None:
            input_size = 6 + (6 * num_players) + 10 + 1  # 6 fixed + 6 per player + 10 community + 1 bet amount
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
