import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device# Ensure device is a torch.device object
        self.layer1 = nn.Linear(n_observations, 1024).to(self.device)
        self.layer2 = nn.Linear(1024, 1024).to(self.device)
        self.layer3 = nn.Linear(1024, n_actions).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure inputs are on the correct device
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
