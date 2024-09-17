import torch
import torch.nn.functional as F
import math
from collections import namedtuple, deque

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward','sampletime', 'next_state'])

class Optimizer:
    def __init__(self, policy_net, target_net, optimizer, memory, device = 'cpu', GAMMA=0.99, BATCH_SIZE=64, TAU=0.01):
        self.optimizer = optimizer
        self.memory = memory
        self.device = device
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
    
    

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return  # Not enough samples to train

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.stack([torch.tensor(s).float().to(self.device) for s in batch.next_state if s is not None])
        state_batch = torch.stack([torch.tensor(s).float().to(self.device) for s in batch.state])
        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).to(self.device)
        reward_batch = torch.stack([torch.tensor(r).float().to(self.device) for r in batch.reward])
        time_batch = torch.stack([torch.tensor(t).float().to(self.device) for t in batch.sampletime])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        if non_final_mask.any():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * torch.exp(-1 * self.GAMMA * time_batch)) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def optimize_model(self):
    #     if len(self.memory) < self.BATCH_SIZE:
    #         return  # Not enough samples to train

    #     transitions = self.memory.sample(self.BATCH_SIZE)
    #     batch = Experience(*zip(*transitions))

        
    #     # Convert tuple states to tensors and ensure they are on the correct device
    #     #States and next states are tuples, so we use torch.tensor(s).float()
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
    #     non_final_next_states = torch.stack([torch.tensor(s).float().to(self.device) for s in batch.next_state if s is not None])
    #     state_batch = torch.stack([torch.tensor(s).float().to(self.device) for s in batch.state])
    #     action_batch = torch.tensor([a for a in batch.action], dtype=torch.long).to(self.device)
    
    #     reward_batch = torch.stack([torch.tensor(r).float().to(self.device) for r in batch.reward])
    #     time_batch = torch.stack([torch.tensor(t).float().to(self.device) for t in batch.sampletime])

    #     state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    #     next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
    #     if non_final_mask.any():
    #         next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    #     expected_state_action_values = (next_state_values * torch.exp(-1 * self.GAMMA * time_batch)) + reward_batch

    #     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def update_target_soft(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
    
    def update_target_hard(self):
        # Copy the weights from policy_net to target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
