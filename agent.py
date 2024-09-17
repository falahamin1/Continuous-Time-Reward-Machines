import numpy as np
import torch as torch
class Agent: 
    def __init__(self, epsilon, actions, policy_net, device = 'cpu'):
        self.epsilon = epsilon
        self.actions = actions
        self.policy_net = policy_net
        self.device = device
        self.policy_net.to(self.device) 
    def epsilon_greedy_policy(self,state ,epsilon):
        self.epsilon = epsilon
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)  # input is assumed to be x,y coordinate of environment and q state of the ctrm
            action_values = self.policy_net(state_tensor)
            max_value, max_index = torch.max(action_values, dim=1)
            best_action = max_index.item()
            return self.actions[best_action]
    def get_best_action(self,state):
        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)  # input is assumed to be x,y coordinate of environment and q state of the ctrm
        action_values = self.policy_net(state_tensor)
        max_value, max_index = torch.max(action_values, dim=1)
        best_action = max_index.item()
        return self.actions[best_action]





