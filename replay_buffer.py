import random
from collections import namedtuple, deque

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward','sampletime', 'next_state'])

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the ReplayBuffer.
        
        Parameters:
        capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque([], maxlen=capacity)  # Use deque for efficient FIFO operations
        self.capacity = capacity

    def add(self, state, action, reward,sampletime, next_state):
        """
        Add an experience to the buffer.
        
        Parameters:
        state: The current state of the environment.
        action: The action taken in the current state.
        reward: The reward received after taking the action.
        next_state: The state of the environment after the action is taken.
        """
        experience = Experience(state, action, reward,sampletime, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        
        Parameters:
        batch_size (int): The number of experiences to sample.
        
        Returns:
        list of Experience: A list of sampled experiences.
        """
        if len(self.buffer) < batch_size: 
            return None
        else:
            return random.sample(self.buffer, min(len(self.buffer), batch_size))
        

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.buffer)
