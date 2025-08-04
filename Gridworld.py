import numpy as np
class Gridworld:
    def __init__(self, rows, columns, probability):
        self.initial_state = (0,0)
        self.rows = rows
        self.cols = columns
        self.prob = probability
        self.actions = [0,1,2,3]
        self.directions = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
        self.time = 0
        self.d = ['N', 'E', 'S','W']

    def move(self, start, movement):
        x, y = start
        dx, dy = movement
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            # print("Now in environment state:", new_x, new_y)
            return (new_x,new_y)
        else:
            return start

    # Takes in the state, action and rate and gives the new state and time taken to make the transition in the 
    # gridworld
    def step(self, state, action, rate):
        x, y = state
        timetaken = np.random.exponential(scale= 1/rate) #sampling time taken from an exponential distribution
        if np.random.rand() >= (1 - self.prob): # probability of taking the action
            if action == 0:
                movement = self.directions['N']
                x1,y1 = self.move((x, y), movement)
            elif action == 1:
                movement = self.directions['E']
                x1, y1 = self.move((x, y), movement)
            
            elif action == 2:
                movement = self.directions['W']
                x1,y1 = self.move((x, y), movement)
            elif action == 3:
                movement = self.directions['S']
                x1, y1 = self.move((x, y), movement)
            else: 
                print("Error, given direction is not enabled")
            return x1, y1,  timetaken
        else:
            return x,y, timetaken
    
    def next_states(self,state):
        x,y = state
        next_states = []
        for movement in self.directions.values():
             next_states.append(self.move((x, y), movement))
        return next_states


    def reset(self):
        self.time = 0
        return self.state


    def next_state(self, state, action):
            x, y = state
            if action == 0:
                movement = self.directions['N']
                x1,y1 = self.move((x, y), movement)
            elif action == 1:
                movement = self.directions['E']
                x1, y1 = self.move((x, y), movement)
            
            elif action == 2:
                movement = self.directions['W']
                x1,y1 = self.move((x, y), movement)
            elif action == 3:
                movement = self.directions['S']
                x1, y1 = self.move((x, y), movement)
            else: 
                print("Error, given direction is not enabled")
            return x1, y1
        
        