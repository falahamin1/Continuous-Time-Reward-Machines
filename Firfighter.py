import numpy as np
from Gridworld import Gridworld

class FireFighterEnv():
    def __init__(self, rows, cols, probability):
        self.initstate1 = (0,cols//2, 0) # starting points
        self.initstate2 = (0,cols//2 + 1,0)  
        self.state1 = self.initstate1
        self.state2 = self.initstate2
        self.turn = 0 #This state indicates the turn of the fire engine
        self.initstate = self.initstate1 + self.initstate2 + (self.turn,)
        self.actions = [0,1,2,3] # 0:north, 1:east, 2: west, 3:south
        self.time = 0
        self.env1 = Gridworld(rows,cols, probability)
        self.env2 = Gridworld(rows,cols, probability)
        self.target = (rows-1, cols-1)
        self.statesize = 7
        self.actionsize = 4
        self.state = self.state1 + self.state2 + (self.turn,)
        self.initstate = self.initstate1 + self.initstate2 + (self.turn,)
        self.states = (rows **2) * (cols **2) * 2
        self.probability = probability

    def next_states(self, s):#provides the set of next states for value iteration
        next_states = []
        if s[6] == 0:#turn of player 1
            next_states1 = self.env1.next_states((s[0], s[1]))
            for nstate in next_states1:
                x, y = nstate 
                if (x,y) == self.target:
                    target = 1
                else:
                    target = 0
                nstate = (x,y,target)
                next_states.append(nstate + s[3:])
                next_states.append(nstate + s[3:-1] + ((s[-1] + 1) % 2,))
        else:
            next_states1 = self.env2.next_states((s[3], s[4]))
            for nstate in next_states1:
                x, y = nstate 
                if (x,y) == self.target:
                    target = 1
                else:
                    target = 0
                nstate = (x,y,target)
                next_states.append(s[:3] + nstate + (s[-1],))
                next_states.append(s[:3] + nstate + ((s[-1] + 1)%2,))
        return next_states


    def next_state(self, state, action):
        turn = state[-1]
        if turn == 0: 
            n_turn = (turn + 1) % 2 #change of turn
            x, y, target = state[:3]
            if target == 1: 
                n_state = state[:-1] + (n_turn,)
                return n_state 
            else:
                x1, y1 = self.env1.next_state((x,y), action)
                if (x1, y1) == self.target:
                    target = 1
                else: 
                    target = 0
                state1 = (x1, y1, target)   
                return state1 + state[3:-1] + (n_turn,)
        else:
            x, y, target = state[3:6]
            n_turn = (turn + 1) % 2
            if target == 1: 
                n_state = state[:-1] + (n_turn,)
                return n_state
            else:
                x1, y1 = self.env2.next_state((x,y), action)
                if (x1, y1) == self.target:
                    target = 1
                else: 
                    target = 0
                state2 = (x1,y1,target)
                return state[:3] + state2 +(n_turn,) 
        
        

    def step(self, action, rate):
        turn = self.turn
        if turn == 0: 
            x, y, target = self.state1
            if target == 1: 
                self.turn = (turn + 1) % 2
                self.state = self.state1 + self.state2 + (self.turn,)
                return self.state, 0 
            else:
                x1, y1, time = self.env1.step((x,y), action, rate)
                if (x1, y1) == self.target:
                    target = 1
                else: 
                    target = 0
                self.state1 = (x1, y1, target)   
        else:
            x, y, target = self.state2
            if target == 1: 
                self.turn = (turn + 1) % 2
                self.state = self.state1 + self.state2 + (self.turn,)
                return self.state, 0
            else:
                x1, y1, time = self.env2.step((x,y), action, rate)
                if (x1, y1) == self.target:
                    target = 1
                else: 
                    target = 0
                self.state2 = (x1, y1, target) 
        self.turn = (turn + 1) % 2    
        self.state = self.state1 + self.state2 + (self.turn,)
        self.time = self.time + time
        return self.state, self.time


    def reset(self):
        self.state1 = self.initstate1
        self.state2 = self.initstate2   # Reset to initial state
        self.turn = 0
        self.time = 0
        self.state = self.state1 + self.state2 + (self.turn,)
        return self.state