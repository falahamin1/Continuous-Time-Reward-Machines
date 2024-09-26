import numpy as np
from Gridworld import Gridworld

class FireFighterCarEnv():
    def __init__(self, rows, cols, probability):
        self.initstate1 = (0,0,0) # starting point of fire engine
        self.initstate2 = (0,cols - 1,0)  # starting point of car
        self.state1 = self.initstate1
        self.state2 = self.initstate2
        self.turn = 0 #This state indicates the turn of the player
        self.actions = list(range(17))  # 16 combination of actions representing actions of the two vehicles 
        self.time = 0
        self.env1 = Gridworld(rows, cols, probability)
        self.env2 = Gridworld(rows, cols, probability)
        self.target1 = (rows-1, cols-1) # Target of the fire engine
        self.target2 = (rows-1, 0)
        self.statesize = 7
        self.actionsize = 16
        self.state = self.state1 + self.state2 
        self.initstate = self.initstate1 + self.initstate2 # No turn variable, synchronous transition
        self.states = (rows ** 2) * (cols **2) 
        self.probability = probability

    def next_states(self, s):   #provides the set of next states of every action for value iteration
        next_states = []
        next_states1 = self.env1.next_states((s[0], s[1]))
        next_states2 = self.env2.next_states((s[3], s[4]))
        for nstate in next_states1:
            x, y = nstate 
            if (x,y) == self.target1:
                    target1 = 1
            else:
                    target1 = 0
            nstate1 = (x,y,target1)
            for nstate2 in next_states2:
                x1, y1 = nstate2
                if (x1, y1) == self.target2:
                    target2 = 1
                else:
                    target2 = 0
                nstate2 = (x1,y1,target2)
                state = nstate1 + nstate2
                next_states.append(state)
        return next_states
    def convert_action(self,action):
        if 0 <= action <= 16:
            first_value = action // 5  # Integer division by 5 to get the first value (row)
            second_value = action % 5  # Modulo 5 to get the second value (column)
            return first_value, second_value
        else:
            raise ValueError("Index must be between 0 and 16 inclusive.")

    def next_state(self, state, action): # provides a dictionary that gives the next state of an action and also the probability of transition
        action1, action2 = self.convert_action(action)
        next_states = {} 
        next_states[state] = 1 - (self.probability * self.probability)
        x,y,target = state[:3]
        x1,y1,target1 = state[3:6]
        if target == 1: 
            next_state1 = (x,y,target)
        else:
            xn1,yn1 = self.env1.next_state((x,y), action1)
            if (xn1, yn1) == self.target1:
                    target = 1
            else:
                    target = 0
            next_state1 = (xn1,yn1,target)
        
        if target1 == 1: 
            next_state2 = (x1,y1,target1)
        else: 
            xn2,yn2 = self.env2.next_state((x1,y1), action2)
            if (xn2, yn2) == self.target2:
                    target = 1
            else:
                    target = 0
            next_state2 = (xn2,yn2,target)
        
        next_state = next_state1 + next_state2
        if target == 1:
            if target1 == 1:
                next_states[state] = 1
            else: 
                next_states[state] = 1 - self.probability
                next_states[next_state] = self.probability
        else:
            if target1 == 1: 
                next_states[state] = 1 - self.probability
                next_states[next_state] = self.probability
            else: 
                next_states[next_state] = self.probability * self.probability
        if next_states[state] + next_states[next_state] == 1: 
            return next_states
            
        else:
            raise ValueError("Next state error, distribution is not valid") 
        
        

    def step(self, action, rate):
        action1, action2 = self.convert_action(action)
        x,y,target = self.state1
        x1,y1,target1 = self.state2
        if target == 1: 
            next_state = self.state1
        else: 
            xn, yn, time1 = self.env1.step((x,y), action1)
            if (xn,yn) == self.target1:
                target = 1
            else: 
                target = 0
            next_state = (xn,yn,target)
        if target1 == 1: 
            next_state1 = self.state2
        else: 
            xn1, yn1 = self.env2.step((x1,y1), action2)
            if (xn1,yn1) == self.target2:
                target1 = 1
            else: 
                target1 = 0
            next_state1 = (xn1,yn1,target1)
        new_state = next_state + next_state1
        self.state = new_state
        self.state1 = next_state
        self.state2 = next_state1




        






        turn = self.turn
        if turn == 0: 
            x, y, target = self.state1
            if target == 1: 
                self.turn = (turn + 1) % 2
                self.state = self.state1 + self.state2 + (self.turn,)
                return self.state, 0 
            else:
                x1, y1, time = self.env1.step((x,y), action, rate)
                if (x1, y1) == self.target1:
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
                if (x1, y1) == self.target2:
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