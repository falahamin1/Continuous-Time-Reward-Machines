import numpy as np
from Gridworld import Gridworld
class TreasureMapEnv():
    def __init__(self, probability): 
        self.initstate = (0,0,0,0,0,0,0)  #Starting at (0,0) with atomic propositions map, tool, vehicle, treasure, jeweller set to 0
        self.state = self.initstate
        self.actions = [0,1,2,3] # 0:north, 1:east, 2: west, 3:south
        # self.actionmapping = {'north': 0, 'east': 1, 'west': 2, 'south': 3}
        self.map = (6,6) # position of map
        self.tool = (11, 6) # position of the tool
        self.vehicle = (1, 6) # position of vehicle 
        self.treasure = (6, 11)
        self.jeweller = (6, 1)
        self.time = 0
        self.env = Gridworld(13,13, probability) #13 rows and columns in the gridworld
        self.statesize = 7
        self.actionsize = 4
        self.states = 13 * 13 #total number of states
        self.probability = probability

    def check_ap (self, s): # provides the x,y position and returns the corresponding atomic propositions for that
        x , y = s
        mapap, tool, vehicle, treasure, jeweller = 0, 0, 0, 0, 0
        if (x,y) == self.map:
            mapap = 1
        else: 
            mapap = 0
        if (x,y) == self.tool:
            tool = 1
        else: 
            tool = 0
        if (x,y) == self.vehicle:
            vehicle = 1
        else: 
            vehicle = 0
        if (x,y) == self.treasure:
            treasure = 1
        else: 
            treasure = 0
        if (x,y) == self.jeweller:
            jeweller = 1
        else: 
            jeweller = 0
        return (mapap,tool,vehicle,treasure,jeweller)
        
            
    def next_states(self, s):#provides the set of next states for value iteration
        next_states = self.env.next_states((s[0], s[1]))
        nstates = []
        for nstate in next_states:
            x, y = nstate
            ap = self.check_ap(nstate)
            state = nstate +  ap
            nstates.append(state)
            
        return nstates
    

    def step(self, action, rate):
        x, y, mapap, tool, vehicle, treasure, jeweller = self.state
        x1, y1, time = self.env.step((x,y), action, rate)
        ap = self.check_ap ((x1,y1))
        self.state = (x1, y1) + ap
        self.time = self.time + time
        return self.state, time


    def next_state(self,state, action):
        next_states = {}
        next_states[state] = 1 - self.probability
        x, y, mapap, tool, vehicle, treasure, jeweller = state
        x1, y1 = self.env.next_state((x,y), action)
        ap = self.check_ap ((x1,y1))
        state = (x1, y1) + ap
        next_states[state] = self.probability
        return next_states

    def reset(self):
        self.state = self.initstate   # Reset to initial state
        self.time = 0
        return self.state