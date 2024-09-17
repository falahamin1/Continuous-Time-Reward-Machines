import numpy as np
import math
from collections import deque
class ValueIteration:
    def __init__(self, gamma=0.001, environment = None, ctrm = None):
        self.V = {}
        self.gamma = gamma
        self.env = environment
        self.ctrm = ctrm
        self.states = self.fill_states()
        self.ctrm_states = tuple(self.ctrm.states)
        self.fill_vtable()
        self.gamma = gamma




    def startegy_analaysis(self, action_selector):
        enable = True
        initstate = self.env.initstate + (self.ctrm.initstate,) # Initial state of the product
        while enable:
            delta = 0
            for state in self.V:   # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                a = action_selector(state)
                ctrm_state = state[-1]     #Get the CTRM state
                env_state = state[:-1]      # Get the environment state
                time = 1/self.ctrm.get_rate_counterfactual(ctrm_state,env_state) # Gets the rate
                next_state = self.env.next_state(env_state, a) # Gets the next state of the environment
                reward, ctrm_next = self.ctrm.transition_function_counterfactual(ctrm_state, next_state)
                next_state = next_state + (ctrm_next,)
                if reward is None:
                    action_value = 0
                else:
                    action_value = (self.env.probability * (reward + math.exp(-1 * time * self.gamma) * self.V[next_state])) 
                    + (1-self.env.probability) * math.exp(-1 * time * self.gamma) *  self.V[state]
                self.V[state] = action_value
                delta = max(delta, abs(v - self.V[state]))
                # print(f"Value of initial state: {self.V[initstate]}")
            # print (f"Value of delta: {delta}")
            # print(f"---------------Iteration {i}--------------------")
            # self.print_table()
            if delta < 0.01: 
                enable = False
        return self.V[initstate]



# This function returns the set of states of the environment
# Requirements: - Environment: A function next_states(state) that would provide the next states of the 
# input state
    def fill_states(self):
        initial_state = self.env.initstate
        state_set = {initial_state} #This set stores all the states seen so far
    # Initialize the queue with the initial state
        queue = deque([initial_state]) # Initializes the queue with the initial state
        while queue: # Runs until the queue is empty
            current_state = queue.popleft() #pops the state from the queue
            for next_state in self.env.next_states(current_state): #Gets the next states of the current state
                    if next_state not in state_set: #Checks if the state is seen or not
                        state_set.add(next_state)  # Adds the unseen state to the state set
                        queue.append(next_state) #Adds to the queue to check for unseen next states
        return state_set

# The function fills the V table with the states of the product and gives them initial value 0
# Requirements: ctrm: A variable states with the set of states
    def fill_vtable(self): 
        for state in self.states:
                for ctrm_state in self.ctrm.states:  
                    s = state + (ctrm_state,)
                    self.V[s]= 0 


# Returns the value of the initial state 
    def value_iteration(self):
        enable = True
        initstate = self.env.initstate + (self.ctrm.initstate,) # Initial state of the product
        while enable:
            delta = 0
            for state in self.V:   # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                max_value = float('-inf')  
                for a in self.env.actions: 
                    ctrm_state = state[-1]     #Get the CTRM state
                    env_state = state[:-1]      # Get the environment state
                    time = 1/self.ctrm.get_rate_counterfactual(ctrm_state,env_state) # Gets the rate
                    next_state = self.env.next_state(env_state, a) # Gets the next state of the environment
                    reward, ctrm_next = self.ctrm.transition_function_counterfactual(ctrm_state, next_state)
                    next_state = next_state + (ctrm_next,)
                    if reward is None:
                        action_value = 0
                    else:
                        action_value = (self.env.probability * (reward + math.exp(-1 * time * self.gamma) * self.V[next_state])) 
                    + (1-self.env.probability) * math.exp(-1 * time * self.gamma) *  self.V[state]
                    
                    if action_value >= max_value:
                        max_value = action_value
                self.V[state] = max_value
                delta = max(delta, abs(v - self.V[state]))
                # print(f"Value of initial state: {self.V[initstate]}")
            # print (f"Value of delta: {delta}")
            # print(f"---------------Iteration {i}--------------------")
            # self.print_table()
            if delta < 0.01: 
                enable = False
        return self.V[initstate]
    
    def print_table(self,indent=0, separator='=>'):
        indent_str = ' ' * indent
        for key, value in self.V.items():
                print(f"{indent_str}{key}{separator} {value}")
            


    
    


   