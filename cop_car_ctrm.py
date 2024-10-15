import numpy as np

class CopCarCTRM:
    def __init__(self):
        self.states = [0,1,2]
        self.totalstates = 3
        self.initstate = 0 #initial state
        self.state = self.initstate
        self.function1 = {
    (0, 0): 0.03 , (0, 1): 0.03, (0, 2): 0.03, (0, 3): 0.2, (0, 4): 0.06, (0, 5): 0.06, (0, 6): 0.06,
    (1, 0): 0.03, (1, 1): 0.2, (1, 2): 0.2, (1, 3): 0.03, (1, 4): 0.03, (1, 5): 0.03, (1, 6): 0.03,
    (2, 0): 0.03, (2, 1): 0.06, (2, 2): 0.2, (2, 3): 0.03, (2, 4): 0.03, (2, 5): 0.06, (2, 6): 0.03,
    (3, 0): 0.03, (3, 1): 0.2, (3, 2): 0.06, (3, 3): 0.06, (3, 4): 0.2, (3, 5): 0.06, (3, 6): 0.03,
    (4, 0): 0.03, (4, 1): 0.06, (4, 2): 0.2, (4, 3): 0.06, (4, 4): 0.06, (4, 5): 0.2, (4, 6): 0.2,
    (5, 0): 0.2, (5, 1): 0.06, (5, 2): 0.2, (5, 3): 0.03, (5, 4): 0.03, (5, 5): 0.03, (5, 6): 0.03,
    (6, 0): 0.06, (6, 1): 0.03, (6, 2): 0.2, (6, 3): 0.06, (6, 4): 0.06, (6, 5): 0.2, (6, 6): 0.03
}
        self.function1 = {position: round(value * 3, 2) for position, value in self.function1.items()}
        self.function2 = {position: round(value * 10, 2) for position, value in self.function1.items()}
        # print("New reward machine")
    
    def transitionfunction(self, input_state): #Takes the transition in the reward machine and gives the reward
        if self.state == 0: 
            if input_state[2] > 0: 
                self.state = 1
                return 0
            else:
                self.state = 0
                return 0
        elif self.state == 1: 
            if input_state[3] > 0: 
                self.state = 2
                return 1
            else:
                self.state = 1
                return 0
        elif self.state == 2: 
            return None

    def transition_function_counterfactual(self, ctrmstate, input_state):
        if ctrmstate == 0: 
            if input_state[2] > 0: 
                next_state = 1
                return 0, next_state
            else:
                next_state = 0
                return 0, next_state
        elif ctrmstate == 1: 
            if input_state[3] > 0: 
                next_state = 2
                return 1, next_state
            else:
                next_state = 1
                return 0, next_state
        elif ctrmstate == 2: 
            return None, None


    def get_rate_counterfactual(self,ctrmstate, input_state):
        state = (input_state[0], input_state[1])
        if ctrmstate == 0:
            return self.function1[state]
        elif ctrmstate == 1: 
            return self.function2[state]

    def get_rate(self,input_state):
        # print("Input state is:", input_state)
        if self.state == 0:
            return self.function1[(input_state[0], input_state[1])]
        elif self.state == 1: 
            return self.function2[(input_state[0], input_state[1])]
        else: 
            return None
    


    def reset(self):
        self.state = self.initstate   # Reset to initial state
        return self.state

# # Example usage:
# example = RewardMachineCopCarUT()
# print("function1 at (0,3):", example.function2[(0, 3)])
# print("function2 at (0,3):", example.function2[(0, 3)])
