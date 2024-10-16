import numpy as np

class FireFighterCTRM:
    def __init__(self):
        self.states = [0,1,2,3]
        self.actions = [0,1,2,3]
        self.totalstates = 4
        self.initstate = 0 #initial state
        self.finalstate = 3
        self.state = self.initstate
        self.function1 = self.generate_rates()
        # self.function1 = {position: round(value * 3, 2) for position, value in self.function1.items()}
        self.function2 = {position: round(value * 10, 2) for position, value in self.function1.items()}
        # print("New reward machine")

    def generate_rates(self):
        function1 =  {
    (0, 0): 0.03 , (0, 1): 0.03, (0, 2): 0.03, (0, 3): 0.2, (0, 4): 0.06, (0, 5): 0.06, (0, 6): 0.06,
    (1, 0): 0.03, (1, 1): 0.2, (1, 2): 0.2, (1, 3): 0.03, (1, 4): 0.03, (1, 5): 0.03, (1, 6): 0.03,
    (2, 0): 0.03, (2, 1): 0.06, (2, 2): 0.2, (2, 3): 0.03, (2, 4): 0.03, (2, 5): 0.06, (2, 6): 0.03,
    (3, 0): 0.03, (3, 1): 0.2, (3, 2): 0.06, (3, 3): 0.06, (3, 4): 0.2, (3, 5): 0.06, (3, 6): 0.03,
    (4, 0): 0.03, (4, 1): 0.06, (4, 2): 0.2, (4, 3): 0.06, (4, 4): 0.06, (4, 5): 0.2, (4, 6): 0.2,
    (5, 0): 0.2, (5, 1): 0.06, (5, 2): 0.2, (5, 3): 0.03, (5, 4): 0.03, (5, 5): 0.03, (5, 6): 0.03,
    (6, 0): 0.06, (6, 1): 0.03, (6, 2): 0.2, (6, 3): 0.06, (6, 4): 0.06, (6, 5): 0.2, (6, 6): 0.03
}
        rates_with_actions = {}
        for (state1, state2), base_value in function1.items():
            for action in self.actions:
                rates_with_actions[(state1, state2, action)] = self.deterministic_transformation(base_value, action)
        
        return rates_with_actions

    def deterministic_transformation(self, base_value, action):
        return round(base_value * (1 + 0.2 * action), 2)
    
    def transitionfunction(self, input_state): #Takes the transition in the reward machine and gives the reward
        if self.state == 0: 
            if input_state[2] > 0 and input_state[5] > 0: #checks if player 1 reaches target
                self.state = 3
                return 1
            elif input_state[2] > 0:
                self.state = 1
                return 0
            elif input_state[5] > 0: 
                self.state = 2
                return 0
            else: 
                return 0

        elif self.state == 1: 
            if input_state[5] > 0: 
                self.state = 3
                return 1
            else:
                self.state = 1
                return 0
        
        elif self.state == 2: 
            if input_state[2] > 0: 
                self.state = 3
                return 1
            else:
                return 0
        else:
            return None

    def transition_function_counterfactual(self, ctrmstate, input_state):

        if ctrmstate == 0: 
            if input_state[2] > 0 and input_state[5] > 0: #checks if player 1 reaches target
                next_state = 3
                reward = 1
            elif input_state[2] > 0:
                next_state = 1
                reward = 0
            elif input_state[5] > 0: 
                next_state = 2
                reward = 0
            else: 
                next_state = 0
                reward = 0

        elif ctrmstate == 1: 
            if input_state[5] > 0: 
                next_state = 3
                reward = 1
            else:
                next_state = 1
                reward = 0
        
        elif ctrmstate == 2: 
            if input_state[2] > 0: 
                next_state = 3
                reward =  1
            else:
                next_state = 2
                reward = 0
        else:
            next_state = None
            reward = None
        return reward, next_state


    def get_rate_counterfactual(self,ctrmstate, input_state,action):
        x1, y1, target1, x2, y2, target2, turn = input_state
        if turn == 0: 
            if x1 in {x2, x2 + 1, x2 - 1} and y1 in {y2, y2 + 1, y2 - 1}:
                return self.function1[(x1,y1,action)]
            else: 
                return self.function2[(x1,y1,action)]
        
        else: 
            if x2 in {x1, x1 + 1, x1 - 1} and y2 in {y1, y1 + 1, y1 - 1}:
                return self.function1[(x2,y2,action)]
            else: 
                return self.function2[(x2,y2,action)]
        

    def get_rate(self,input_state,action):
        x1, y1, target1, x2, y2, target2, turn = input_state
        if turn == 0: 
            if x1 in {x2, x2 + 1, x2 - 1} and y1 in {y2, y2 + 1, y2 - 1}:
                return self.function1[(x1,y1,action)]
            else: 
                return self.function2[(x1,y1,action)]
        
        else: 
            if x2 in {x1, x1 + 1, x1 - 1} and y2 in {y1, y1 + 1, y1 - 1}:
                return self.function1[(x2,y2,action)]
            else: 
                return self.function2[(x2,y2,action)]

    


    def reset(self):
        self.state = self.initstate   # Reset to initial state
        return self.state
