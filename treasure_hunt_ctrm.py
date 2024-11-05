import numpy as np

class TreasureMapCTRM:
    def __init__(self):
        self.states = [0,1,2,3,4,5,6]
        self.actions = [0,1,2,3]
        self.totalstates = 7
        self.initstate = 0 #initial state
        self.state = self.initstate
        self.function1 = self.generate_rates()
        self.function1 = {position: round(value * 0.04, 3) for position, value in self.function1.items()}
        self.function2 = {position: round(value * 5, 2) for position, value in self.function1.items()}
        self.reward_mag = 1000
    
    def generate_rates(self):
        function1 = {
    (0, 0): 0.06,(0, 1): 0.2,(0, 2): 0.03,(0, 3): 0.2,(0, 4): 0.06,(0, 5): 0.06,
 (0, 6): 0.03,(0, 7): 0.06,(0, 8): 0.03,(0, 9): 0.06,(0, 10): 0.06,(0, 11): 0.06,(0, 12): 0.06,(1, 0): 0.03,(1, 1): 0.03,(1, 2): 0.03,
 (1, 3): 0.06,(1, 4): 0.2,(1, 5): 0.2,(1, 6): 0.06,(1, 7): 0.03,(1, 8): 0.2,(1, 9): 0.03,(1, 10): 0.06,(1, 11): 0.2,(1, 12): 0.2,
 (2, 0): 0.2,(2, 1): 0.06,(2, 2): 0.03,(2, 3): 0.03,(2, 4): 0.06,(2, 5): 0.06,(2, 6): 0.03,(2, 7): 0.06,(2, 8): 0.06,(2, 9): 0.03,
 (2, 10): 0.2,(2, 11): 0.06,(2, 12): 0.06,(3, 0): 0.03,(3, 1): 0.03,(3, 2): 0.03,(3, 3): 0.03,(3, 4): 0.06,(3, 5): 0.2,
 (3, 6): 0.2,(3, 7): 0.03,(3, 8): 0.06,(3, 9): 0.03,(3, 10): 0.03,(3, 11): 0.2,(3, 12): 0.06,(4, 0): 0.2,(4, 1): 0.06,
 (4, 2): 0.2,(4, 3): 0.03,(4, 4): 0.06,(4, 5): 0.03,(4, 6): 0.03,(4, 7): 0.03,(4, 8): 0.06,(4, 9): 0.2,(4, 10): 0.2,(4, 11): 0.03,
 (4, 12): 0.2,(5, 0): 0.06,(5, 1): 0.03,(5, 2): 0.06,(5, 3): 0.03,(5, 4): 0.2,(5, 5): 0.03,(5, 6): 0.06,(5, 7): 0.03,(5, 8): 0.03,
 (5, 9): 0.2,(5, 10): 0.03,(5, 11): 0.03,(5, 12): 0.06,(6, 0): 0.03,(6, 1): 0.06,(6, 2): 0.2,(6, 3): 0.06,(6, 4): 0.06,(6, 5): 0.06,(6, 6): 0.03,
 (6, 7): 0.03,(6, 8): 0.06,(6, 9): 0.03,(6, 10): 0.03,(6, 11): 0.06,(6, 12): 0.2,(7, 0): 0.2,(7, 1): 0.06,(7, 2): 0.06,(7, 3): 0.06,(7, 4): 0.2,
 (7, 5): 0.2,(7, 6): 0.06,(7, 7): 0.2,(7, 8): 0.03,(7, 9): 0.06,(7, 10): 0.06,(7, 11): 0.2,(7, 12): 0.03,(8, 0): 0.06,(8, 1): 0.2,(8, 2): 0.03,
(8, 3): 0.2,(8, 4): 0.06,(8, 5): 0.06,(8, 6): 0.2,(8, 7): 0.06,(8, 8): 0.03,(8, 9): 0.2,(8, 10): 0.03,(8, 11): 0.06,(8, 12): 0.2,(9, 0): 0.06,
 (9, 1): 0.2,(9, 2): 0.2,(9, 3): 0.2,(9, 4): 0.03,(9, 5): 0.2,(9, 6): 0.03,(9, 7): 0.2,(9, 8): 0.06,(9, 9): 0.06,(9, 10): 0.06,(9, 11): 0.03,
 (9, 12): 0.03,(10, 0): 0.2,(10, 1): 0.2,(10, 2): 0.2,(10, 3): 0.06,(10, 4): 0.06,(10, 5): 0.2,(10, 6): 0.2,(10, 7): 0.2,(10, 8): 0.03,(10, 9): 0.06,
 (10, 10): 0.2,(10, 11): 0.2,(10, 12): 0.2,(11, 0): 0.06,(11, 1): 0.06,(11, 2): 0.06,(11, 3): 0.2,(11, 4): 0.2,(11, 5): 0.06,(11, 6): 0.2,(11, 7): 0.06,(11, 8): 0.2,(11, 9): 0.03,(11, 10): 0.06,(11, 11): 0.2,
 (11, 12): 0.03,(12, 0): 0.03,(12, 1): 0.06,(12, 2): 0.2,(12, 3): 0.06,(12, 4): 0.2,(12, 5): 0.06,(12, 6): 0.03,(12, 7): 0.06,(12, 8): 0.2,(12, 9): 0.03,(12, 10): 0.03,(12, 11): 0.06,(12, 12): 0.03}
        rates_with_actions = {}
        for (state1, state2), base_value in function1.items():
            for action in self.actions:
                rates_with_actions[(state1, state2, action)] = self.deterministic_transformation(base_value, action)
        
        return rates_with_actions

    def deterministic_transformation(self, base_value, action):
        return round(base_value * (1 + 0.05 * action), 2)


  # Ordering of atomic propositions :   map, tool, vehicle, treasure, jeweller
    def transitionfunction(self, input_state): #Takes the transition in the reward machine and gives the reward
        if self.state == 0: 
            if input_state[2] > 0: 
                self.state = 1
                return 0 * self.reward_mag
            else:
                self.state = 0
                return 0 * self.reward_mag
        elif self.state == 1: 
            if input_state[3] > 0: 
                self.state = 2
                return 0.4 * self.reward_mag
            elif input_state[4] > 0:
                self.state = 3
                return 0.2 * self.reward_mag
            else: 
                return 0 * self.reward_mag
        elif self.state == 2: 
            if input_state[5] > 0: 
                self.state = 5
                return 0 * self.reward_mag
            else:
                return 0* self.reward_mag
        elif self.state == 3: 
            if input_state[5]>0:
                self.state = 4
                return 0 * self.reward_mag
            else:
                return 0 * self.reward_mag
        elif self.state ==4: 
            if input_state[6] > 0 : 
                self.state = 6
                return 1 * self.reward_mag
            else: 
                return 0 * self.reward_mag
        elif self.state == 5:
            if input_state[6] > 0:
                self.state = 6
                return 1 * self.reward_mag
            else:
                return 0 * self.reward_mag
        elif self.state == 6: 
            return None

    def transition_function_counterfactual(self, ctrmstate, input_state):
        reward = None
        next_state = None
        if ctrmstate == 0: 
            if input_state[2] > 0: 
                next_state = 1
                reward = 0 * self.reward_mag
            else:
                next_state = 0
                reward = 0 * self.reward_mag
        elif ctrmstate == 1: 
            if input_state[3] > 0: 
                next_state = 2
                reward = 0.4 * self.reward_mag
            elif input_state[4] > 0:
                next_state = 3
                reward = 0.2 * self.reward_mag
            else: 
                next_state = 1
                reward = 0 * self.reward_mag
        elif ctrmstate == 2: 
            if input_state[5] > 0: 
                next_state = 5
                reward = 0 * self.reward_mag
            else:
                next_state = 2
                reward = 0 * self.reward_mag
        elif ctrmstate == 3: 
            if input_state[5]>0:
                next_state = 4
                reward = 0 * self.reward_mag
            else:
                next_state = 3
                reward = 0 * self.reward_mag
        elif ctrmstate ==4: 
            if input_state[6] > 0 : 
                next_state = 6
                reward = 1 * self.reward_mag
            else: 
                next_state = 4
                reward = 0 * self.reward_mag
        elif ctrmstate == 5:
            if input_state[6] > 0:
                next_state = 6
                reward = 1 * self.reward_mag
            else:
                next_state = 5
                reward = 0 * self.reward_mag
                
        elif ctrmstate == 6: 
            next_state = None
            reward = None
        return reward, next_state


    def get_rate_counterfactual(self,ctrmstate, input_state, action):
        state = (input_state[0], input_state[1], action)
        if ctrmstate == 3 or ctrmstate == 4:
            return self.function2[state]
        elif ctrmstate == 6: 
            return None
        else: 
            return self.function1[state]

    def get_rate(self,input_state, action):
        # print("Input state is:", input_state)
        if self.state == 3 or self.state == 4:
            return self.function2[(input_state[0], input_state[1],action)]
        elif self.state ==6: 
            return None
        else:
            return self.function1[(input_state[0], input_state[1],action)]
    
    def transition_VI(self,ctrmstate,next_state):
        reward = None
        next_state = None
        if ctrmstate == 0: 
            if next_state == 1: 
                reward = 0 * self.reward_mag
            else:
                reward = 0 * self.reward_mag
        elif ctrmstate == 1: 
            if next_state == 2: 
                reward = 0.4 * self.reward_mag
            elif next_state == 3:
                reward = 0.2 * self.reward_mag
            else: 
                reward = 0 * self.reward_mag
        elif ctrmstate == 2: 
            if next_state == 5: 
                reward = 0 * self.reward_mag
            else:
                reward = 0 * self.reward_mag
        elif ctrmstate == 3: 
            if next_state == 4:
                reward = 0 * self.reward_mag
            else:
                reward = 0 * self.reward_mag
        elif ctrmstate ==4: 
            if next_state == 6: 
                reward = 1 * self.reward_mag
            else: 
                reward = 0 * self.reward_mag
        elif ctrmstate == 5:
            if next_state == 6:
                reward = 1 * self.reward_mag
        elif ctrmstate == 6: 
            reward = None
        return reward

    def next_states(self,ctrmstate):
        next_states= []
        if ctrmstate == 0: 
            next_states.extend([0,1])
        elif ctrmstate == 1: 
            next_states.extend([1,2,3])
        elif ctrmstate == 2: 
            next_states.extend([2,5])
        elif ctrmstate == 3: 
            next_states.extend([3,4])
        elif ctrmstate ==4: 
            next_states.extend([4,6])
        elif ctrmstate == 5:
            next_states.extend([5,6])
        elif ctrmstate == 6: 
            next_states = None
        return next_states


    def reset(self):
        self.state = self.initstate   # Reset to initial state
        return self.state

