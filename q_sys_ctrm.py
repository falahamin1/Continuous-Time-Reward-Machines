import numpy as np

class QSysCTRM:
    def __init__(self,arrival_rate, service_rate, n_servers, server_cost, q_penalty, jobs, servers):
        self.states = [0,1,2,3]
        self.actions = range(1,n_servers+1)
        self.totalstates = n_servers
        self.initstate = 0 #initial state
        self.state = self.initstate
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server_cost = server_cost
        self.q_penalty = q_penalty
        self.jobs = jobs
        self.servers = servers
        self.speedup = 2
        # print("New reward machine")
    


    def generate_rates(self):
        function1 = {
    (0, 0): 0.03 , (0, 1): 0.03, (0, 2): 0.03, (0, 3): 0.2, (0, 4): 0.06, (0, 5): 0.06, (0, 6): 0.06,
    (1, 0): 0.03, (1, 1): 0.2, (1, 2): 0.2, (1, 3): 0.06, (1, 4): 0.03, (1, 5): 0.03, (1, 6): 0.03,
    (2, 0): 0.03, (2, 1): 0.06, (2, 2): 0.2, (2, 3): 0.03, (2, 4): 0.03, (2, 5): 0.06, (2, 6): 0.03,
    (3, 0): 0.03, (3, 1): 0.2, (3, 2): 0.06, (3, 3): 0.06, (3, 4): 0.2, (3, 5): 0.06, (3, 6): 0.03,
    (4, 0): 0.03, (4, 1): 0.06, (4, 2): 0.2, (4, 3): 0.06, (4, 4): 0.2, (4, 5): 0.2, (4, 6): 0.2,
    (5, 0): 0.2, (5, 1): 0.06, (5, 2): 0.2, (5, 3): 0.03, (5, 4): 0.2, (5, 5): 0.03, (5, 6): 0.03,
    (6, 0): 0.06, (6, 1): 0.03, (6, 2): 0.2, (6, 3): 0.06, (6, 4): 0.2, (6, 5): 0.2, (6, 6): 0.03
}
        rates_with_actions = {}
        for (state1, state2), base_value in function1.items():
            for action in self.actions:
                rates_with_actions[(state1, state2, action)] = self.deterministic_transformation(base_value, action)
        
        return rates_with_actions

    def deterministic_transformation(self, base_value, action):
        return round(base_value * (1 + 0.1 * action), 2)


    def transitionfunction(self, input_state): #Takes the transition in the reward machine and gives the reward
        s,k,q = input_state
        reward =-1 *( self.server_cost * k + self.q_penalty * s)
        if self.state == 0: 
            if q == self.jobs:
                self.state = 3
                return 1
            elif s > 3: 
                self.state = 1
                return reward
            else:
                self.state = 0
                return reward
        elif self.state == 1:
            if q == self.jobs: 
                self.state = 3
                return 1  
            elif s > 5: 
                self.state = 2
                return reward
            else:
                self.state = 1
                return reward
        elif self.state == 2:
            if q == self.jobs: 
                self.state = 3
                return 1  
            else:
                self.state = 2
                return reward 
        else:
            return None

    def transition_function_counterfactual(self, ctrmstate, input_state):
        s,k,q = input_state
        reward =-1 *( self.server_cost * k + self.q_penalty * s)
        if ctrmstate == 0: 
            if q == self.jobs:
                next_state = 3
                reward = 1
            elif s > 3: 
                next_state = 1                
            else:
                next_state = 0
        elif self.state == 1:
            if q == self.jobs: 
                next_state = 3
                reward = 1  
            elif s > 5: 
                next_state = 2
               
            else:
                next_state = 1

        elif self.state == 2:
            if q == self.jobs: 
                next_state = 3
                reward = 1  
            else:
                next_state = 2
                 
        else:
            next_state = None
            reward = None 
        return reward, next_state


    def get_rate_counterfactual(self,ctrmstate, input_state, action):
        s,k,q = input_state
        service_rate = self.service_rate * action * (s/self.servers)
        if ctrmstate == 0:
            arrival_rate = self.arrival_rate
            return (arrival_rate, service_rate)
        elif ctrmstate == 1: 
            arrival_rate = self.arrival_rate * self.speedup 
            return (arrival_rate, service_rate)
        elif ctrmstate == 2:
            arrival_rate = self.arrival_rate * self.speedup * self.speedup
            return (arrival_rate, service_rate)
        else: 
            return None



    def get_rate(self,input_state, action):
        s,k,q = input_state
        service_rate = self.service_rate * action * (s/self.servers)
        if self.state == 0:
            arrival_rate = self.arrival_rate
            return (arrival_rate, service_rate)
        elif self.state == 1: 
            arrival_rate = self.arrival_rate * self.speedup 
            return (arrival_rate, service_rate)
        elif self.state == 2: 
            arrival_rate = self.arrival_rate * self.speedup * self.speedup
            return (arrival_rate, service_rate)
        else: 
            return None


    def transition_VI(self,ctrmstate,next_state):
        if ctrmstate == 0: 
            if next_state == 1:                 
                return 0
            else:
                return 0
        elif ctrmstate == 1: 
            if next_state == 2: 
                return 1
            else:
                return 0
        elif ctrmstate == 2: 
            return None
        

    def next_states(self,ctrmstate):
        next_states= []
        if ctrmstate == 0: 
            next_states.extend([0,1])
        elif ctrmstate == 1: 
            next_states.extend([1,2])
        elif ctrmstate == 2: 
            next_states = None
        return next_states
    


    def reset(self):
        self.state = self.initstate   # Reset to initial state
        return self.state

# # Example usage:
# example = RewardMachineCopCarUT()
# print("function1 at (0,3):", example.function2[(0, 3)])
# print("function2 at (0,3):", example.function2[(0, 3)])
