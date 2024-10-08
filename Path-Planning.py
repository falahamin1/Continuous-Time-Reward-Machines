import numpy as np
from Gridworld import Gridworld

class PathPlanning():
    def __init__(self, rows, cols, num_obstacles,  probability):
        self.initstate1 = (0,0,0) 
        self.initstate2 = (0,cols - 1,0)  
        self.initstate3 = (rows - 1,cols - 1, 0)
        self.initstate4 = (rows - 1, 0, 0)
        self.state1 = self.initstate1
        self.state2 = self.initstate2
        self.state3 = self.initstate3
        self.state4 = self.initstate4
        self.actions = list(range(16))  # 16 combination of actions representing actions of the two vehicles 
        self.time = 0
        self.env1 = Gridworld(rows, cols, probability)
        self.env2 = Gridworld(rows, cols, probability)
        self.env3 = Gridworld(rows, cols, probability)
        self.env4 = Gridworld(rows, cols, probability)
        self.target1 = ((rows - 1)//2, cols - 1)
        self.target2 = (rows-1, (cols - 1)//2)
        self.target3 = ((rows - 1) // 2, 0)
        self.target4 = (0, (cols - 1) // 2)
        self.statesize = 12
        self.actionsize = 4 ** 4
        self.state = self.state1 + self.state2 + self.state3 + self.state3 + self.state4
        self.initstate = self.initstate1 + self.initstate2 + self.initstate3 + self.initstate4 # No turn variable, synchronous transition
        self.states = (rows ** 2) * (cols ** 2) 
        self.probability = probability
    # def intialize_sources():

    def next_states(self, s):   #provides the set of next states of every action for value iteration
        next_states = []
        next_states1 = self.env1.next_states((s[0], s[1]))
        next_states2 = self.env2.next_states((s[3], s[4]))
        next_states3 = self.env3.next_states((s[6], s[7]))
        next_states4 = self.env4.next_states((s[9], s[10]))
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
                for nstate3 in next_states3:
                    x2, y2 = nstate3 
                    if (x2,y2) == self.target3:
                        target3 = 1
                    else:
                        target3 = 0
                    nstate3 = (x2,y2,target2)
                    for nstate4 in next_states4:
                        x3, y3 = nstate4
                        if (x3,y3) == self.target4:
                            target3 = 1 
                        else: 
                            target3 = 0
                        nstate4 = (x3,y3,target3)
                        state = nstate + nstate1 + nstate2 + nstate3 + nstate4
                        next_states.append(state)
        return next_states
    
    
    
    def convert_action(self,action):
        if 0 <= action <= 255:
            first_value = action // (4 * 4 * 4)  # First value (highest place)
            second_value = (action // (4 * 4)) % 4  # Second value
            third_value = (action // 4) % 4  # Third value
            fourth_value = action % 4  # Fourth value (lowest place)
            return first_value, second_value, third_value, fourth_value
        else:
            raise ValueError("Index must be between 0 and 16 inclusive.")

    def next_state(self, state, action): # provides a dictionary that gives the next state of an action and also the probability of transition
        action1, action2, action3, action4 = self.convert_action(action)
        next_states = {} 
        progress = [1,1,1,1]
        probability = 1
        # next_states[state] = 1 - (self.probability * self.probability)
        self.fill_next_state(next_states, state, action, progress, probability)
        total_prob = sum(next_states.values())  # Sum all the values in the dictionary
    
    # Check if the total is approximately 1 (due to floating-point precision issues)
        if abs(total_prob - 1) < 1e-6:
            print("Wrong distribution:",total_prob)       
        return next_states

    def fill_next_state(self, next_states, state,action, progress, probability):
        action1, action2, action3, action4 = self.convert_action(action)
        if progress[3] == 0: #all transitions done 
            self.add_probability_to_key(next_states, state, probability)
        elif progress[2] == 0: 
            progress[3] = 0
            x4,y4,t4 = state[9:12]
            if t4 == 1:
                self.fill_next_state(next_states,state,action, progress,probability)
            else:
                xn4,yn4 = self.env4.next_state((x4,y4), action4)
                if (xn4, yn4) == self.target4:
                    tn4 = 1
                else:
                    tn4 = 0
                self.fill_next_state(next_states, state[:9] + (x4,y4,t4), progress, probability * (1-self.probability))
                self.fill_next_state(next_states, state[:9] + (xn4,yn4,tn4), progress, probability * (self.probability))
        elif progress[1] == 0:
            progress[2] = 0
            x3,y3,t3 = state[6:9]
            if t3 == 1:
                self.fill_next_state(next_states,state,action, progress,probability)
            else:
                xn3,yn3 = self.env3.next_state((x3,y3), action3)
                if (xn3, yn3) == self.target3:
                    tn3 = 1
                else:
                    tn3 = 0
                self.fill_next_state(next_states, state[:6] + (x3,y3,t3) + state[9:12], progress, probability * (1-self.probability))
                self.fill_next_state(next_states, state[:6] + (xn3,yn3,tn3)+ state[9:12], progress, probability * (self.probability))
        
        elif progress[0] == 0:

            progress[1] = 0
            x2,y2,t2 = state[3:6]
            if t2 == 1:
                self.fill_next_state(next_states,state,action, progress,probability)
            else:
                xn2,yn2 = self.env2.next_state((x2,y2), action2)
                if (xn2, yn2) == self.target2:
                    tn2 = 1
                else:
                    tn2 = 0
                self.fill_next_state(next_states, state[:3] + (x2,y2,t2) + state[6:12], progress, probability * (1-self.probability))
                self.fill_next_state(next_states, state[:3] + (xn2,yn2,tn2)+ state[6:12], progress, probability * (self.probability))
        
        else:

            progress[0] = 0
            x1,y1,t1 = state[:3]
            if t1 == 1:
                self.fill_next_state(next_states,state,action, progress,probability)
            else:
                xn1,yn1 = self.env1.next_state((x1,y1), action1)
                if (xn1, yn1) == self.target1:
                    tn1 = 1
                else:
                    tn1 = 0
                self.fill_next_state(next_states, (x1,y1,t1) + state[3:12], progress, probability * (1-self.probability))
                self.fill_next_state(next_states, (xn1,yn1,tn1)+ state[3:12], progress, probability * (self.probability))



                



    def add_probability_to_key(self,next_states, key, value):
        if key in next_states:
            next_states[key] += value  # Add to the existing value
        else:
            next_states[key] = value  # Set the new value
            
            
        
        

    def step(self, action, rate):
        action1, action2, action3, action4 = self.convert_action(action)
        x1,y1,target1 = self.state1
        x2,y2,target2 = self.state2
        x3,y3,target3 = self.state3
        x4,y4,target4 = self.state4
        if target1 == 1: 
            next_state1 = self.state1
            time1 = 0
        else: 
            xn1, yn1, time1 = self.env1.step((x,y), action1, rate)
            if (xn1,yn1) == self.target1:
                target1 = 1
            else: 
                target1 = 0
            next_state1 = (xn,yn,target)
        if target2 == 1: 
            next_state2 = self.state2
            time2 = 0
        else: 
            xn2, yn2, time2 = self.env2.step((x2,y2), action2, rate)
            if (xn2,yn2) == self.target2:
                target2 = 1
            else: 
                target2 = 0
            next_state2 = (xn2,yn2,target2)
        
        if target3 == 1: 
            next_state3 = self.state3
            time3 = 0
        else: 
            xn3, yn3, time3 = self.env3.step((x3,y3), action3, rate)
            if (xn3,yn3) == self.target3:
                target3 = 1
            else: 
                target3 = 0
            next_state3 = (xn3,yn3,target3)

        if target4 == 1: 
            next_state4 = self.state4
            time4 = 0
        else: 
            xn4, yn4, time4 = self.env4.step((x4,y4), action4, rate)
            if (xn4,yn4) == self.target4:
                target4 = 1
            else: 
                target4 = 0
            next_state4 = (xn4,yn4,target4)

        new_state = next_state1 + next_state2 + next_state3 + next_state4
        self.state = new_state
        self.state1 = next_state1
        self.state2 = next_state2
        self.state3 = next_state3
        self.state4 = next_state4
        time = max(time1, time2,time3,time4)
        return self.state, time


    def reset(self):
        self.state1 = self.initstate1
        self.state2 = self.initstate2 
        self.state3 = self.initstate3   # Reset to initial state
        self.state4 = self.initstate4
        self.time = 0
        self.state = self.state1 + self.state2 + self.state3 + self.state4
        return self.state