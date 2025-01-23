import numpy as np
from Gridworld import Gridworld
import numpy as np
import random
from Gridworld import Gridworld

class FireFighterMultEnv():
    def __init__(self, rows, cols, probability, num_targets):
        self.initstate1 = (0, 0)  # Starting position of fire engine (no target in state)
        self.initstate2 = (0, cols - 1, 0)  # Starting position of car, including target indicator
        self.state1 = self.initstate1
        self.state2 = self.initstate2
        self.turn = 0  # Indicates the turn of the player

        # Generate random targets for the fire engine
        self.fire_engine_targets = self.generate_random_targets(rows, cols, num_targets)
        self.visited_targets = tuple(0 for _ in range(num_targets))  # Track visited targets (0: unvisited, 1: visited)

        self.actions = [0, 1, 2, 3]  # 0:north, 1:east, 2:west, 3:south
        self.time = 0
        self.env1 = Gridworld(rows, cols, probability)
        self.env2 = Gridworld(rows, cols, probability)
        self.target2 = (rows - 1, 0)  # Single target for car

        # Updated state structure without individual fire engine target tracking
        self.state = self.state1 + self.state2 + (self.turn,) + self.visited_targets
        self.statesize = len(self.state)
        self.probability = probability

    def generate_random_targets(self, rows, cols, num_targets):
        """Generates a random set of unique target positions in the grid."""
        targets = set()
        while len(targets) < num_targets:
            target = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            if target not in [(0, 0), (0, cols - 1)]:  # Avoid starting positions
                targets.add(target)
        return list(targets)

    def next_states(self, s):
        next_states = []
        if s[5] == 0:  # Turn of fire engine
            next_states1 = self.env1.next_states((s[0], s[1]))
            visited_targets = list(s[6:])  # Extract current visited status

            for nstate in next_states1:
                x, y = nstate
                new_visited_targets = visited_targets[:]
                if (x, y) in self.fire_engine_targets:
                    idx = self.fire_engine_targets.index((x, y))
                    new_visited_targets[idx] = 1  # Mark target as visited

                next_states.append(nstate + s[2:5] + tuple(new_visited_targets))
                next_states.append(nstate + s[2:4] + ((s[4] + 1) % 2,) + tuple(new_visited_targets))
        else:  # Turn of car
            next_states2 = self.env2.next_states((s[2], s[3]))
            target2_reached = 1 if (s[2], s[3]) == self.target2 else 0

            for nstate in next_states2:
                x, y = nstate
                next_states.append(s[:2] + (x, y, target2_reached) + (s[4],) + s[6:])
                next_states.append(s[:2] + (x, y, target2_reached) + ((s[4] + 1) % 2,) + s[6:])
        return next_states

    def next_state(self, state, action):
        next_states = {}
        next_states[state] = 1 - self.probability 
        turn = state[5]
        visited_targets = list(state[6:])

        if turn == 0:  # Fire engine move
            x, y = state[:2]
            n_turn = (turn + 1) % 2
            x1, y1 = self.env1.next_state((x, y), action)

            if (x1, y1) in self.fire_engine_targets:
                idx = self.fire_engine_targets.index((x1, y1))
                visited_targets[idx] = 1  # Mark target as visited

            new_state = (x1, y1) + state[2:5] + (n_turn,) + tuple(visited_targets)
        else:  # Car move
            x, y, target2 = state[2:5]
            n_turn = (turn + 1) % 2
            x1, y1 = self.env2.next_state((x, y), action)

            target2 = 1 if (x1, y1) == self.target2 else target2
            new_state = state[:2] + (x1, y1, target2) + (n_turn,) + state[6:]

        next_states[new_state] = self.probability
        return next_states

    def step(self, action, rate):
        turn = self.turn
        visited_targets = list(self.state[6:])

        if turn == 0:  # Fire engine move
            x, y = self.state1
            x1, y1, time = self.env1.step((x, y), action, rate)

            if (x1, y1) in self.fire_engine_targets:
                idx = self.fire_engine_targets.index((x1, y1))
                visited_targets[idx] = 1  # Mark the target as visited

            self.state1 = (x1, y1)
        else:  # Car move
            x, y, target2 = self.state2
            x1, y1, time = self.env2.step((x, y), action, rate)
            target2 = 1 if (x1, y1) == self.target2 else target2
            self.state2 = (x1, y1, target2)

        self.turn = (turn + 1) % 2
        self.state = self.state1 + self.state2 + (self.turn,) + tuple(visited_targets)
        self.time += time

        return self.state, time

    def reset(self):
        self.state1 = self.initstate1
        self.state2 = self.initstate2  # Reset to initial state
        self.turn = 0
        self.time = 0
        self.visited_targets = tuple(0 for _ in self.fire_engine_targets)
        self.state = self.state1 + self.state2 + (self.turn,) + self.visited_targets
        return self.state

# class FireFighterMultipleEnv():
#     def __init__(self, rows, cols, probability, fires = 5):
#         self.initstate1 = (0,) * (fires + 2) # starting point of fire engine
#         self.initstate2 = (0,cols - 1,0)  # starting point of car
#         self.state1 = self.initstate1
#         self.state2 = self.initstate2
#         self.turn = 0 #This state indicates the turn of the player
#         self.initstate = self.initstate1 + self.initstate2 + (self.turn,)
#         self.actions = [0,1,2,3] # 0:north, 1:east, 2: west, 3:south
#         self.time = 0
#         self.env1 = Gridworld(rows, cols, probability)
#         self.env2 = Gridworld(rows, cols, probability)
#         self.fires = fires
#         self.targets = self.get_random_coordinates(rows, cols, fires)
#         self.target1 = (rows-1, cols-1) # Target of the fire engine
#         self.target2 = (rows-1, 0)
#         self.statesize = 7
#         self.actionsize = 4
#         self.state = self.state1 + self.state2 + (self.turn,)
#         self.initstate = self.initstate1 + self.initstate2 + (self.turn,)
#         self.states = (rows ** 2) * (cols **2) 
#         self.probability = probability



#     def get_random_coordinates(self, rows, columns, num_points=5):
#         coordinates = set()
#         while len(coordinates) < num_points:
#             x = random.randint(0, rows - 1)
#             y = random.randint(0, columns - 1)
#             coordinates.add((x, y))
    
#         return list(coordinates)


#     def next_states(self, s):   #provides the set of next states of every action for value iteration
#         next_states = []
#         if s[6] == 0:#turn of player 1
#             next_states1 = self.env1.next_states((s[0], s[1]))
#             for nstate in next_states1:
#                 nstate = self.check_targets(nstate)
#                 next_states.append(nstate + s[3:])
#                 next_states.append(nstate + s[3:-1] + ((s[-1] + 1) % 2,))
#         else:
#             next_states1 = self.env2.next_states((s[3], s[4]))
#             for nstate in next_states1:
#                 x, y = nstate 
#                 if (x,y) == self.target2:
#                     target = 1
#                 else:
#                     target = 0
#                 nstate = (x,y,target)
#                 next_states.append(s[:3] + nstate + (s[-1],))
#                 next_states.append(s[:3] + nstate + ((s[-1] + 1)%2,))
#         return next_states
#     def check_targets(self, state):
#         coordinate = (state[0],state[1])
#         if coordinate in self.targets: 
#             index = self.targets.index(coordinate)
#             index += 2 #move two points 
#             state_list = list(state)
#             state_list[index] = 1 
#             n_state = tuple(state_list)
#             return n_state
#         else: 
#             return state



#     def next_state(self, state, action): # provides a dictionary that gives the next state of an action and also the probability of transition
#         next_states = {} 
#         next_states[state] = 1 - self.probability 
#         turn = state[-1]
#         if turn == 0: 
#             n_turn = (turn + 1) % 2 #change of turn
#             x, y, target = state[:3]
#             if target == 1: 
#                 n_state = state[:-1] + (n_turn,)
#             else:
#                 x1, y1 = self.env1.next_state((x,y), action)
#                 if (x1, y1) == self.target1:
#                     target = 1
#                 else: 
#                     target = 0
#                 state1 = (x1, y1, target)   
#                 n_state = state1 + state[3:-1] + (n_turn,)
#         else:
#             x, y, target = state[3:6]
#             n_turn = (turn + 1) % 2
#             if target == 1: 
#                 n_state = state[:-1] + (n_turn,)
#             else:
#                 x1, y1 = self.env2.next_state((x,y), action)
#                 if (x1, y1) == self.target2:
#                     target = 1
#                 else: 
#                     target = 0
#                 state2 = (x1,y1,target)
#                 n_state = state[:3] + state2 +(n_turn,) 
#         next_states[n_state] = self.probability
#         return next_states
        
        

#     def step(self, action, rate):
#         turn = self.turn
#         if turn == 0: 
#             x, y, target = self.state1
#             if target == 1: 
#                 self.turn = (turn + 1) % 2
#                 self.state = self.state1 + self.state2 + (self.turn,)
#                 return self.state, 0 
#             else:
#                 x1, y1, time = self.env1.step((x,y), action, rate)
#                 if (x1, y1) == self.target1:
#                     target = 1
#                 else: 
#                     target = 0
#                 self.state1 = (x1, y1, target)   
#         else:
#             x, y, target = self.state2
#             if target == 1: 
#                 self.turn = (turn + 1) % 2
#                 self.state = self.state1 + self.state2 + (self.turn,)
#                 return self.state, 0
#             else:
#                 x1, y1, time = self.env2.step((x,y), action, rate)
#                 if (x1, y1) == self.target2:
#                     target = 1
#                 else: 
#                     target = 0
#                 self.state2 = (x1, y1, target) 
#         self.turn = (turn + 1) % 2    
#         self.state = self.state1 + self.state2 + (self.turn,)
#         self.time = self.time + time
#         return self.state, time


#     def reset(self):
#         self.state1 = self.initstate1
#         self.state2 = self.initstate2   # Reset to initial state
#         self.turn = 0
#         self.time = 0
#         self.state = self.state1 + self.state2 + (self.turn,)
#         return self.state