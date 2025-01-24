import numpy as np
import random
from Gridworld import Gridworld

class FireFighterMultEnv():
    def __init__(self, rows, cols, probability):
        self.initstate1 = (0, 0)  # Fire engine start position
        self.initstate2 = (0, cols - 1, 0)  # Car start position with target indicator
        self.state1 = self.initstate1
        self.state2 = self.initstate2
        self.turn = 0  # 0: fire engine's turn, 1: car's turn

        # Fixed fire engine targets
        self.fire_engine_targets = [(1, 1), (3, 4), (rows - 2, cols - 2)] 
        self.visited_targets = (0, 0, 0)  # Track visited targets (0: unvisited, 1: visited)

        self.actions = [0, 1, 2, 3]  # 0:north, 1:east, 2:west, 3:south
        self.time = 0
        self.env1 = Gridworld(rows, cols, probability)
        self.env2 = Gridworld(rows, cols, probability)
        self.target2 = (rows - 1, 0)  # Single fixed target for the car

        # State initialization
        self.state = self.state1 + self.state2 + (self.turn,) + self.visited_targets
        self.initstate = self.state
        self.statesize = len(self.state)
        self.probability = probability

    def next_states(self, s):
        next_states = []
        if s[5] == 0:  # Fire engine's turn
            next_positions = self.env1.next_states((s[0], s[1]))
            visited_targets = list(s[6:])

            for nstate in next_positions:
                x, y = nstate
                new_visited_targets = visited_targets.copy()
                
                if (x, y) in self.fire_engine_targets:
                    idx = self.fire_engine_targets.index((x, y))
                    new_visited_targets[idx] = 1  # Mark target as visited
                
                next_states.append(nstate + s[2:6] + tuple(new_visited_targets))
                next_states.append(nstate + s[2:5] + ((s[5] + 1) % 2,) + tuple(new_visited_targets))
        else:  # Car's turn
            next_positions = self.env2.next_states((s[2], s[3]))

            for nstate in next_positions:
                x, y = nstate
                target2_reached = 1 if (x, y) == self.target2 else s[4]
                next_states.append(s[:2] + (x, y, target2_reached) + (s[5],) + s[6:])
                next_states.append(s[:2] + (x, y, target2_reached) + ((s[5] + 1) % 2,) + s[6:])
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
        self.visited_targets = (0, 0, 0)  # Reset visited targets
        self.state = self.state1 + self.state2 + (self.turn,) + self.visited_targets
        self.initstate = self.state
        return self.state
