from neural_network import DQN
from replay_buffer import ReplayBuffer
from optimize_model import Optimizer
# from copcarenv import CopCarEnv
# from CTRMcopcarUT import RewardMachineCopCarUT
from agent import Agent
import torch.optim as optim
import math
from collections import deque
import numpy as np

class DeepRLCounterFactualSampling():
    def __init__(self, capacity = 5000, epsilon = 1, Gamma= 0.0001, batchsize = 64, learnrate = 0.001, 
    max_episode_length = 1000, update_length = 64, number_of_episodes = 5000, UPDATE_FREQUENCY = 50, hard_update_frequency = 5,env = None, ctrm = None, decay_rate = 0.05,sampling = 10):
    #Initializing the environemnt and CTRM
        #Initializing the environemnt and CTRM
        self.env = env
        self.ctrm = ctrm

        #for exponential decay
        self.max_epsilon = epsilon
        self.min_epsilon = 0.1
        self.decay_rate = decay_rate

        self.epsilon = self.max_epsilon

        #Initializing the target and policy networks
        self.policy_net = DQN(self.env.statesize + 1,self.env.actionsize) # 1 extra input for the ctrm state
        self.target_net = DQN(self.env.statesize + 1,self.env.actionsize) 
        # Initializing the optimizer, replay buffer and the agent
        self.capacity = capacity # capacity of the buffer (hyperparameter)
        self.adamoptimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # Initialized the optimizer
        self.Gamma = Gamma # This is the continuous time discount parameter (hyperparameter)
        self.batchsize = batchsize # Batch size for training (hyperparameter)
        self.learnrate = learnrate # learning rate for updating the target network (hyperparameter)


        self.agent = Agent(self.epsilon, self.env.actions, self.policy_net) # Initialized the agent
        self.buffer = ReplayBuffer(capacity=self.capacity) # Initialized the buffer
        self.optimizer = Optimizer(policy_net=self.policy_net, target_net= self.target_net, optimizer= self.adamoptimizer,device= 'cpu', memory= self.buffer, GAMMA= self.Gamma, BATCH_SIZE= self.batchsize )  
        #Initialized the optimized class

        self.max_episode_length = max_episode_length #max episode length before termination
        self.update_length = update_length  # Number of steps before the policy network and target network(soft update) is updated
        self.number_of_episodes = number_of_episodes # Number of episodes 
        self.evaluation_results = [0]
        self.UPDATE_FREQUENCY = UPDATE_FREQUENCY
        self.hard_update_frequency = hard_update_frequency # hard updates for target network 
         # Information for checking the value of the strategy
        self.V = {}
        self.states = self.fill_states()
        self.ctrm_states = tuple(self.ctrm.states)
        self.fill_vtable()
        self.sampling = sampling




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

    def fill_vtable(self): 
        for state in self.states:
                for ctrm_state in self.ctrm.states:  
                    s = state + (ctrm_state,)
                    self.V[s]= 0 
    def startegy_analaysis(self):
        enable = True
        initstate = self.env.initstate + (self.ctrm.initstate,) # Initial state of the product
        while enable:
            delta = 0
            for state in self.V:   # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                a = self.agent.get_best_action(state)
                ctrm_state = state[-1]     #Get the CTRM state
                env_state = state[:-1]      # Get the environment state
                rate = self.ctrm.get_rate_counterfactual(ctrm_state,env_state) # Gets the rate
                next_states = self.env.next_state(env_state, a) # Gets the next state of the environment
                for next_state, probability in next_states.items():
                        action_value = 0
                        reward, ctrm_next = self.ctrm.transition_function_counterfactual(ctrm_state, next_state)
                        next_state1 = next_state + (ctrm_next,)
                        if reward is not None and rate is not None:
                                time = 1/rate
                                action_value += (probability * (reward + math.exp(-1 * time * self.Gamma) * self.V[next_state1])) 
                self.V[state] = action_value
                delta = max(delta, abs(v - self.V[state]))
            if delta < 0.01: 
                enable = False
        return self.V[initstate]

    # def explore(self, max_depth=100):
    #     depth = 0
    #     total_discount = 1.0
    #     accumulated_reward = 0

    #     while depth < max_depth:
    #         env_state = self.env.state
    #         # env_x, env_y, env_siren, env_thief = self.env.state
    #         ctrm_state = self.ctrm.state
    #         rate = self.ctrm.get_rate(env_state)
    #         action = self.agent.get_best_action(env_state + (ctrm_state,))
    #         env_state, sampled_time = self.env.step(action=action, rate=rate)
            
    #         reward = self.ctrm.transitionfunction(self.env.state)

    #         if reward > 0:
    #             accumulated_reward += reward * total_discount
    #             break

    #         accumulated_reward += reward * total_discount  # Add current reward
    #         total_discount *= math.exp(-1 * sampled_time * self.Gamma)  # Update discount factor

    #         depth += 1

    #     return accumulated_reward

    # def get_agent_value(self):
        
    #     sum_reward = 0
    #     for i in range(100): # takes average of 100 runs
    #         self.env.reset()
    #         self.ctrm.reset()
    #         sum_reward += self.explore (max_depth=100)

    #     return sum_reward/100

    def doRL(self):
        
        k = 0
        sum_perfomance = 0
        for i in range (self.number_of_episodes): 
            self.env.reset()
            self.ctrm.reset()
            # env_x, env_y, env_siren, env_theif = self.env.state
            env_state = self.env.state
            # print("Initial state:", env_state)
            ctrm_state = self.ctrm.state
            rate = self.ctrm.get_rate(env_state)

            for j in range(self.max_episode_length): 
                if rate is None:
                    break
                action = self.agent.epsilon_greedy_policy(env_state + (ctrm_state,),self.epsilon) #epsilon greedy action
                env_state1, sampled_time = self.env.step(action=action, rate= rate)
                reward = self.ctrm.transitionfunction(env_state1) #transition in the ctrm which gives the new state and the reward
                if reward is None: 
                    break
                ctrm_state1 = self.ctrm.state # new ctrm state
                previous_state = env_state + (ctrm_state,)
                next_state = env_state1 + (ctrm_state1,)
                #  add(self, state, action, reward,sampletime, next_state):
                # self.buffer.add(previous_state, action, reward, sampled_time, next_state)
                self.add_counterfactual_experience(env_state, action, env_state1)
                k += 1
                if k > self.update_length: 
                    self.optimizer.optimize_model() #optimization step
                    self.optimizer.update_target_soft() #soft update for target
                    k = 0
                # if reward > 0: 
                #     break
                env_state = self.env.state
                ctrm_state = self.ctrm.state
                rate = self.ctrm.get_rate(env_state)
            if (i + 1) % self.UPDATE_FREQUENCY == 0:
                sum_perfomance = self.get_average(sum_perfomance, (i+1)/self.UPDATE_FREQUENCY)
            # self.epsilon = max(self.epsilon - self.epsilon_decay * i, 0.01) # linear decay
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate *i)
        return self.evaluation_results

    def doRLwithconvergence(self , value, threshold, max_episodes = 1000000): # This is the same RL algorithm but the termination condition occurs when a threshold is reached wrt the given value
        
        k = 0
        sum_perfomance = 0
        termination = 0
        i = 0
        while termination == 0 and i < max_episodes: 
            self.env.reset()
            self.ctrm.reset()
            # env_x, env_y, env_siren, env_theif = self.env.state
            env_state = self.env.state
            # print("Initial state:", env_state)
            ctrm_state = self.ctrm.state
            rate = self.ctrm.get_rate(env_state)
            # if i % 1000 == 0:
            #     print(f"Episode {i} completed.")

            for j in range(self.max_episode_length): 
                if rate is None:
                    break
                action = self.agent.epsilon_greedy_policy(env_state + (ctrm_state,),self.epsilon) #epsilon greedy action
                env_state1, sampled_time = self.env.step(action=action, rate= rate)
                reward = self.ctrm.transitionfunction(env_state1) #transition in the ctrm which gives the reward
                if reward is None: 
                    break
                # ctrm_state1 = self.ctrm.state # new ctrm state
                # previous_state = env_state + (ctrm_state,)
                # next_state = env_state1 + (ctrm_state1,)
                # sampled_time = 1/rate
                #  add(self, state, action, reward,sampletime, next_state):
                # self.buffer.add(previous_state, action, reward, sampled_time, next_state)
                self.add_counterfactual_experience(env_state, action, env_state1)
                k += 1
                if k > self.update_length: 
                    self.optimizer.optimize_model() #optimization step
                    self.optimizer.update_target_soft() #soft update for target
                    k = 0
                env_state = self.env.state
                ctrm_state = self.ctrm.state
                rate = self.ctrm.get_rate(env_state)
            if (i + 1) % self.UPDATE_FREQUENCY == 0:
                sum_perfomance = self.get_average(sum_perfomance, (i+1)/self.UPDATE_FREQUENCY)
                # print(f"episode: {i}, values given {self.evaluation_results[-1] / value}")
                if self.evaluation_results[-1]/value > threshold:
                    termination = 1
                    break 
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate *i)
            i += 1
        return self.evaluation_results
    
    
    def add_counterfactual_experience(self, state, action, next_state):
        for _ in range(self.sampling): 
            for ctrmstate in self.ctrm.states: # For each state of the CTRM, we add an experience
                rate = self.ctrm.get_rate_counterfactual(ctrmstate, state)

                if rate is not None:
                    reward, ctrm_nextstate = self.ctrm.transition_function_counterfactual(ctrmstate, next_state)
                    timetaken = np.random.exponential(scale= 1/rate)
                    if reward  is not None:
                    
                        previous_state = state + (ctrmstate,)
                        next_state1 = next_state + (ctrm_nextstate,)
                        self.buffer.add(previous_state, action, reward, time, next_state1)

        
    
    def get_average(self, sum_perfomance, step):
        agent_value = self.startegy_analaysis()
        self.fill_vtable() # Resets the v table
        sum_perfomance += agent_value
        value = sum_perfomance/step
        # print(f"Episode {step}: Agent Value = {value}")
        self.evaluation_results.append(value)
        return sum_perfomance

    def print_strategy(self):
        self.fill_vtable() # Resets the v table
        for state in self.V:   # Each state is encoded as (environment state, ctrm state)
                v = self.V[state]  # Previous value
                a = self.agent.get_best_action(state)
                print(f"In state {state}, the action taken is {a}")
            
            






