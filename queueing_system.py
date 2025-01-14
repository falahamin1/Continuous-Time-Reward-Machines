import numpy as np

class QSysEnv():
    # arrival rate: the rate at which jobs arrive 
    # q length: The max capacity of the que
    # n servers: The total number of servers 
    # service rate: rate of serving a job 
    # jobs: Total number of jobs to be completed
    def __init__(self, arrival_rate, q_length, n_servers, service_rate, jobs):
        self.initstate = (0,0,0)#(number of jobs in que, number of servers used, total jobs completed)
        self.state = self.initstate
        self.arrival_rate = arrival_rate 
        self.q_length = q_length
        self.n_servers = n_servers
        self.service_rate = service_rate
        self.jobs = jobs
        self.actions = range(1,self.n_servers+1) #Number of servers to be used 
        # self.actionmapping = {'north': 0, 'east': 1, 'west': 2, 'south': 3}
        self.time = 0
        self.statesize = 3
        self.actionsize = self.n_servers
        self.states = self.q_length #total number of states

        
    def next_states(self, s):#provides the set of next states for value iteration
        s1, k, q = s
        nstates = []
        if q >= self.jobs: 
            nstates.append((s1,k,q))
            return nstates
        elif s1 == self.states: 
            for k1 in range(1,self.n_servers + 1):
                nstates.append((s1-1,k1,q+1))
        else:
            for k1 in range(1,self.n_servers + 1):
                nstates.append((s1-1,k1,q+1))
                nstates.append((s1+1,k,q)) 
        return nstates
    

    def step(self, action, rate):
        arrival_rate, service_rate = rate
        s , k, q = self.state
        if q == self.jobs:
            self.state = self.state 
            time = 0 
            return self.state, time
        elif s == self.q_length: 
            self.state = (s-1,action,q+1)
        else:
            prob_s1 = arrival_rate / (arrival_rate + service_rate) # probability of a job arriving
            if np.random.rand() >= prob_s1:
                s += 1
            else: 
                s -= 1
                q += 1
        time = np.random.exponential(scale= 1/ (arrival_rate + service_rate))
        self.time += time
        self.state = (s,action,q)
        return self.state, time


    def next_state(self,state, action, rate):
        arrival_rate, service_rate = rate
        s,k,q = state
        next_state = {}
        next_state[(s+1,action,q)] = arrival_rate/(arrival_rate + service_rate)
        next_state[(s-1,action,q+1)] = 1 - next_state[(s+1,action,q)]
        return next_state

    def reset(self):
        self.state = self.initstate   # Reset to initial state
        self.time = 0
        return self.state