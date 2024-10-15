import numpy as np
import matplotlib.pyplot as plt
from Classic_DeepRL import DeepRLClassic
from cop_car_ctrm import CopCarCTRM
from cop_car import CopCarEnv
from firefightercar_ctrm import FireFighterCarCTRM
from firefighter_car import FireFighterCarEnv
from treasure_hunt_ctrm import TreasureMapCTRM
from treasure_hunt import TreasureMapEnv
from firefighter_sync import FireFighterCarSynchEnv
from firefighter_sync_ctrm import FireFighterCarSynchCTRM
from counterfactualDeepRL import DeepRLCounterFactual
from value_iteration import ValueIteration
from CounterFactualDRLSampling import DeepRLCounterFactualSampling
from TabularCounterSampling import DynamicQLearningCounterFactualSampling
from TabularLearning import DynamicQLearning
from TabularLearningCounterFactual import DynamicQLearningCounterFactual
import time 
import argparse
from TabularLearning import DynamicQLearning
import pickle


class Comparison:
    def __init__(self, env, specify_dimension, deep_rl, rows, columns, discount_factor, learning_rate, runs, threshold, max_episodes, episode_length, decay_rate, buffer_size, batch_size, update_frequency, save_file,save_data,method):
        self.env = env
        self.specify_dimension = specify_dimension
        self.deep_rl = deep_rl
        self.rows = rows
        self.columns = columns
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.runs = runs
        self.threshold = threshold
        self.max_episodes = max_episodes
        self.episode_length = episode_length
        self.decay_rate = decay_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.save_file = save_file
        self.save_data = save_data
        self.ctrm, self.env_class = self.get_ctrm_env()
        self.value = None
        self.method = method

#From the input name of environment, this function returns the CTRM and the environment classes 
    def get_ctrm_env(self):
        if self.env == "firefighter-car":
            ctrm = FireFighterCarCTRM()
            env = FireFighterCarEnv(self.rows, self.columns, probability= 0.95)
        elif self.env == "treasure-map":
            ctrm = TreasureMapCTRM()
            env = TreasureMapEnv(probability= 1)
        elif self.env == "cop-car":
            ctrm = CopCarCTRM()
            env = CopCarEnv(self.rows, self.columns, probability= 0.9)
        elif self.env == "firefighter-synch":
            ctrm = FireFighterCarSynchCTRM()
            env = FireFighterCarSynchEnv(self.rows,self.columns, probability= 0.95)
        return ctrm, env
        

    def display_parameters(self):
        print(f"Environment: {self.env}",flush=True)
        print(f"Specify Dimension: {self.specify_dimension}",flush=True)
        if self.specify_dimension == "yes":
            print(f"Rows: {self.rows}, Columns: {self.columns}",flush=True)
        print(f"Discount Factor: {self.discount_factor}",flush=True)
        print(f"Learning Rate: {self.learning_rate}",flush=True)
        print(f"Runs: {self.runs}",flush=True)
        print(f"Threshold: {self.threshold}",flush=True)
        print(f"Max Episodes: {self.max_episodes}",flush=True)
        print(f"Episode Length: {self.episode_length}",flush=True)
        print(f"Decay Rate: {self.decay_rate}",flush=True)
        print(f"Buffer Size: {self.buffer_size}",flush=True)
        print(f"Batch Size: {self.batch_size}",flush=True)
        print(f"Update Frequency: {self.update_frequency}",flush=True)
        print(f"Deep Rl: {self.deep_rl}",flush=True)
    
    def run_classic(self):
        DRL =  DeepRLClassic(capacity = self.buffer_size, epsilon = 1, Gamma= self.discount_factor, batchsize = self.batch_size, learnrate = self.learning_rate, 
    max_episode_length = self.episode_length , number_of_episodes = 5000, UPDATE_FREQUENCY = self.update_frequency, env = self.env_class, ctrm = self.ctrm, decay_rate = self.decay_rate)
        results = DRL.doRLwithconvergence(value= self.value, threshold= self.threshold, max_episodes= self.max_episodes)
        return results

    def run_classic_tabular(self):
        DRL =  DynamicQLearning(alpha=self.learning_rate, gamma= self.discount_factor, epsilon= 1, UPDATE_FREQUENCY= self.update_frequency,
        environment= self.env_class, ctrm= self.ctrm, decay_rate= self.decay_rate)
        results = DRL.trainwithconvergence(max_episode_length = self.episode_length, value = self.value, threshold = self.threshold, max_episodes = self.max_episodes)
        return results


    def run_counterfactual(self):
        CFDRL = DeepRLCounterFactual(capacity = self.buffer_size, epsilon = 1, Gamma= self.discount_factor, batchsize = self.batch_size, learnrate = self.learning_rate, 
    max_episode_length = self.episode_length, number_of_episodes = 5000, UPDATE_FREQUENCY = self.update_frequency, env = self.env_class, ctrm = self.ctrm, decay_rate = self.decay_rate)
        results = CFDRL.doRLwithconvergence(value= self.value, threshold= self.threshold, max_episodes= self.max_episodes)
        return results
# def trainwithconvergence(self, num_episodes, max_episode_length, value, threshold, max_episodes = 100000):
    def run_counterfactual_tabular(self):
        DRL =  DynamicQLearningCounterFactual(alpha=self.learning_rate, gamma= self.discount_factor, epsilon= 1, UPDATE_FREQUENCY= self.update_frequency,
        environment= self.env_class, ctrm= self.ctrm, decay_rate= self.decay_rate)
        results = DRL.trainwithconvergence(max_episode_length = self.episode_length, value = self.value, threshold = self.threshold, max_episodes = self.max_episodes)
        return results

    def run_counterfactual_sampling(self, sampling):
        CFDRL = DeepRLCounterFactualSampling(capacity = self.buffer_size, epsilon = 1, Gamma= self.discount_factor, batchsize = self.batch_size, learnrate = self.learning_rate, 
    max_episode_length = self.episode_length, number_of_episodes = 5000, UPDATE_FREQUENCY = self.update_frequency, env = self.env_class, ctrm = self.ctrm, decay_rate = self.decay_rate, sampling= sampling)
        results = CFDRL.doRLwithconvergence(value= self.value, threshold= self.threshold, max_episodes = self.max_episodes)
        return results

    def run_counterfactual_sampling_tabular(self, sampling):
        DRL =  DynamicQLearningCounterFactualSampling(alpha=self.learning_rate, gamma= self.discount_factor, epsilon= 1, UPDATE_FREQUENCY= self.update_frequency,
        environment= self.env_class, ctrm= self.ctrm, decay_rate= self.decay_rate,sampling= sampling)
        results = DRL.trainwithconvergence(max_episode_length = self.episode_length, value = self.value, threshold = self.threshold, max_episodes = self.max_episodes)
        return results

    def run_comparison_tabular(self):
        value = self.value
        cf_sampling_data = self.counterfactualsampling_tabular()
        all_classic = []
        all_counter = []
        length_classic = 0
        length_counter = 0
        for i in range(self.runs):
                    classic_data = self.run_classic_tabular()
                    counterfactual_data = self.run_counterfactual_tabular()
                    all_classic.append(classic_data)
                    all_counter.append(counterfactual_data)
                    length_classic = max(length_classic, len(classic_data))
                    length_counter = max(length_counter, len(counterfactual_data))
        print(f"Length counter is {length_counter}", flush=True)
        print(f"Length classic is {length_classic}",flush=True)
                
        for sub_array in all_classic:
                    if len(sub_array) < length_classic:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_classic - len(sub_array)) if sub_array else [0] * length_classic)
        for sub_array in all_counter:
                    if len(sub_array) < length_counter:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_counter - len(sub_array)) if sub_array else [0] * length_counter)
        self.save_plot(all_classic, all_counter,cf_sampling_data, self.save_file, self.rows)


           
    def run_comparison(self):
        print("In run comparison.",flush=True)
        vi = ValueIteration(gamma = self.discount_factor, environment = self.env_class, ctrm = self.ctrm )
        Value = vi.doVI()
        self.value = Value
        print(f"Value is {self.value}",flush=True)
        if self.deep_rl == "no":
            self.run_comparison_tabular()
        else:
            if self.specify_dimension == "yes":
                cf_sampling_data = self.counterfactualsampling()
                all_classic = []
                all_counter = []
                length_classic = 0
                length_counter = 0
                for i in range(self.runs):
                    classic_data = self.run_classic()
                    counterfactual_data = self.run_counterfactual()
                    all_classic.append(classic_data)
                    all_counter.append(counterfactual_data)
                    length_classic = max(length_classic, len(classic_data))
                    length_counter = max(length_counter, len(counterfactual_data))
                print(f"Length counter is {length_counter}", flush=True)
                print(f"Length classic is {length_classic}",flush=True)
                
                for sub_array in all_classic:
                    if len(sub_array) < length_classic:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_classic - len(sub_array)) if sub_array else [0] * length_classic)
                for sub_array in all_counter:
                    if len(sub_array) < length_counter:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_counter - len(sub_array)) if sub_array else [0] * length_counter)
            self.save_plot(all_classic, all_counter,cf_sampling_data, self.save_file, self.rows)
        
        #     else: 
        #     i = 3
        #     self.rows = i
        #     self.columns = i
        #     self.ctrm, self.env_class = self.get_ctrm_env()
        #     while i <= 7: 
        #         all_classic = []
        #         all_counter = []
        #         length_classic = 0
        #         length_counter = 0
        #         for i in range(self.runs):
        #             classic_data = self.run_classic()
        #             counterfactual_data = self.run_counterfactual()
        #             all_classic.append(classic_data)
        #             all_counter.append(counterfactual_data)
        #             length_classic = max(length_classic, len(classic_data))
        #             length_counter = max(length_counter, len(counterfactual_data))
                
        #         for sub_array in all_classic:
        #             if len(sub_array) < length_classic:
        # # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
        #                 sub_array.extend([sub_array[-1]] * (length_classic - len(sub_array)) if sub_array else [0] * length_classic)
        #         for sub_array in all_counter:
        #             if len(sub_array) < length_counter:
        # # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
        #                 sub_array.extend([sub_array[-1]] * (length_counter - len(sub_array)) if sub_array else [0] * length_classic)
                
        #         save_file = self.save_file + str(i)
        #         self.save_plot(all_classic, all_counter, save_file, i)
        #         i += 2




    def counterfactualsampling_tabular(self):
        sampling_size = [10,20,30]
        all_data_by_sampling_size = {}
        for sample in sampling_size:
            all_data = []
            length = 0 
            for i in range(self.runs):
                data = self.run_counterfactual_sampling_tabular(sample)
                all_data.append(data)
                length = max (length, len(data))
            for sub_array in all_data:
                    if len(sub_array) < length:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length - len(sub_array)) if sub_array else [0] * length)
            all_data_by_sampling_size[sample] = all_data
        return all_data_by_sampling_size


    def counterfactualsampling(self):
        sampling_size = [10,20,30]
        all_data_by_sampling_size = {}
        for sample in sampling_size:
            all_data = []
            length = 0 
            for i in range(self.runs):
                data = self.run_counterfactual_sampling(sample)
                all_data.append(data)
                length = max (length, len(data))
            for sub_array in all_data:
                    if len(sub_array) < length:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length - len(sub_array)) if sub_array else [0] * length)
            all_data_by_sampling_size[sample] = all_data
        return all_data_by_sampling_size
                            

    def create_plot(self):
        plt.figure(figsize=(10, 6))


    def save_plot(self, all_classic, all_counter, all_counter_sampling, savefile, rows):
    # Convert lists to numpy arrays for easier percentile calculations
        all_classic = np.array(all_classic)
        all_counter = np.array(all_counter)
    
    # Calculate the 25th, 50th (median), and 75th percentiles for both datasets
        classic_25 = np.percentile(all_classic, 25, axis=0)
        classic_50 = np.median(all_classic, axis=0)
        classic_75 = np.percentile(all_classic, 75, axis=0)
    
        counter_25 = np.percentile(all_counter, 25, axis=0)
        counter_50 = np.median(all_counter, axis=0)
        counter_75 = np.percentile(all_counter, 75, axis=0)
    
    # Create a plot
        plt.figure(figsize=(10, 6))
    
    # Plot for classic data
        plt.plot(np.arange(classic_50.shape[0]) * self.update_frequency, classic_50, color='orange', label='Classic (Median)')
        # plt.fill_between(np.arange(classic_50.shape[0]) * self.update_frequency, classic_25, classic_75, color='orange', alpha=0.3)
    
    # Plot for counterfactual data
        plt.plot(np.arange(counter_50.shape[0]) * self.update_frequency, counter_50, color='blue', label='Counterfactual (Median)')
        # plt.fill_between(np.arange(counter_50.shape[0]) * self.update_frequency, counter_25, counter_75, color='blue', alpha=0.3)
    
    # Now, add plots for each sampling size on the same figure
        # sampling_size = [10, 20, 30, 40]  # Assuming these are the sampling sizes
        colors = ['green', 'purple', 'red', 'brown']  # Different colors for different sampling sizes
        j = 0
        for i, sampling_data in all_counter_sampling.items():
            # sample_size = sampling_size[i]  # Get the current sampling size
        # Convert current sampling data to numpy array and calculate percentiles
            sampling_data = np.array(sampling_data)
            # print("sampling data:", sampling_data)
            # print("i:" ,i)
            sampling_25 = np.percentile(sampling_data, 25, axis=0)
            sampling_50 = np.median(sampling_data, axis=0)
            sampling_75 = np.percentile(sampling_data, 75, axis=0)

        # Plot for counterfactual sampling data with the sampling size in the legend
            plt.plot(np.arange(sampling_50.shape[0]) * self.update_frequency, sampling_50, color=colors[j], label=f'Counterfactual with Sampling (size = {i})')
            # plt.fill_between(np.arange(sampling_50.shape[0]) * self.update_frequency, sampling_25, sampling_75, color=colors[j], alpha=0.3)
            j+=1

    # Add labels and title
        plt.xlabel('Time Steps')
        plt.ylabel('Performance')
        plt.title(f'Comparison of Classic, Counterfactual and Counterfactual with Sampling on {self.env}. Grid size {rows}')
    
    # Add a legend
        plt.legend()

    # Save the plot for classic, counterfactual, and all sampling sizes
        plt.savefig(savefile)  
        plt.close()

        data_to_save = {
        'all_classic': all_classic,
        'all_counter': all_counter,
        'all_counter_sampling': all_counter_sampling
    }

        with open(self.save_data, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Data saved to {self.save_data}",flush=True)
            
        
def main():
    parser = argparse.ArgumentParser(description="Comparison class input arguments")

    # Add arguments
    parser.add_argument("--env", type=str, default="default_env", help="Environment name (firefighter-car, cop-car, treasure-map, firefighter-synch)")
    parser.add_argument("--specify_dimension", type=str, choices=["yes", "no"], default="no", help="Specify dimensions (yes/no)")
    parser.add_argument("--deep_rl", type=str, choices=["yes", "no"], default="yes", help="Specify if Deep RL needs to be used")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows if dimensions are specified")
    parser.add_argument("--columns", type=int, default=None, help="Number of columns if dimensions are specified")
    parser.add_argument("--discount_factor", type=float, default=0.001, help="Discount factor for learning")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold value")
    parser.add_argument("--max_episodes", type=int, default=100000, help="Maximum number of episodes")
    parser.add_argument("--episode_len", type=int, default=1000, help="Length of each episode")
    parser.add_argument("--decay_rate", type=float, default=0.001, help="Decay rate")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Buffer size")
    parser.add_argument("--batch_size", type=int, default=1500, help="Batch size")
    parser.add_argument("--update_frequency", type=int, default=50, help="Update frequency")
    parser.add_argument("--save_file", type=str, default="default_plot", help="filename of the saved plot")
    parser.add_argument("--save_data", type=str, default="default_data", help="filename of the saved data")
    parser.add_argument("--method", type=str, default="no", help="Only a spcific method (no, counterfactual, classic, counterfactual_sampling)")


    args = parser.parse_args()

    # Check if dimensions are specified
    rows = args.rows if args.specify_dimension == "yes" else 5
    columns = args.columns if args.specify_dimension == "yes" else 5

    # Create Comparison object
    comparison = Comparison(
        env=args.env,
        specify_dimension=args.specify_dimension,
        deep_rl = args.deep_rl,
        rows=rows,
        columns=columns,
        discount_factor=args.discount_factor,
        learning_rate=args.learning_rate,
        runs=args.runs,
        threshold=args.threshold,
        max_episodes=args.max_episodes,
        episode_length = args.episode_len,
        decay_rate=args.decay_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_frequency=args.update_frequency,
        save_file = args.save_file,
        save_data = args.save_data,
        method = args.method
    )
    # Display parameters
    comparison.display_parameters()
    #Run comparison
    comparison.run_comparison()

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time}",flush=True)
