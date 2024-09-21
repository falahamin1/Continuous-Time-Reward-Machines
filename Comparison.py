import numpy as np
import matplotlib.pyplot as plt
from Classic_DeepRL import DeepRLClassic
from cop_car_ctrm import CopCarCTRM
from cop_car import CopCarEnv
from firefightercar_ctrm import FireFighterCarCTRM
from firefighter_car import FireFighterCarEnv
from treasure_hunt_ctrm import TreasureMapCTRM
from treasure_hunt import TreasureMapEnv
from counterfactualDeepRL import DeepRLCounterFactual
from value_iteration import ValueIteration


import argparse

class Comparison:
    def __init__(self, env, specify_dimension, rows, columns, discount_factor, learning_rate, runs, threshold, max_episodes, decay_rate, buffer_size, batch_size, update_frequency, save_file):
        self.env = env
        self.specify_dimension = specify_dimension
        self.rows = rows
        self.columns = columns
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.runs = runs
        self.threshold = threshold
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.save_file = save_file
        self.ctrm, self.env_class = self.get_ctrm_env()
        self.value = None

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
        return ctrm, env
        

    def display_parameters(self):
        print(f"Environment: {self.env}")
        print(f"Specify Dimension: {self.specify_dimension}")
        if self.specify_dimension == "yes":
            print(f"Rows: {self.rows}, Columns: {self.columns}")
        print(f"Discount Factor: {self.discount_factor}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Runs: {self.runs}")
        print(f"Threshold: {self.threshold}")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"Decay Rate: {self.decay_rate}")
        print(f"Buffer Size: {self.buffer_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Update Frequency: {self.update_frequency}")
    
    def run_classic(self):
        DRL =  DeepRLClassic(capacity = self.buffer_size, epsilon = 1, Gamma= self.discount_factor, batchsize = self.batch_size, learnrate = self.learning_rate, 
    max_episode_length = self.max_episodes, number_of_episodes = 5000, UPDATE_FREQUENCY = self.update_frequency, env = self.env_class, ctrm = self.ctrm, decay_rate = self.decay_rate)
        results = DRL.doRLwithconvergence(value= self.value, threshold= self.threshold, max_episodes= self.max_episodes)
        return results


            
    

    def run_counterfactual(self):
        CFDRL = DeepRLCounterFactual(capacity = self.buffer_size, epsilon = 1, Gamma= self.discount_factor, batchsize = self.batch_size, learnrate = self.learning_rate, 
    max_episode_length = self.max_episodes, number_of_episodes = 5000, UPDATE_FREQUENCY = self.update_frequency, env = self.env_class, ctrm = self.ctrm, decay_rate = self.decay_rate)
        results = CFDRL.doRLwithconvergence(value= self.value, threshold= self.threshold, max_episodes= self.max_episodes)
        return results


            
    def run_comparison(self):
        print("In run comparison.")
        vi = ValueIteration(gamma = self.discount_factor, environment = self.env_class, ctrm = self.ctrm )
        Value = vi.doVI()
        self.value = Value
        if self.specify_dimension == "yes":
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
                
            for sub_array in all_classic:
                if len(sub_array) < length_classic:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                    sub_array.extend([sub_array[-1]] * (length_classic - len(sub_array)) if sub_array else [0] * length)
            for sub_array in all_counter:
                if len(sub_array) < length_counter:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                    sub_array.extend([sub_array[-1]] * (length_counter - len(sub_array)) if sub_array else [0] * length)
            self.save_plot(all_classic, all_counter, self.save_file, self.rows)
        
        else: 
            i = 3
            self.rows = i
            self.columns = i
            self.ctrm, self.env_class = self.get_ctrm_env()
            while i <= 7: 
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
                
                for sub_array in all_classic:
                    if len(sub_array) < length_classic:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_classic - len(sub_array)) if sub_array else [0] * length)
                for sub_array in all_counter:
                    if len(sub_array) < length_counter:
        # Extend the sub-array to reach the desired length. Use the last element if available, or 0 if empty.
                        sub_array.extend([sub_array[-1]] * (length_counter - len(sub_array)) if sub_array else [0] * length)
                
                save_file = self.save_file + str(i)
                self.save_plot(all_classic, all_counter, save_file, i)
                i += 2




            

            
            
            
        
    def save_plot(self, all_classic, all_counter, savefile, rows):
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
        plt.plot(range(classic_50.shape[0]) * self.update_frequency,classic_50, color='orange', label='Classic (Median)')
        plt.fill_between(range(classic_50.shape[0]) * self.update_frequency, classic_25, classic_75, color='orange', alpha=0.3, label='Classic (25th-75th Percentile)')
    
    # Plot for counterfactual data
        plt.plot(range(counter_50.shape[0]) * self.update_frequency, counter_50, color='blue', label='Counterfactual (Median)')
        plt.fill_between(range(counter_50.shape[0]) * self.update_frequency, counter_25, counter_75, color='blue', alpha=0.3, label='Counterfactual (25th-75th Percentile)')
    
    # Add labels and title
        plt.xlabel('Time Steps')
        plt.ylabel('Performance')
        plt.title(f'Comparison of Classic and Counterfactual Data Over Runs. Grid size {rows}')
    
    # Add a legend
        plt.legend()
    
    # Save the plot
        plt.savefig(self.save_file)
    
    # Show the plot
        

            
        


def main():
    parser = argparse.ArgumentParser(description="Comparison class input arguments")

    # Add arguments
    parser.add_argument("--env", type=str, default="default_env", help="Environment name (firefighter-car, cop-car, treasure-map)")
    parser.add_argument("--specify_dimension", type=str, choices=["yes", "no"], default="no", help="Specify dimensions (yes/no)")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows if dimensions are specified")
    parser.add_argument("--columns", type=int, default=None, help="Number of columns if dimensions are specified")
    parser.add_argument("--discount_factor", type=float, default=0.0001, help="Discount factor for learning")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold value")
    parser.add_argument("--max_episodes", type=int, default=100000, help="Maximum number of episodes")
    parser.add_argument("--decay_rate", type=float, default=0.001, help="Decay rate")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Buffer size")
    parser.add_argument("--batch_size", type=int, default=1500, help="Batch size")
    parser.add_argument("--update_frequency", type=int, default=50, help="Update frequency")
    parser.add_argument("--save_file", type=str, default="default_plot", help="filename of the saved plot")

    args = parser.parse_args()

    # Check if dimensions are specified
    rows = args.rows if args.specify_dimension == "yes" else 5
    columns = args.columns if args.specify_dimension == "yes" else 5

    # Create Comparison object
    comparison = Comparison(
        env=args.env,
        specify_dimension=args.specify_dimension,
        rows=rows,
        columns=columns,
        discount_factor=args.discount_factor,
        learning_rate=args.learning_rate,
        runs=args.runs,
        threshold=args.threshold,
        max_episodes=args.max_episodes,
        decay_rate=args.decay_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_frequency=args.update_frequency,
        save_file = args.save_file,
    )
    # Display parameters
    comparison.display_parameters()
    #Run comparison
    comparison.run_comparison()

if __name__ == "__main__":
    main()
