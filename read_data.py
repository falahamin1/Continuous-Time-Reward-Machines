import pickle
import matplotlib.pyplot as plt
import numpy as np


def save_plot( all_classic, all_counter, all_counter_sampling):
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
        # plt.plot(np.arange(classic_50.shape[0]) * 50, classic_50, color='orange', label='Classic (Median)')
        # plt.fill_between(np.arange(classic_50.shape[0]) * 50, classic_25, classic_75, color='orange', alpha=0.3)
    
    # Plot for counterfactual data
        plt.plot(np.arange(counter_50.shape[0]) * 50, counter_50, color='blue', label='Counterfactual (Median)')
        plt.fill_between(np.arange(counter_50.shape[0]) * 50, counter_25, counter_75, color='blue', alpha=0.3)
    
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
            plt.plot(np.arange(sampling_50.shape[0]) * 50, sampling_50, color=colors[j], label=f'Counterfactual with Sampling (size = {i})')
            # plt.fill_between(np.arange(sampling_50.shape[0]) * 50, sampling_25, sampling_75, color=colors[j], alpha=0.3)
            j+=1

    # Add labels and title
        plt.xlabel('Time Steps')
        plt.ylabel('Performance')
        plt.title(f'Comparison of Classic, Counterfactual and Counterfactual with Sampling on treasure-hunt (tabular)')
    
    # Add a legend
        plt.legend()

        plt.show()
        
    # Save the plot for classic, counterfactual, and all sampling sizes
        # plt.savefig(savefile)  
        plt.close()

# Path to the file you saved earlier
datafile = 'treasurehunt-server-tabular-data'

# Open the file in binary read mode and load the data
with open(datafile, 'rb') as f:
    loaded_data = pickle.load(f)

# Access the loaded data
all_classic = loaded_data['all_classic']
all_counter = loaded_data['all_counter']
all_counter_sampling = loaded_data['all_counter_sampling']

# Now you can use the data as needed
print("Classic Data:", all_classic)
print("Counter Data:", all_counter)
print("Counter Sampling Data:", all_counter_sampling)
save_plot(all_classic=all_classic, all_counter= all_counter, all_counter_sampling= all_counter_sampling)
