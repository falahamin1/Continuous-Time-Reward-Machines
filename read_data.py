import pickle
import matplotlib.pyplot as plt
import numpy as np


# def save_plot( all_classic, all_counter, all_counter_sampling):
#     # Convert lists to numpy arrays for easier percentile calculations
#         all_classic = np.array(all_classic)
#         all_counter = np.array(all_counter)
    
#     # Calculate the 25th, 50th (median), and 75th percentiles for both datasets
#         classic_25 = np.percentile(all_classic, 25, axis=0)
#         classic_50 = np.median(all_classic, axis=0)
#         classic_75 = np.percentile(all_classic, 75, axis=0)
    
#         counter_25 = np.percentile(all_counter, 25, axis=0)
#         counter_50 = np.median(all_counter, axis=0)
#         counter_75 = np.percentile(all_counter, 75, axis=0)
    
#     # Create a plot
#         plt.figure(figsize=(10, 6))
    
#     # Plot for classic data
#         plt.plot(np.arange(classic_50.shape[0]) * 50, classic_50, color='orange', label='Classic (Median)')
#         plt.fill_between(np.arange(classic_50.shape[0]) * 50, classic_25, classic_75, color='orange', alpha=0.3)
    
#     # Plot for counterfactual data
#         plt.plot(np.arange(counter_50.shape[0]) * 50, counter_50, color='blue', label='Counterfactual (Median)')
#         plt.fill_between(np.arange(counter_50.shape[0]) * 50, counter_25, counter_75, color='blue', alpha=0.3)
    
#     # Now, add plots for each sampling size on the same figure
#         # sampling_size = [10, 20, 30, 40]  # Assuming these are the sampling sizes
#         colors = ['green', 'purple', 'red', 'brown']  # Different colors for different sampling sizes
#         j = 0
#         for i, sampling_data in all_counter_sampling.items():
#             # sample_size = sampling_size[i]  # Get the current sampling size
#         # Convert current sampling data to numpy array and calculate percentiles
#             sampling_data = np.array(sampling_data)
#             # print("sampling data:", sampling_data)
#             # print("i:" ,i)
#             sampling_25 = np.percentile(sampling_data, 25, axis=0)
#             sampling_50 = np.median(sampling_data, axis=0)
#             sampling_75 = np.percentile(sampling_data, 75, axis=0)

#         # Plot for counterfactual sampling data with the sampling size in the legend
#             plt.plot(np.arange(sampling_50.shape[0]) * 50, sampling_50, color=colors[j], label=f'Counterfactual with Sampling (size = {i})')
#             plt.fill_between(np.arange(sampling_50.shape[0]) * 50, sampling_25, sampling_75, color=colors[j], alpha=0.3)
#             j+=1

#     # Add labels and title
#         plt.xlabel('Time Steps')
#         plt.ylabel('Performance')
#         plt.title(f'Comparison of Classic, Counterfactual and Counterfactual with Sampling on treasurehunt (tabular)')
    
#     # Add a legend
#         plt.legend()

#         plt.show()
        
#     # Save the plot for classic, counterfactual, and all sampling sizes
#         # plt.savefig(savefile)  
#         plt.close()

def save_plot(all_classic, all_counter, all_counter_sampling, all_classic1, all_counter1, all_counter_sampling1):
    # Prepend 0 to all the lists in all_classic
    all_classic = [([0] + data) for data in all_classic]
    all_counter = [([0] + data) for data in all_counter]

    all_classic1 = [([0] + data) for data in all_classic1]
    all_counter1 = [([0] + data) for data in all_counter1]
    
    # Prepend 0 to all the lists in all_counter_sampling
    all_counter_sampling = {k: [([0] + sublist) for sublist in v] for k, v in all_counter_sampling.items()}
    all_counter_sampling1 = {k: [([0] + sublist) for sublist in v] for k, v in all_counter_sampling1.items()}

    # Convert lists to numpy arrays for easier percentile calculations
    all_classic = np.array(all_classic)
    all_counter = np.array(all_counter)
    all_classic1 = np.array(all_classic1)
    all_counter1 = np.array(all_counter1)
    
    # Calculate the 25th, 50th (median), and 75th percentiles for both datasets
    classic_25 = np.percentile(all_classic, 25, axis=0)
    classic_50 = np.median(all_classic, axis=0)
    classic_75 = np.percentile(all_classic, 75, axis=0)

    classic_251 = np.percentile(all_classic1, 25, axis=0)
    classic_501 = np.median(all_classic1, axis=0)
    classic_751 = np.percentile(all_classic1, 75, axis=0)
    
    counter_25 = np.percentile(all_counter, 25, axis=0)
    counter_50 = np.median(all_counter, axis=0)
    counter_75 = np.percentile(all_counter, 75, axis=0)

    counter_251 = np.percentile(all_counter1, 25, axis=0)
    counter_501 = np.median(all_counter1, axis=0)
    counter_751 = np.percentile(all_counter1, 75, axis=0)
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot for classic data
    plt.plot(np.arange(classic_50.shape[0]) * 10, classic_50, color='#90EE90', label='Baseline')
    plt.fill_between(np.arange(classic_50.shape[0]) * 10, classic_25, classic_75, color='#90EE90', alpha=0.3)

    plt.plot(np.arange(classic_501.shape[0]) * 10, classic_501, color='#006400', label='Baseline(RS)')
    plt.fill_between(np.arange(classic_501.shape[0]) * 10, classic_251, classic_751, color='#006400', alpha=0.3)
    
    # Plot for counterfactual data
    plt.plot(np.arange(counter_50.shape[0]) * 10, counter_50, color='#ADD8E6', label='Counterfactual')
    plt.fill_between(np.arange(counter_50.shape[0]) * 10, counter_25, counter_75, color='#ADD8E6', alpha=0.3)

    plt.plot(np.arange(counter_501.shape[0]) * 10, counter_501, color='#00008B', label='Counterfactual(RS)')
    plt.fill_between(np.arange(counter_501.shape[0]) * 10, counter_251, counter_751, color='#00008B', alpha=0.3)
    
    # Now, add plots for each sampling size on the same figure
    colors = ['green', '#FF6666', 'red', 'brown']  # Different colors for different sampling sizes
    j = 0
    for i, sampling_data in all_counter_sampling.items():
        # Convert current sampling data to numpy array and calculate percentiles
        sampling_data = np.array(sampling_data)
        sampling_25 = np.percentile(sampling_data, 25, axis=0)
        sampling_50 = np.median(sampling_data, axis=0)
        sampling_75 = np.percentile(sampling_data, 75, axis=0)

        # Plot for counterfactual sampling data with the sampling size in the legend
        if j ==1:
            plt.plot(np.arange(sampling_50.shape[0]) * 10, sampling_50, color=colors[j], label=f'Counterfactual with Sampling ')
            plt.fill_between(np.arange(sampling_50.shape[0]) * 10, sampling_25, sampling_75, color=colors[j], alpha=0.3)
        
        j += 1

    colors = ['green', '#8B0000', 'red', 'brown']  # Different colors for different sampling sizes
    j = 0
    for i, sampling_data in all_counter_sampling1.items():
        # Convert current sampling data to numpy array and calculate percentiles
        sampling_data = np.array(sampling_data)
        sampling_25 = np.percentile(sampling_data, 25, axis=0)
        sampling_50 = np.median(sampling_data, axis=0)
        sampling_75 = np.percentile(sampling_data, 75, axis=0)

        # Plot for counterfactual sampling data with the sampling size in the legend
        if j ==1:
            plt.plot(np.arange(sampling_50.shape[0]) * 10, sampling_50, color=colors[j], label=f'Counterfactual with Sampling (RS)')
            plt.fill_between(np.arange(sampling_50.shape[0]) * 10, sampling_25, sampling_75, color=colors[j], alpha=0.3)
            
        
        j += 1

    # Add labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('Performance')
    plt.title('Comparison of Classic, Counterfactual, and Counterfactual with Sampling on treasure map example')
    
    # Add a legend
    # plt.legend()

    plt.show()
    
    # Close the plot to free memory
    plt.close()

import matplotlib.pyplot as plt
import os

def save_legend():
    try:
        # Define dummy handles and labels for the legend
        handles = [
            plt.Line2D([0], [0], color='#90EE90', lw=2, label='Baseline'),
            plt.Line2D([0], [0], color='#006400', lw=2, label='Baseline(RS)'),
            plt.Line2D([0], [0], color='#ADD8E6', lw=2, label='Counterfactual'),
            plt.Line2D([0], [0], color='#00008B', lw=2, label='Counterfactual(RS)'),
            plt.Line2D([0], [0], color='#FF6666', lw=2, label='Counterfactual with Sampling'),
            plt.Line2D([0], [0], color='#8B0000', lw=2, label='Counterfactual with Sampling (RS)')
        ]

        # Create a blank figure for the legend
        fig_leg = plt.figure(figsize=(10, 1))  # Adjust width to fit horizontally
        legend = fig_leg.legend(handles, [h.get_label() for h in handles], 
                                loc='center', ncol=len(handles), frameon=False)

        # Remove axes
        plt.axis('off')

        # Define save path
        save_path = os.path.join(os.getcwd(), "legend.png")
        
        # Save the legend figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig_leg)

        print(f"Legend saved successfully at: {save_path}")
    except Exception as e:
        print(f"An error occurred: {e}")







# Path to the file you saved earlier
datafile = 'treasurehunt-server-tabular-data'
datafile1 = 'treasurehunt-server-tabular-data-rs'

# Open the file in binary read mode and load the data
with open(datafile, 'rb') as f:
    loaded_data = pickle.load(f)

with open(datafile1, 'rb') as f:
    loaded_data1 = pickle.load(f)

# Access the loaded data
all_classic = loaded_data['all_classic']
all_counter = loaded_data['all_counter']
all_counter_sampling = loaded_data['all_counter_sampling']

all_classic1 = loaded_data1['all_classic']
all_counter1 = loaded_data1['all_counter']
all_counter_sampling1 = loaded_data1['all_counter_sampling']

# Now you can use the data as needed
print("Classic Data:", all_classic)
print("Counter Data:", all_counter)
print("Counter Sampling Data:", all_counter_sampling)
save_plot(all_classic=all_classic, all_counter= all_counter, all_counter_sampling= all_counter_sampling, all_classic1= all_classic1, all_counter1= all_counter1, all_counter_sampling1= all_counter_sampling1)
# save_legend()
