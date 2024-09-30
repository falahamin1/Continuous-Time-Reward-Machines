import pickle

# Path to the file you saved earlier
datafile = 'cop-car-tabular-data'

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