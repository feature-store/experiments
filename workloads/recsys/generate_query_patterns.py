import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset, use_plots, log_plots

from tqdm import tqdm
import os

def average_interarrival_time(df): 

    # Sort the DataFrame based on the "timestamp" column
    df.sort_values(by='timestamp', inplace=True)

    # Calculate time differences between consecutive rows
    df['time_diff'] = df['timestamp'].diff()

    # Calculate the average inter-arrival time
    average_interarrival_time = df['time_diff'].mean()

    print("Average Inter-Arrival Time:", average_interarrival_time)
    return average_interarrival_time


def reset_timestamps(df, dist="exponential"):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")

    # compute mean 
    ev = average_interarrival_time(df)
    print("original", ev)

    # Sort the DataFrame based on the "timestamp" column
    df.sort_values(by='timestamp', inplace=True)

    # Generate exponential inter-arrival times
    if dist == "exponential":
        lambda_param = 1/ev.total_seconds()  # Adjust this parameter as needed
        times = np.random.exponential(scale=1/lambda_param, size=len(df) - 1)
    elif dist == "gaussian":
        times = np.random.normal(loc=ev.total_seconds, size=len(df) - 1)

    # Calculate the new timestamps based on the exponential inter-arrival times
    new_timestamps = [df['timestamp'].iloc[0]]
    for time_diff in times:
        new_timestamps.append(new_timestamps[-1] + pd.Timedelta(seconds=time_diff))

    # Assign the new timestamps back to the DataFrame
    df['timestamp'] = new_timestamps

    # convert back to timestamp in seconds 
    df['timestamp'] = df['timestamp'].astype(int) // 10**9

    return df

if __name__ == "__main__":
    experiment = "ml-1m"
    dataset_dir = use_dataset(experiment)
    result_dir = use_results(experiment, download=False)

    splits = [0.5] 
    dists = ["gaussian", "exponential"] 

    for split in splits: 
        for dist in dists: 
            test_df = pd.read_csv(f'{dataset_dir}/stream_{split}.csv')
            train_df = pd.read_csv(f'{dataset_dir}/train_{split}.csv')
            
            updated_test_df = reset_timestamps(test_df) 
            updated_train_df = reset_timestamps(train_df) 


            updated_test_df.to_csv(f'{dataset_dir}/stream_{split}_{dist}.csv')
            updated_train_df.to_csv(f'{dataset_dir}/train_{split}_{dist}.csv')
    

