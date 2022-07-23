import json 
import time
from collections import defaultdict
import os
from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
from absl import app, flags
from sktime.performance_metrics.forecasting import mean_squared_scaled_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from tqdm import tqdm
from statsmodels.tsa.seasonal import STL

from workloads.util import use_results, use_dataset, read_config, log_dataset, log_results

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "window_size",
    default=48,
    help="Size of window to fit"
)
flags.DEFINE_integer(
    "num_keys",
    default=67,
    help="Number of keys to run on"
)
flags.DEFINE_integer(
    "max_len",
    default=700,
    help="Length of time-series"
)
def simulate(data, start_ts, runtime, policy):
    """
    Simulate streaming computation with given runtime and policy 

    :data: Dict[str] -> list mapping keys to values
    :start_ts: timestamp (i.e. index) to start at
    :runtime: runtime of map function 
    :policy: policy to select keys 
    """

    predictions = defaultdict(list)
    values = defaultdict(list)
    staleness = defaultdict(list)
    score = [0 for i in range(1, FLAGS.num_keys+1, 1)]
    update_times = defaultdict(list)
   
    # start with initial set of models
    last_model = {key: get_model(data, key, start_ts) for key in range(1, FLAGS.num_keys+1, 1)} 
    next_update_time = start_ts #+ runtime # time when model is completed
    last_update_time = start_ts

    for ts in tqdm(range(start_ts, FLAGS.max_len, 1)):

        # run predictions per key 
        for key in last_model.keys(): 
            last_time = last_model[key]["time"]
            predictions[key].append(last_model[key]["forecast"][ts-last_time])
            values[key].append(float(data[key][ts]))
            staleness[key].append(ts-last_time)

            # policy scoring
            t = ts - last_time
            if policy == "total_error" and len(predictions[key]) > 1 and t > 1: 
                e = mean_absolute_scaled_error(
                    np.array(values[key][-t:]), 
                    np.array(predictions[key][-t:]), 
                    y_train=np.array(values[key][-t:]), 
                    sp=1
                )
                score[key-1] = e * t # use total, not mean 
            elif policy == "round_robin": 
                score[key-1] += 1
            elif policy == "max_staleness": 
                score[key-1] = ts-last_time

        if policy == "batch": 
            print(ts - last_update_time, runtime, len(score))
            if (ts - last_update_time) / runtime >= len(score): 

                # update all keys
                for i in range(len(score)): 
                    key = i + 1

                    # mark as update time for key 
                    update_times[key].append(ts) 
                    last_model[key] = get_model(data, key, ts)

                last_update_time = ts
        else:

            # can update model
            while ts >= next_update_time: 
                # pick max error key 
                key = np.array(score).argmax() + 1
                #print(key, score)

                if max(score) == 0: # nothing to update
                    print("nothing to update", ts)
                    break
                
                # mark as update time for key 
                update_times[key].append(ts) 
                last_model[key] = get_model(data, key, ts)
                score[key-1] = 0
                
                # update next update time 
                next_update_time += runtime

        print(policy, update_times.keys())

    results_df = pd.concat([
        pd.DataFrame({
            "y_pred": predictions[key], 
            "y_true": values[key], 
            "staleness": staleness[key], 
            "key": [key] * len(predictions[key])
        })
        for key in predictions.keys()
    ])
    return update_times, results_df

def remove_anomaly(df): 
    for index, row in df.iterrows(): 
        if not row["is_anomaly"] or index < FLAGS.window_size: continue 
            
        chunk = df.iloc[index-FLAGS.window_size:index].value
        model = STLForecast(
            chunk, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=24
        ).fit()
        row["value"] = model.forecast(1).tolist()[0]
        df.iloc[index] = pd.Series(row)

    return df

def error(df):
    """
    Calculate total error given prediction dataframe 
    """
    total = 0
    for key in df.groupby("key").groups.keys(): 
        e = mean_absolute_scaled_error(
            df[df["key"] == key].y_pred, 
            df[df["key"] == key].y_true, 
            y_train=df[df["key"] == key].y_true
        )
        total += e
    return total 

def get_model(data, key, ts): 
    """
    Get STL forecast model 
    
    :data: Dict[str] -> list mapping keys to values
    :key: key to fit 
    :ts: last timestamp (fit ts - window of data)
    """

    chunk = data[key][ts - FLAGS.window_size: ts]
    st = time.time()
    last_model = STLForecast(
        chunk, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=24
    ).fit()
    return {"model": last_model, "forecast": last_model.forecast(2000), "data": chunk, "time": ts, "runtime": time.time() - st}

def read_data(dataset_dir):
    """
    Read data for each key and return dict 

    :dataset_dir: dir containing CSV files
    """

    data = {}
    for i in tqdm(range(1, FLAGS.num_keys+1)):
        df = remove_anomaly(pd.read_csv(f"{dataset_dir}/{i}.csv"))
        assert len(df.index) >= FLAGS.max_len, f"Dataset size too small {len(df.index)}"
        arr = df.value.values 
        data[i] = arr 
    return data

def main(argv):
    #runtime = [24, 12, 4, 2, 1, 0]
    #runtime = [4, 6, 8, 12, 24] #[1, 2, 3]
    policy = ["round_robin", "total_error", "batch"]
    #runtime = [0, 1000000]
    runtime = [0.05, 0.02] #0.5, 0.2, 0.1]
    name = f"yahoo_A1_window_{FLAGS.window_size}_keys_{FLAGS.num_keys}_length_{FLAGS.max_len}"

    result_dir = use_results(name)
    dataset_dir = use_dataset("yahoo/A1")

    # aggregate data structures
    results_df = pd.DataFrame()
    updates_df = pd.DataFrame()
    df_all = pd.DataFrame()

    data = read_data(dataset_dir)
    
    for r in runtime: 
        for p in policy: 

            try:
                update_times, df = simulate(data, start_ts=FLAGS.window_size, runtime=r, policy=p)
            except Exception as e:
                print(e) 
                continue
            e = error(df)
            s = df.staleness.mean()
            u = sum([len(v) for v in update_times.values()])
           
            r_df = pd.DataFrame([[r, p, e, s, u]])
            r_df.columns = ["runtime", "policy", "total_error", "average_staleness", "total_updates"]
            u_df = pd.DataFrame([
                [r, p, k, i, update_times[k][i]]
                for k, v in update_times.items() for i in range(len(v))
            ])
           
            # write experiment CSV
            folder = f"{p}_{r}_A1"
            os.makedirs(f"{result_dir}/{folder}", exist_ok=True)
            df.to_csv(f"{result_dir}/{folder}/simulation_predictions.csv")
            r_df.to_csv(f"{result_dir}/{folder}/simulation_result.csv")
            u_df.to_csv(f"{result_dir}/{folder}/simulation_update_time.csv")
            print(u_df)
            u_df.columns = ["runtime", "policy", "key", "i", "time"]
            u_df.to_csv(f"{result_dir}/{folder}/simulation_update_time.csv")
           
            # aggregate data 
            df_all = pd.concat([df_all, df])
            results_df = pd.concat([results_df, r_df])
            updates_df = pd.concat([updates_df, u_df])
            print("done", folder)

	
            
    results_df.to_csv(f"{result_dir}/simulation_results.csv")
    while True:
        try:
            log_results(name)
            break
        except Exception as e:
            print(e) 
            time.sleep(5)



if __name__ == "__main__":
    app.run(main)
