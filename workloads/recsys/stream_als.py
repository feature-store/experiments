import pandas as pd
import time
import concurrent.futures
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

import time

from sklearn.metrics import mean_squared_error

import pickle
from collections import defaultdict
import json

from absl import app, flags
from tqdm import tqdm 
import concurrent.futures


import pandas as pd, numpy as np, matplotlib.pyplot as plt

import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset
from workloads.queue import UserEventQueue, Policy

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dataset",
    default="ml-1m",
    help="Dataset to use", 
    required=False
)

flags.DEFINE_float(
    "split",
    default=None,
    help="Data split",
    required=True,
)

flags.DEFINE_integer(
    "dimentions",
    default=50,
    help="Dimentionality of features",
    required=False,
)

flags.DEFINE_integer(
    "workers",
    default=50,
    help="Number of workers to use",
    required=False,
)

flags.DEFINE_boolean(
    "download_data",
    default=False,
    help="Whether to download required data",
    required=False,
)



# ratings is an n by m sparse matrix
def update_user(user_id, user_data, movie_features, d=50, runtime_file=None):
   # user_data = ratings.getrow(uid)
    st = time.time()
    user_data = user_data.tocoo() # lookup all rating data for this user
    movie_ids = user_data.col
    values = user_data.data

    k = len(movie_ids)
    X = movie_features[movie_ids, :d] # k x d matrix 

    # The movie bias is the d+1 (last column) of the movie features
    movie_biases = movie_features[movie_ids, -1] # k x 1 matrix
    # Subtract off the movie biases
    Y = values - movie_biases

    # Use Sklearn to solve l2 regularized regression problem
    #model = Ridge(alpha=alpha) #Maybe use RidgeCV() instead so we don't tune lambda
    model = Ridge(alpha=0.001, fit_intercept=True)
    model.fit(X, Y)

    runtime = time.time() - st

    if runtime_file is not None: 
        open(runtime_file, "a").write(str(runtime) + "\n")

    #print(time.time() - st)
    
    return np.append(model.coef_, model.intercept_)

def loss(ratings, user_features, movie_features): 
    y_pred = []
    y_true = []
    for item in ratings.items():
        y_pred.append(predict_user_movie_rating(user_features[item[0][0], :], movie_features[item[0][1], :]))
        y_true.append(item[1])
    return mean_squared_error(y_true, y_pred)
    

def predict_user_movie_rating(user_feature, movie_feature, d=50):
    p = user_feature[:d] @ movie_feature[:d] + user_feature[-1] + movie_feature[-1] 
    if p < 1: 
        return 1
    if p > 5: 
        return 5
    return p

def experiment(policy, updates_per_ts, ts_factor, dataset_dir=".", result_dir=".", limit=None, d=50, split=0.5): 


    # create/clear runtime file
    runtime_file = f"{result_dir}/runtimes.txt"
    open(runtime_file, "w").close()
    print(runtime_file)

    # read data 
    test_df = pd.read_csv(f'{dataset_dir}/stream_{split}.csv')
    train_df = pd.read_csv(f'{dataset_dir}/train_{split}.csv')
    start_ts = test_df.timestamp.min()
    test_df.timestamp = test_df.timestamp.apply(lambda ts: int((ts - start_ts)/ts_factor))
    data = {}
    for ts, group in tqdm(test_df.groupby("timestamp")):
        data[ts] = group.to_dict("records")
    ratings = pickle.load(open(f"{result_dir}/ratings_{split}.pkl", "rb"))
#
    # updated features
    user_features = pickle.load(open(f"{result_dir}/train_user_features_{split}.pkl", "rb"))
    movie_features = pickle.load(open(f"{result_dir}/train_movie_features_{split}.pkl", "rb"))



    # original features
    train_user_features = pickle.load(open(f"{result_dir}/train_user_features_{split}.pkl", "rb"))
    train_movie_features = pickle.load(open(f"{result_dir}/train_movie_features_{split}.pkl", "rb"))

    past_updates = pickle.load(open(f"{result_dir}/past_updates_{split}.pkl", "rb"))
    
    print(user_features.shape[0], len(past_updates.keys()))
    for uid in range(user_features.shape[0]):
        if uid not in past_updates: 
            user_features[uid] = 0 #np.zeros(user_features[uid].shape)

    y_pred = [] 
    y_true = []
    users = []
    movies = []
    timestamps = []
    update_times = defaultdict(list)
    
    if limit is None: 
        limit = len(list(data.keys()))

    user_event_queue = UserEventQueue(user_features.shape[0], policy, past_updates)
    #for ts in tqdm(list(data.keys())[:limit]): 
    next_ts = 0
    update_budget = 0
    for ts in list(data.keys())[:limit]: 

        # process events
        for event in data[ts]:

            uid = event["user_id"]
            mid = event["movie_id"]
            rating = event["rating"]

            ratings[uid, mid] = rating 
            predicted_rating = predict_user_movie_rating(user_features[uid], movie_features[mid])
            y_pred.append(predicted_rating)
            y_true.append(rating)
            error_score = rating - predicted_rating
            user_event_queue.push(uid, error_score)
            #print("orig loss:", loss(ratings))
            timestamps.append(ts)
            users.append(uid)
            movies.append(mid)

        if policy == Policy.BATCH:
            update_budget += updates_per_ts
            if user_event_queue.size() <= update_budget:

                # update all uids in queue 
                while True: 
                    uid = user_event_queue.pop()
                    if uid is None: 
                        break
                    user_features[uid] = update_user(uid, ratings, movie_features, d, runtime_file)
                    update_times[uid].append(ts)
                update_budget = 0 

        elif updates_per_ts is not None and updates_per_ts >= 1:
            
            for i in range(updates_per_ts): 
                uid = user_event_queue.pop()
                if uid is None: 
                    #print(f"{ts}: No updates in queue")
                    break 
                user_features[uid] = update_user(uid, ratings, movie_features, d, runtime_file)
                update_times[uid].append(ts)
                
                # TODO: make sure overall loss is decreasing (training error)
        elif updates_per_ts is not None: 
            if ts >= next_ts:
                uid = user_event_queue.pop()
                if uid is None: 
                    #print(f"{ts}: No updates in queue")
                    break 
                user_features[uid] = update_user(uid, ratings, movie_features, d, runtime_file)
                update_times[uid].append(ts)
                next_ts = ts + 1/updates_per_ts
                #print(ts, next_ts, len(data[ts]))
            #else:
                #print(ts, len(data[ts]))

 

        if ts % 1000 == 0:
            runtime = 1/updates_per_ts if updates_per_ts is not None else None
            update_df = pd.DataFrame([
                [policy, runtime, k, i, update_times[k][i]]
                for k, v in update_times.items() for i in range(len(v))
            ], columns = ["policy", "runtime", "key", "i", "time"])
            results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "user_id": users, "movie_id": movies, "timestamp": timestamps})
    

            print("saving", {ts}, f"{result_dir}/{policy}_{updates_per_ts}_{ts_factor}_split_{split}_results.csv")
            update_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_{ts_factor}_split_{split}_updates.csv")
            results_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_{ts_factor}_split_{split}_results.csv")

         
    runtime = 1/updates_per_ts if updates_per_ts is not None else None
    update_df = pd.DataFrame([
        [policy, runtime, k, i, update_times[k][i]]
        for k, v in update_times.items() for i in range(len(v))
    ], columns = ["policy", "runtime", "key", "i", "time"])
    results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "user_id": users, "movie_id": movies, "timestamp": timestamps})
    

    update_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_{ts_factor}_split_{split}_updates.csv")
    results_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_{ts_factor}_split_{split}_results.csv")

def main(argv):

    dataset = FLAGS.dataset
    split = FLAGS.split # split between model train / stream 
    d = FLAGS.dimentions # dimentions 

    dataset_dir = use_dataset(dataset, download=FLAGS.download_data)
    result_dir = use_results(dataset, download=FLAGS.download_data)

    workers = FLAGS.workers

    limit = None
    
    policies = ["min_past"] #["round_robin", "query_proportional", "total_error_cold", "max_pending", "min_past", "round_robin"]
    #updates_per_ts = [7] #[100000] #[0.5, 0.2, None]
    #updates_per_ts = [None, 10000] #[4, 8, 16] #[100000] #[0.5, 0.2, None]
    updates_per_ts = [3, 4, 5, 8]
    #updates_per_ts = [3]
    #ts_factors = [60, 60*60, 60*60*24]
    #ts_factors = [10, 100]
    ts_factors = [60] #, 60*60, 60*60*24]
    
    #experiments = [(p, u, ".") for p in policies for u in updates_per_ts]
    futures = []
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for p in policies: 
            for u in updates_per_ts: 
                for ts_factor in ts_factors:
                    futures.append(executor.submit(experiment, p, u, ts_factor, dataset_dir,  result_dir, limit, d=d, split=split))
         
        for f in futures: 
            try: 
                f.result()
            except Exception as e:
                print(e)
        res = concurrent.futures.wait(futures)
        executor.shutdown()
    
if __name__ == "__main__":
    app.run(main)
 
