import pandas as pd
import concurrent.futures
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from sklearn.metrics import mean_squared_error

import pickle
from collections import defaultdict
import json

from tqdm import tqdm 
import concurrent.futures


import pandas as pd, numpy as np, matplotlib.pyplot as plt

import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset


# ratings is an n by m sparse matrix
def update_user(user_id, user_data, movie_features, d=50):
   # user_data = ratings.getrow(uid)
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
    
    return np.append(model.coef_, model.intercept_)

def loss(ratings): 
    y_pred = []
    y_true = []
    for item in ratings.items():
        y_pred.append(predict_user_movie_rating(user_features[item[0][0], :], movie_features[item[0][1], :]))
        y_true.append(item[1])
    return mean_squared_error(y_true, y_pred)
    

def predict_user_movie_rating(user_feature, movie_feature, d=50):
    return user_feature[:d] @ movie_feature[:d] + user_feature[-1] + movie_feature[-1] 

class UserEventQueue: 
    
    """
    Event queue that selects group of user updates
    (note: we can have another implementation which triggers a bunch of updates together)
    """
    
    def __init__(self, num_keys, policy, past_updates): 
        self.policy = policy 
        self.num_keys = num_keys
        
        # metric tracking
        self.total_error = np.zeros((num_keys))
        self.past_updates = past_updates
        self.queue = defaultdict(list)
        self.staleness = defaultdict(lambda: 0)
        
    def push(self, uid, mid, rating, user_features, movie_features): 
        
        # calcualte error 
        pred = predict_user_movie_rating(user_features[uid], movie_features[mid])
        error = abs(pred - rating) #error*error # square 
        
        if uid not in self.past_updates and self.policy == "total_error_cold": # unseen keys start with high total error
            self.total_error[uid] = 1000000
        else:
            # update per key 
            self.total_error[uid] += error
        self.queue[uid].append((uid, mid, rating))
        
    def arg_max(self, data_dict): 
        max_key = None
        max_val = None
        valid_keys = 0
        for key in self.queue.keys():
            
            # cannot select empty queue 
            if len(self.queue[key]) <= 0: 
                continue 
                
            valid_keys += 1
            value = data_dict[key]
            if max_key is None or max_val <= value: 
                assert key is not None, f"Invalid key {data_dict}"
                max_key = key
                max_val = value
        return max_key, max_val
        
        
    def choose_key(self): 
        if self.policy == "total_error_cold" or self.policy == "total_error":
            key = self.arg_max(self.total_error)
        elif self.policy == "max_pending":
            key = self.arg_max({key: len(self.queue[key]) for key in self.queue.keys()})
        elif self.policy == "min_past": 
            key = self.arg_max({key: 1/(self.past_updates.setdefault(key, 0)+1) for key in self.queue.keys()})
        elif self.policy == "round_robin": 
            key = self.arg_max(self.staleness)
        else: 
            raise ValueError("Invalid policy")
       
        assert key is not None or sum([len(v) for v in self.queue.values()]) == 0, f"Key is none, {self.queue}"
        return key 
    
    def pop(self): 
        key, score = self.choose_key()
        #print(key, score, self.past_updates.setdefault(key, 0))
        if key is None:
            #print("no updates", self.queue)
            return None 
        events = self.queue[key]
        self.queue[key] = []

        # update metrics 
        for k in self.queue.keys():
            self.staleness[k] += 1
        self.staleness[key] = 0
        self.total_error[key] = 0
        
        # TODO: this is wrong
        self.past_updates[key] = self.past_updates.setdefault(key, 0) + len(events)
            
        return key 

def experiment(policy, updates_per_ts, dataset_dir=".", result_dir=".", limit=None, d=50): 

    # read data 
    test_df = pd.read_csv(f'{dataset_dir}/stream.csv')
    train_df = pd.read_csv(f'{dataset_dir}/train.csv')
    start_ts = test_df.timestamp.min()
    test_df.timestamp = test_df.timestamp.apply(lambda ts: int((ts - start_ts)/100))
    data = {}
    for ts, group in tqdm(test_df.groupby("timestamp")):
        data[ts] = group.to_dict("records")
    ratings = pickle.load(open(f"{result_dir}/ratings.pkl", "rb"))
    user_features = pickle.load(open(f"{result_dir}/train_user_features.pkl", "rb"))
    movie_features = pickle.load(open(f"{result_dir}/train_movie_features.pkl", "rb"))
    train_user_features = pickle.load(open(f"{result_dir}/train_user_features.pkl", "rb"))
    train_movie_features = pickle.load(open(f"{result_dir}/train_movie_features.pkl", "rb"))

    past_updates = pickle.load(open(f"{result_dir}/past_updates.pkl", "rb"))
    
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

    queue = UserEventQueue(user_features.shape[0], policy, past_updates) 
    for ts in tqdm(list(data.keys())[:limit]): 

        # process events
        updated_users = set([])
        for event in data[ts]:

            uid = event["user_id"]
            mid = event["movie_id"]
            rating = event["rating"]

            ratings[uid, mid] = rating 
            queue.push(uid, mid, rating, user_features, movie_features)
            #print("orig loss:", loss(ratings))
            y_pred.append(predict_user_movie_rating(user_features[uid], movie_features[mid]))
            y_true.append(rating)
            timestamps.append(ts)
            users.append(uid)
            movies.append(mid)
            
        for i in range(updates_per_ts): 
            uid = queue.pop()
            if uid is None: 
                #print(f"{ts}: No updates in queue")
                break 
            user_features[uid] = update_user(uid, ratings, movie_features, d)
            update_times[uid].append(ts)
            #print("update", uid)
            
            # TODO: make sure overall loss is decreasing (training error)
        
            
         
    runtime = 1/updates_per_ts
    update_df = pd.DataFrame([
        [policy, runtime, k, i, update_times[k][i]]
        for k, v in update_times.items() for i in range(len(v))
    ], columns = ["policy", "runtime", "key", "i", "time"])
    results_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "user_id": users, "movie_id": movies, "timestamp": timestamps})
    

    print("saving", f"{result_dir}/{policy}_{updates_per_ts}_results.csv")
    update_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_updates.csv")
    results_df.to_csv(f"{result_dir}/{policy}_{updates_per_ts}_results.csv")

if __name__ == "__main__":


    name = "ml-1m"
    dataset_dir = use_dataset(name)
    result_dir = use_results(name)
    #os.makedirs(result_dir, exist_ok=True)
    workers = 40
    limit = None
    
    policies = ["total_error", "total_error_cold", "max_pending", "min_past", "round_robin"]
    #policies = ["min_past"]
    updates_per_ts = [1, 2, 4, 8, 10, 100]

    
    #experiments = [(p, u, ".") for p in policies for u in updates_per_ts]
    futures = []
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for p in policies: 
            for u in updates_per_ts: 
                futures.append(executor.submit(experiment, p, u, dataset_dir,  result_dir, limit))
         
        for f in futures: 
            try: 
                f.result()
            except Exception as e:
                print(e)
        res = concurrent.futures.wait(futures)
        executor.shutdown()
    
    