import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

import ast
import os
import pickle
import json


from pprint import pprint

import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset, join_queries_features
from workloads.recsys.recsys_util import get_features, get_feature_update, predict_rating

dataset_dir = use_dataset("ml-latest-small")
results_dir = use_results("ml-latest-small")
movie_to_index = json.load(open(f"{dataset_dir}/movie_to_index.json", "r"))
user_to_index = json.load(open(f"{dataset_dir}/user_to_index.json", "r"))
user_matrix = pickle.load(open(f"{dataset_dir}/user_matrix.pkl", "rb"))
movie_matrix = pickle.load(open(f"{dataset_dir}/movie_matrix.pkl", "rb"))
A = pickle.load(open(f"{dataset_dir}/A.pkl", "rb"))
R = pickle.load(open(f"{dataset_dir}/R.pkl", "rb"))
events_df = pd.read_csv(f'{dataset_dir}/test.csv')

oracle_user_matrix = pickle.load(open(f"{results_dir}/oracle_models/user_matrix_5665418.pkl", "rb")).cpu().detach().numpy()
oracle_movie_matrix = pickle.load(open(f"{results_dir}/oracle_models/movie_matrix_5665418.pkl", "rb")).cpu().detach().numpy()


def evaluate(experiment): 

    results_df = pd.read_csv(f'{results_dir}/{experiment}.csv')
    timestamp_df = pd.read_csv(f'{results_dir}/timestamps/{experiment}.csv').set_index("timestamp_ms")

    
    def predict_oracle(user):
        ui = user_to_index[str(user)]
        return [
            np.dot(np.array(oracle_user_matrix[ui]), oracle_movie_matrix[mid]) 
            for mid in range(oracle_movie_matrix.shape[0])
        ]


    def predict(user, ts): 
        ui = user_to_index[str(user)]
        processing_time = timestamp_df.loc[ts].timestamp

        # processed updates in results
        update_df = results_df[(results_df["user_id"] == user) & (results_df["processing_time"] <= processing_time)]
        processed_df = events_df[(events_df["user_id"] == user) & (events_df["timestamp"] <= update_df.timestamp.max())]

        # all events that should have been processed
        total_df = events_df[(events_df["user_id"] == user) & (events_df["timestamp"] <= ts)]

        if len(total_df.index) == 0: 
            return None

        if len(update_df.index) == 0:
            feature = user_matrix[ui]
            update_ts = total_df.timestamp.min()
        else:
            update_ts = update_df.timestamp.max()
            feature = ast.literal_eval(update_df.user_features.tolist()[-1])
        
        staleness = timestamp_df.loc[total_df.timestamp.max()].timestamp - timestamp_df.loc[update_ts].timestamp

        # metrics 
        past_updates = len(processed_df.index)
        pending_updates = len(total_df.index) - past_updates
        
        prediction = [
            np.dot(np.array(feature), movie_matrix[mid]) 
            for mid in range(movie_matrix.shape[0])
        ]

        return {"prediction": prediction, "ts": ts, "user": user, "past_updates": past_updates, "pending_updates": pending_updates}


    oracle_pred = {}
    for user in events_df.user_id.value_counts().index.tolist():
        oracle_pred[user] = predict_oracle(user)

    data = []
    for ts in range(100, 35000, 500): 
        for user in events_df.user_id.value_counts().index.tolist():
            res = predict(user, ts)
            if res is None: 
                continue
            res["error"] = mean_squared_error(oracle_pred[user], res["prediction"])
            data.append(res)

        df = pd.DataFrame(data)
        df.to_csv(f"{experiment}.csv")
        print("timestamp", ts)
        print("    error:", df.error.mean())
        print("    pending:", df.pending_updates.mean())
        print("    past:", df.past_updates.mean())

    print(f"{experiment}.csv")


if __name__ == "__main__":

    experiment = "results_user_workers_1_random_learningrate_0.02_userfeaturereg_0.01_sleep_0.001"
    evaluate(experiment)


