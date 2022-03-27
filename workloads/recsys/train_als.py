import pandas as pd
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

dataset_dir = use_dataset("ml-1m")
result_dir = use_results("ml-1m")
workers = 40

def split_data(df):  
    start_ts = df['timestamp'].min()
    med_ts = df['timestamp'].quantile(.25)
    end_ts = df['timestamp'].max()
    train_df = df[df['timestamp'] <= med_ts]
    stream_df = df[df['timestamp'] > med_ts]
    seen_movies = set(train_df['movie_id'])
    print(len(seen_movies), len(set(stream_df['movie_id'])), len(stream_df))
    stream_df = stream_df.drop(stream_df[stream_df['movie_id'].map(lambda x: x not in seen_movies)].index)
    train_df.to_csv(f'{dataset_dir}/train.csv', header=True, index = False)
    stream_df.to_csv(f'{dataset_dir}/stream.csv', header=True, index = False)
    return start_ts, med_ts, end_ts

ratings_path = f"{dataset_dir}/ratings.csv"
ratings_df = pd.read_csv(ratings_path)
ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

start_ts, med_ts, end_ts = split_data(ratings_df)
train_df = pd.read_csv(f'{dataset_dir}/train.csv')
test_df = pd.read_csv(f'{dataset_dir}/stream.csv')


n = ratings_df.user_id.max()+1
m = ratings_df.movie_id.max()+1
d = 50 
print(n, m, d)

def initialize_features(n, m, d):
    #user_features = np.random.randn(n, d+1)
    #movie_features = np.random.randn(m, d+1)
    user_features = np.random.normal(loc=0., scale=0.01, size=(n, d+1))
    movie_features = np.random.normal(loc=0., scale=0.01, size=(m, d+1))
    user_features[:, -1] = 0
    movie_features[:, -1] = 0
    return user_features, movie_features

#model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], fit_intercept=True)


# ratings is an n by m sparse matrix
def update_user(user_id, user_data, d=50):
   # user_data = ratings.getrow(uid)
    user_data = user_data.tocoo() # lookup all rating data for this user
    movie_ids = user_data.col
    values = user_data.data
    model = Ridge(alpha=0.001, fit_intercept=True)

    #user_data = ratings.getrow(user_id).tocoo() # lookup all rating data for this user
    #movie_ids = user_data.col
    #values = user_data.data
        #movie_ids and values are both k x 1 
    k = len(movie_ids)
    X = movie_features[movie_ids, :d] # k x d matrix 
    # The movie bias is the d+1 (last column) of the movie features
    movie_biases = movie_features[movie_ids, -1] # k x 1 matrix
    # Subtract off the movie biases
    Y = values - movie_biases

    # Use Sklearn to solve l2 regularized regression problem
    #from sklearn.lienar_model import Ridge, RidgeCV # we could use the CV version ...
    #model = Ridge(alpha=alpha) #Maybe use RidgeCV() instead so we don't tune lambda
    model.fit(X, Y)
    
    #print(model.predict(X))
    #print(Y)

    #user_features[user_id] = np.append(model.coef_, model.intercept_)
    #user_features[user_id, -1] = model.intercept_
    
    #print("coef", model.coef_)
    #print(X @ model.coef_.T + model.intercept_)
    
    #print("pred", [predict_user_movie_rating(user_features[user_id, :], movie_features[mid, :d]) for mid in list(movie_ids)])

    return np.append(model.coef_, model.intercept_)

def update_movie(movie_id, movie_data, d=50):
    
    #movie_data = ratings.getcol(mid)
    movie_data = movie_data.tocoo() # lookup all rating data for this user
    user_ids = movie_data.row
    values = movie_data.data
    model = Ridge(alpha=0.001, fit_intercept=True)

    #movie_data = ratings.getcol(movie_id).tocoo() # lookup all rating data for this user
    #user_ids = movie_data.row
    #values = movie_data.data
        #movie_ids and values are both k x 1 
    k = len(user_ids)
    X = user_features[user_ids, :d] # k x d matrix 
    # The movie bias is the d+1 (last column) of the movie features
    user_biases = user_features[user_ids, -1] # k x 1 matrix
    # Subtract off the movie biases
    Y = values - user_biases

    # Use Sklearn to solve l2 regularized regression problem
    #from sklearn.lienar_model import Ridge, RidgeCV # we could use the CV version ...
    #model = Ridge(alpha=alpha) #Maybe use RidgeCV() instead so we don't tune lambda
    #model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(X, Y)
    
    #movie_features[movie_id] = np.append(model.coef_, model.intercept_)

    return np.append(model.coef_, model.intercept_)

def loss(ratings): 
    #mf = movie_features[ratings.col, :d]
    #uf = user_features[ratings.row, :d]
    #y_true = ratings.data
    #y_pred = [predict_user_movie_rating(uf[i], mf[i]) for i in range(len(y_true))]
    y_pred = []
    y_true = []
    for item in ratings.items():
        y_pred.append(predict_user_movie_rating(user_features[item[0][0], :], movie_features[item[0][1], :]))
        y_true.append(item[1])
    print(np.array(y_pred))
    print(y_true[:10])
    return mean_squared_error(y_true, y_pred)
    

def predict_user_movie_rating(user_feature, movie_feature):
    return user_feature[:d] @ movie_feature[:d] + user_feature[-1] + movie_feature[-1] 

ratings = coo_matrix(
    (
        train_df.rating.to_numpy(), 
        (train_df.user_id.to_numpy(dtype=np.int32), train_df.movie_id.to_numpy(dtype=np.int32))),
    shape=(n, m)
)
# convert to dok matrix so can be updated 
ratings = dok_matrix(ratings)

user_features, movie_features = initialize_features(n, m, d)
uids = list(set([k[0] for k in ratings.keys()]))
mids = list(set([k[1] for k in ratings.keys()]))

pickle.dump(ratings, open(f"{result_dir}/ratings.pkl", "wb"))
# store past updates 
past_updates = {uid: ratings.getrow(uid).size for uid in uids}
pickle.dump(past_updates, open(f"{result_dir}/past_updates.pkl", "wb"))

for i in range(5): 
    print("update users...")
    futures = []
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for uid in tqdm(uids):
            user_data = ratings.getrow(uid)
            #print(update_user(uid, movie_ids, values))
            futures.append(executor.submit(update_user, uid, user_data))
            
    for uid, f in tqdm(zip(uids, futures)):
        user_features[uid] = f.result()
        
        
    print("loss", loss(ratings))
    print("update movies...")
    futures = []
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        for mid in tqdm(mids): 
            movie_data = ratings.getcol(mid)
            futures.append(executor.submit(update_movie, mid, movie_data))

    for mid, f in tqdm(zip(mids, futures)):
        movie_features[mid] = f.result()
    print("loss", loss(ratings))
    pickle.dump(user_features, open(f"{result_dir}/train_user_features.pkl", "wb"))
    pickle.dump(movie_features, open(f"{result}/train_movie_features.pkl", "wb"))
