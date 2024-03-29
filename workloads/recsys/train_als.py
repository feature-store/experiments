import pandas as pd
import os
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

import pickle

from tqdm import tqdm 
import concurrent.futures

from absl import app, flags
import pandas as pd, numpy as np

import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset

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
    "resume",
    default=False,
    help="Whether to resume from existing checkpoint",
    required=False,
)

flags.DEFINE_boolean(
    "download_data",
    default=False,
    help="Whether to download required data",
    required=False,
)

def split_data(quantile, df, dataset_dir):  
    start_ts = df['timestamp'].min()
    med_ts = df['timestamp'].quantile(quantile)
    end_ts = df['timestamp'].max()
    train_df = df[df['timestamp'] <= med_ts]
    stream_df = df[df['timestamp'] > med_ts]
    seen_movies = set(train_df['movie_id'])
    stream_df = stream_df.drop(stream_df[stream_df['movie_id'].map(lambda x: x not in seen_movies)].index)
    print("number movies", len(seen_movies), len(set(stream_df['movie_id'])), "stream length", len(stream_df))
    train_df.to_csv(f'{dataset_dir}/train_{quantile}.csv', header=True, index = False)
    stream_df.to_csv(f'{dataset_dir}/stream_{quantile}.csv', header=True, index = False)
    return start_ts, med_ts, end_ts

def initialize_features(n, m, d):
    user_features = np.random.normal(loc=0., scale=0.01, size=(n, d+1))
    movie_features = np.random.normal(loc=0., scale=0.01, size=(m, d+1))
    user_features[:, -1] = 0
    movie_features[:, -1] = 0
    return user_features, movie_features

#model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], fit_intercept=True)


# ratings is an n by m sparse matrix
def update_user(movie_features, user_id, user_data, d=50):
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

def update_movie(user_features, movie_id, movie_data, d=50):
    
    #movie_data = ratings.getcol(mid)
    movie_data = movie_data.tocoo() # lookup all rating data for this user
    user_ids = movie_data.row
    values = movie_data.data

    k = len(user_ids)
    X = user_features[user_ids, :d] # k x d matrix 

    # The movie bias is the d+1 (last column) of the movie features
    user_biases = user_features[user_ids, -1] # k x 1 matrix
    # Subtract off the movie biases
    Y = values - user_biases

    # Use Sklearn to solve l2 regularized regression problem
    model = Ridge(alpha=0.001, fit_intercept=True)
    model.fit(X, Y)
    
    #movie_features[movie_id] = np.append(model.coef_, model.intercept_)

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
    if p < 1: p = 1
    if p > 5: p = 5
    return p 

def main(argv):

    dataset = FLAGS.dataset
    split = FLAGS.split # split between model train / stream 
    d = FLAGS.dimentions # dimentions 
    resume = FLAGS.resume

    dataset_dir = use_dataset(dataset, download=FLAGS.download_data)
    result_dir = use_results(dataset)
    workers = FLAGS.workers

    ratings_path = f"{dataset_dir}/ratings.csv"
    ratings_df = pd.read_csv(ratings_path)
    ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    
    start_ts, med_ts, end_ts = split_data(split, ratings_df, dataset_dir)
    train_df = pd.read_csv(f'{dataset_dir}/train_{split}.csv')
    test_df = pd.read_csv(f'{dataset_dir}/stream_{split}.csv')
    
    n = ratings_df.user_id.max()+1
    m = ratings_df.movie_id.max()+1
    print(n, m, d)
   
    # construct sparse ratings matrix
    ratings = coo_matrix(
    (
        train_df.rating.to_numpy(), 
        (
            train_df.user_id.to_numpy(dtype=np.int32), 
            train_df.movie_id.to_numpy(dtype=np.int32))
        ),
        shape=(n, m)
    )

    # get feature matrix
    movie_features_file = f"{result_dir}/train_movie_features_{split}.pkl"
    user_features_file = f"{result_dir}/train_user_features_{split}.pkl"
    if resume:
        user_features = pickle.load(open(user_features_file, "rb"))
        movie_features = pickle.load(open(movie_features_file, "rb"))
    else:
        user_features, movie_features = initialize_features(n, m, d)

    # convert to dok matrix so can be updated 
    ratings = dok_matrix(ratings)
    uids = list(set([k[0] for k in ratings.keys()]))
    mids = list(set([k[1] for k in ratings.keys()]))
   
    os.makedirs(result_dir, exist_ok=True)
    pickle.dump(ratings, open(f"{result_dir}/ratings_{split}.pkl", "wb"))
    # store past updates 
    print("Saving past updates..")
    past_updates = {uid: ratings.getrow(uid).size for uid in uids}
    pickle.dump(past_updates, open(f"{result_dir}/past_updates_{split}.pkl", "wb"))
    
    
    max_iter = 50
    for i in range(max_iter): 
        print("update users...")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            for uid in tqdm(uids):
                user_data = ratings.getrow(uid)
                #print(update_user(uid, movie_ids, values))
                futures.append(executor.submit(update_user, movie_features, uid, user_data))
                
        for uid, f in tqdm(zip(uids, futures)):
            user_features[uid] = f.result()
            
        print("update movies...")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(workers) as executor:
            for mid in tqdm(mids): 
                movie_data = ratings.getcol(mid)
                futures.append(executor.submit(update_movie, user_features, mid, movie_data))
    
        for mid, f in tqdm(zip(mids, futures)):
            movie_features[mid] = f.result()
        print("loss", loss(ratings, user_features, movie_features))
        
        pickle.dump(user_features, open(user_features_file, "wb"))
        pickle.dump(movie_features, open(movie_features_file, "wb"))

if __name__ == "__main__":
    app.run(main)
 
