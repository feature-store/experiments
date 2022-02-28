from ast import Global
import torch
from absl import app, flags
import pickle

# might need to do  export PYTHONPATH='.'
from workloads.util import read_config, use_dataset, log_dataset, log_results, WriteFeatures
from workloads.recsys.als import runALS, step
from workloads.util import read_config, use_dataset, log_dataset, use_results
import pandas as pd
import json

FLAGS = flags.FLAGS

#flags.DEFINE_string(
#    "experiment",
#    default=None, 
#    help="Experiment name (corresponds to run name)",
#    required=True,
#)

def updated_embedding(events_df, A, R, user, ts): 

    df = events_df[(events_df["user_id"] == user) & (events_df["timestamp"] < ts)]
    
    # calculate embedding from dataframe of updates form a single user

    ui = user_to_index[str(ui)]
    streaming_user_matrix = np.array(user_matrix, copy=True) 
    A_matrix = np.array(A, copy=True)
    R_matrix = np.array(R, copy=True)
    user_feature_reg = 10

    for index, row in df.iterrows():
        ui = user_to_index[str(int(row["user_id"]))]
        mi = movie_to_index[str(int(row["movie_id"]))]
        A_matrix[ui][mi] = row["rating"]
        R_matrix[ui][mi] = 1
        
    Ri = R_matrix[ui]
    n_factors = len(user_matrix[ui])

    return np.linalg.solve(
        np.dot(movie_matrix.T, np.dot(np.diag(Ri), movie_matrix)) + user_feature_reg * np.eye(n_factors),
        np.dot(movie_matrix.T, np.dot(np.diag(Ri), A_matrix[ui].T))
    ).T

def main(argv):
    dataset_dir = use_dataset("ml-latest-small")
    results_dir = use_results("ml-latest-small")

    events_df = pd.read_csv(f'{dataset_dir}/test.csv')

    movie_to_index = json.load(open(f"{dataset_dir}/movie_to_index.json", "r"))
    user_to_index = json.load(open(f"{dataset_dir}/user_to_index.json", "r"))
    #user_matrix = pickle.load(open(f"{dataset_dir}/trained_users.pkl", "rb"))
    #movie_matrix = pickle.load(open(f"{dataset_dir}/trained_items.pkl", "rb"))
    user_matrix = pickle.load(open(f"{dataset_dir}/user_matrix.pkl", "rb"))
    movie_matrix = pickle.load(open(f"{dataset_dir}/movie_matrix.pkl", "rb"))
    A = pickle.load(open(f"{dataset_dir}/A.pkl", "rb"))
    R = pickle.load(open(f"{dataset_dir}/R.pkl", "rb"))

    print("A", A.shape)
    print("R", R.shape)
    print("R0", R[0].shape)

    user_matrix = torch.tensor(user_matrix).to('cuda')
    movie_matrix = torch.tensor(movie_matrix).to('cuda')
    A = torch.tensor(A).to('cuda')
    R = torch.tensor(R).to('cuda')
    lambda_ = torch.tensor(0.1).to('cuda')
    n_factors = torch.tensor(150).to('cuda')



    n_iterations = 20
    # move to GPU 

    # loop through timesteps 

    timestamps = sorted(list(set(events_df.timestamp.tolist())))

    timestamps = [events_df.timestamp.max()]

    print(len(timestamps))
    for ts in timestamps:

        df = events_df[events_df["timestamp"] <= ts]
        print("timestep", ts)
        for index, row in df.iterrows():
            ui = user_to_index[str(int(row["user_id"]))]
            mi = movie_to_index[str(int(row["movie_id"]))]
            A[ui][mi] = row["rating"]
            R[ui][mi] = 1


        user_matrix, movie_matrix = step(user_matrix, movie_matrix.T, A, R, 0.1, n_factors, n_iterations) 

        pickle.dump(user_matrix, open(f"{results_dir}/oracle_models/user_matrix_{ts}.pkl", "wb"))
        pickle.dump(movie_matrix, open(f"{results_dir}/oracle_models/movie_matrix_{ts}.pkl", "wb"))





if __name__ == "__main__":
    app.run(main)

