__author__ = 'paulthompson'
import pickle
import torch

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from workloads.util import use_results, use_dataset, read_config, log_dataset

#tags_file ='/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/tags.txt'
#ratings_file = '/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/ratings.txt'
#movies_file = '/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/movies.txt'


dataset_dir = use_dataset("ml-25m")
#dataset_dir = use_dataset("ml-latest-small")
ratings_path = f"{dataset_dir}/ratings.csv"
tags_path = f"{dataset_dir}/tags.csv"
movies_path = f"{dataset_dir}/movies.csv"

def getInitialMatrix(filename, gpu=True): 

    print("reading", filename)
    df = pd.read_csv(filename)
    print("done reading")
    print(df.user_id.max()+1, df.movie_id.max()+1)

    idx = df[["user_id", "movie_id"]].to_numpy().astype(int)
    val = df[["rating"]].to_numpy().astype(int).squeeze()
    has_val = (df["rating"] > 0).to_numpy().astype(int).squeeze()

    print(idx.shape)
    print(val.shape)
    print(has_val)

    #A = torch.sparse_coo_tensor(torch.tensor(idx).t(), val, (df.user_id.max()+1, df.movie_id.max()+1))
    #R = torch.sparse_coo_tensor(torch.tensor(idx).t(), has_val, (df.user_id.max()+1, df.movie_id.max()+1))

    A[idx[:,0], idx[:,1]] = torch.LongTensor(idx[:,2])
    R[idx[:,0], idx[:,1]] = torch.LongTensor(idx[:,2] > 0)

    if gpu:
        return A.cuda(), R.cuda()
    return A, R

#def getInitialMatrix(): 
#
#    A = pickle.load(open(f"{dataset_dir}/A.pkl", "rb"))
#    R = pickle.load(open(f"{dataset_dir}/R.pkl", "rb"))
#
#    return A, R

def runALS(A, R, n_factors, n_iterations, lambda_, user_matrix=None, movie_matrix=None, users=None, gpu=True):
    '''
    Runs Alternating Least Squares algorithm in order to calculate matrix.
    :param A: User-Item Matrix with ratings
    :param R: User-Item Matrix with 1 if there is a rating or 0 if not
    :param n_factors: How many factors each of user and item matrix will consider
    :param n_iterations: How many times to run algorithm
    :param lambda_: Regularization parameter
    :return:
    '''
    print("Initiating ")
    #lambda_ = 0.1; n_factors = 3; 
    n, m = A.shape
    #n_iterations = 20
    if user_matrix is None:
        Users = 5 * np.random.rand(n, n_factors)
    else: 
        Users = user_matrix

    if movie_matrix is None:
        Items = 5 * np.random.rand(n_factors, m)
    else: 
        Items = movie_matrix.T

    if gpu: 
        device = torch.device('cuda')
        Users = torch.tensor(Users).to('cuda')
        Items = torch.tensor(Items).to('cuda')
        A = torch.tensor(A).to('cuda')
        R = torch.tensor(R).to('cuda')
        lambda_ = torch.tensor(lambda_).to('cuda')
        n_factors = torch.tensor(n_factors).to('cuda')

    return step(Users, Items, A, R, lambda_, n_factors, n_iterations)


def step(Users, Items, A, R, lambda_, n_factors, n_iterations, users=None, gpu=True):

    MSE_List = []
    eye = torch.eye(n_factors).cuda()

    def get_error(A, Users, Items, R):
        if gpu:
            return torch.sum((R * (A - torch.mm(Users, Items))) ** 2) / torch.sum(R)
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)


    print("Starting Iterations")
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            if users is not None and i not in users: 
                continue 
            #print("updating user", i)
            if gpu: 
                mat_a = torch.mm(Items, torch.mm(torch.diag(Ri), Items.T)) + lambda_* eye 
                mat_b = Items @ (torch.diag(Ri) @ A[i].T)
                Users[i] = torch.linalg.solve(mat_a, mat_b).T
            else:
                Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lambda_ * np.eye(n_factors), np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T

        for j, Rj in enumerate(R.T):
            if gpu:
                Items[:,j] = torch.linalg.solve(
                    torch.mm(Users.T, torch.mm(torch.diag(Rj), Users)) + lambda_ * eye,
                    Users.T @ (torch.diag(Rj) @ A[:, j])
                )
            else:
                Items[:,j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lambda_ * np.eye(n_factors), np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))

        print("Error after solving for Item Matrix:", get_error(A, Users, Items, R))

        MSE_List.append(get_error(A, Users, Items, R))
        print('%sth iteration is complete...' % iter)

    print(MSE_List)
    if gpu: 
        #return Users.cpu().detach().numpy(), Items.T.cpu().detach().numpy()
        return Users, Items.T

    return Users, Items.T
    
if __name__ == '__main__':
    A, R = getInitialMatrix(f"{dataset_dir}/train.csv")
    Users, Items = runALS(A, R, n_factors = 10, n_iterations = 20, lambda_ = .1)

    pickle.dump(Users, open(f"{dataset_dir}/trained_users.pkl", "wb")) 
    pickle.dump(Items, open(f"{dataset_dir}/trained_items.pkl", "wb"))


