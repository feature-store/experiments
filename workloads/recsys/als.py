__author__ = 'paulthompson'
import pickle
import torch

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from workloads.util import use_results, use_dataset, read_config, log_dataset

#tags_file ='/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/tags.txt'
#ratings_file = '/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/ratings.txt'
#movies_file = '/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/movies.txt'


#dataset_dir = use_dataset("ml-25m")
dataset_dir = use_dataset("ml-latest-small")
ratings_path = f"{dataset_dir}/ratings.csv"
tags_path = f"{dataset_dir}/tags.csv"
movies_path = f"{dataset_dir}/movies.csv"


#def getInitialMatrix():
#    '''
#    Gets data from files and creates user-item matrices
#    :return: A, R user-item matrices
#    '''
#    #tags = pd.read_table(tags_file, sep=':', header=None, names=['user_id', 'movie_id', 'tag', 'timestamp'])
#    #ratings = pd.read_table(ratings_file, sep=':', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
#    #movies = pd.read_table(movies_file, sep=':', header=None, names=['movie_id', 'title', 'genres'])
#
#    tags = pd.read_csv(tags_path)
#    tags.columns = ['user_id', 'movie_id', 'tag', 'timestamp']
#    ratings = pd.read_csv(ratings_path)
#    ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
#    movies = pd.read_csv(movies_path)
#    movies.columns = ['movie_id', 'title', 'genres']
#
#    print("Join movies, ratings, and tags data frames together...")
#    combined_df = ratings.join(movies, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
#    del combined_df['movie_id_r']; del combined_df['user_id_t']; del combined_df['movie_id_t']; del combined_df['timestamp_t']
#
#    combined_df = combined_df[0:5054]
#
#    print("Getting 'A' matrix with rows: user and columns: movies...")
#    A = combined_df.pivot_table(columns=['movie_id'], index=['user_id'], values='rating').fillna(0).values
#
#    print(" 'A' matrix shape is", A.shape)
#
#    print("Getting 'R' Binary Matrix of rating or no rating...")
#    R = A>0.5; R[R == True] = 1; R[R == False] = 0; R = R.astype(np.float64, copy=False)
#
#
#    return A, R

def getInitialMatrix(): 

    A = pickle.load(open(f"{dataset_dir}/A.pkl", "rb"))
    R = pickle.load(open(f"{dataset_dir}/R.pkl", "rb"))

    return A, R

def runALS(A, R, n_factors, n_iterations, lambda_, user_matrix=None, movie_matrix=None, users=None):
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

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        #print(np.dot(Users, Items))
        #return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)
        return torch.sum((R * (A - torch.mm(Users, Items))) ** 2) / torch.sum(R)


    device = torch.device('cuda')
    Users = torch.tensor(Users).to('cuda')
    Items = torch.tensor(Items).to('cuda')
    A = torch.tensor(A).to('cuda')
    R = torch.tensor(R).to('cuda')
    eye = torch.eye(n_factors).cuda()
    lambda_ = torch.tensor(lambda_).to('cuda')
    n_factors = torch.tensor(n_factors).to('cuda')

    MSE_List = []

    print("Starting Iterations")
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            if users is not None and i not in users: 
                continue 
            #print("updating user", i)
            #Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lambda_ * np.eye(n_factors), np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T
            mat_a = torch.mm(Items, torch.mm(torch.diag(Ri), Items.T)) + lambda_* eye 
            mat_b = Items @ (torch.diag(Ri) @ A[i].T)
            #print("a", mat_a.shape, np.dot(Items.cpu(), np.dot(np.diag(Ri.cpu()), Items.cpu().T)).shape)
            #print("b", mat_b.shape, np.dot(Items.cpu(), np.dot(np.diag(Ri.cpu()), A[i].cpu().T)).shape)
            Users[i] = torch.linalg.solve(mat_a, mat_b).T
        #print("Error after solving for User Matrix:", get_error(A, Users, Items, R))

        for j, Rj in enumerate(R.T):
            #Items[:,j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lambda_ * np.eye(n_factors), np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))
            Items[:,j] = torch.linalg.solve(
                torch.mm(Users.T, torch.mm(torch.diag(Rj), Users)) + lambda_ * eye,
                Users.T @ (torch.diag(Rj) @ A[:, j])
            )
        print("Error after solving for Item Matrix:", get_error(A, Users, Items, R))

        MSE_List.append(get_error(A, Users, Items, R))
        print('%sth iteration is complete...' % iter)

    print(MSE_List)
    return Users.cpu().detach().numpy(), Items.T.cpu().detach().numpy()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.plot(range(1, len(MSE_List) + 1), MSE_List); plt.ylabel('Error'); plt.xlabel('Iteration')
    #plt.title('Python Implementation MSE by Iteration \n with %d users and %d movies' % A.shape);
    #plt.savefig('Python MSE Graph.pdf', format='pdf')
    #plt.show()


if __name__ == '__main__':
    A, R = getInitialMatrix()
    Users, Items = runALS(A, R, n_factors = 10, n_iterations = 20, lambda_ = .1)

    pickle.dump(Users, open("trained_users.pkl", "wb")) 
    pickle.dump(Items, open("trained_items.pkl", "wb"))


