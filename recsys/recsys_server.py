import numpy as np
import pandas as pd
import ray
from ralf.operator import Operator, DEFAULT_STATE_CACHE_SIZE
from ralf.operators import (
    Source,
)
from ralf.state import Record, Schema
from ralf.core import Ralf
from ralf.table import Table

NUM_MOVIES = 193609 # hard-coded for small dataset but could loop through source to check

@ray.remote
class RatingSource(Source):

    """Read in rows from MovieLens rating dataset.
    Each row provides a user_id, movie_id, and rating.
    """

    def __init__(
        self,
        send_rate,
        filename,
        cache_size=DEFAULT_STATE_CACHE_SIZE,
    ):
        schema = Schema(
            "key",
            {
                # generate key?
                "key": int,
                "user_id": int,
                "movie_id": int,
                "rating": int,
            },
        )

        super().__init__(schema, cache_size, num_worker_threads=1)
        print("Reading CSV", filename)
        df = pd.read_csv(filename)
        self.data = []
        for index, row in df.iterrows():
            self.data.append(row.to_dict())
        self.send_rate = send_rate
        self.ts = 0
        self.matrix = dict()

    def next(self):
        try:
            if self.ts < len(self.data):

                d = self.data[self.ts]
                t = time.time()
                
                ratings[movie_id - 1] = rating
                self.matrix[user_id] = ratings

                record = Record(
                    key=d["userId"],
                    user_id=d["userId"],
                    movie_id=d["movieId"],
                    rating=d["rating"],
                )
                self.ts += 1
                time.sleep(1 / self.send_rate)
                return [record]
            else:
                print("STOP ITERATION", self.ts)
        except Exception as e:
            print(e)
            raise StopIteration

@ray.remote
class Users(Operator):
    def __init__(
        self,
        cache_size=DEFAULT_STATE_CACHE_SIZE,
        lazy=False,
        num_worker_threads=1,
        num_features=10,
        alpha=.25,
        l=.1,
    ):

        schema = Schema(
            "key",
            {
                "key": int,
                "user_id": int,
                "features": np.array,
            },
        )
        super().__init__(schema, cache_size, lazy, num_worker_threads)
        self.rating_matrix = dict()
        self.user_matrix = dict()
        self.movie_matrix = dict()
        self.num_features = num_features
        self.alpha = alpha
        self.l = l

    def on_record(self, record: Record) -> Record:

        try:
            user_id = record.user_id
            movie_id = record.movie_id
            rating = record.rating
            
            if user_id in self.user_matrix:
                user_vector = self.user_matrix[user_id]
                ratings = self.rating_matrix[user_id]
            else:
                user_vector = np.random.rand(self.num_features)
                ratings = np.random.rand(NUM_MOVIES)

            if movie_id in self.movie_matrix:
                movie_vector = self.movie_matrix[movie_id]
            else:
                movie_vector = np.random.rand(self.num_features)

            ratings[movie_id-1] = rating
            self.rating_matrix[user_id] = ratings
            
            # recompute features
            sub_result = rating - np.dot(np.transpose(user_vector), movie_vector)
            new_user_vector = alpha * sub_result * movie_vector + self.l * user_vector

            self.matrix[user_id] = new_user_vector
            record = Record(
                    key=d["userId"],
                    user_id=d["userId"],
                    features=new_user_vector,
            )
            return [record]

        except Exception as e:
            print(e)

'''
# Currently unnecessary?
@ray.remote
class Movies(Operator):
    def __init__(
        self,
        cache_size=DEFAULT_STATE_CACHE_SIZE,
        lazy=False,
        num_worker_threads=1,
    ):

        schema = Schema(
            "key",
            {
                "key": int,
                "user_id": int,
                "movie_id": int,
                "rating": int,
            },
        )
        super().__init__(schema, cache_size, lazy, num_worker_threads)

    def on_record(self, record: Record) -> None:
        # Currently, not updating the movies table (only the user)
        return None
'''

def from_file(send_rate: int, f: str):
    return Table([], RatingSource, send_rate, f)

def create_doc_pipeline(args):
    ralf_conn = Ralf(
        metric_dir=os.path.join(args.exp_dir, args.exp), log_wandb=True, exp_id=args.exp
    )

    # create pipeline
    source = from_file(args.send_rate, os.path.join(args.data_dir, args.file))
    user_vectors = source.map(UserOperator, args, num_replicas=8).as_queryable("user_vectors")
    # deploy
    ralf_conn.deploy(source, "source")

    return ralf_conn
