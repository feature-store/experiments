import numpy as np
import pandas as pd
import ray
from ralf.operator import Operator, DEFAULT_STATE_CACHE_SIZE
from ralf.operators.source import Source
from ralf.state import Record, Schema
from ralf.core import Ralf
from ralf.table import Table
import argparse
import os
import time
import csv

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
                "key": str,
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

    def next(self):
        try:
            if self.ts < len(self.data):
                d = self.data[self.ts]
                t = time.time()

                record = Record(
                    key=str(d["userId"]),
                    user_id=int(d["userId"]),
                    movie_id=int(d["movieId"]),
                    rating=int(d["rating"]),
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
class UserOperator(Operator):
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
                "key": str,
                "user_id": int,
                "movie_id": int,
                "user_vector": np.array,
                "movie_vector": np.array,
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
            key = record.key
            user_id = record.user_id
            movie_id = record.movie_id
            rating = record.rating
            
            if user_id in self.user_matrix:
                user_vector = self.user_matrix[user_id]
                ratings = self.rating_matrix[user_id]
            else:
                user_vector = np.random.randint(100, size=self.num_features)
                ratings = np.random.randint(1, size=NUM_MOVIES)
            if movie_id in self.movie_matrix:
                movie_vector = self.movie_matrix[movie_id]
            else:
                movie_vector = np.random.randint(100, size=self.num_features)
                '''
                with open("movie_vectors.csv", "a") as f:
                    csvwriter = csv.writer(f) 
                    csvwriter.writerow([str(movie_id), str(movie_vector)])
                '''
            ratings[movie_id-1] = rating
            self.rating_matrix[user_id] = ratings
            self.movie_matrix[movie_id] = movie_vector
            # recompute features
            print(self.movie_matrix)
            sub_result = rating - np.dot(np.transpose(user_vector), movie_vector)
            new_user_vector = self.alpha * sub_result * movie_vector + self.l * user_vector
            self.user_matrix[user_id] = new_user_vector
            record = Record(
                    key=key,
                    user_id=user_id,
                    movie_id=movie_id,
                    user_vector=new_user_vector,
                    movie_vector=movie_vector,
            )
            print("Sending record from user", record.movie_id)
            return [record]

        except Exception as e:
            print(e)


# Currently unnecessary?
@ray.remote
class MovieOperator(Operator):
    def __init__(
        self,
        cache_size=DEFAULT_STATE_CACHE_SIZE,
        lazy=False,
        num_worker_threads=1,
    ):

        schema = Schema(
            "key",
            {
                "key": str,
                "movie_id": int,
                "movie_vector": np.array,
            },
        )
        super().__init__(schema, cache_size, lazy, num_worker_threads)

    def on_record(self, record: Record) -> Record:
        # Currently, not updating the movies table (only the user)
        print("Hit record", record)
        new_record = Record(
            key=str(record.movie_id),
            movie_id=record.movie_id,
            movie_vector=record.movie_vector,
        )
        return [new_record]

def from_file(send_rate: int, f: str):
    return Table([], RatingSource, send_rate, f)

def create_doc_pipeline(args):
    ralf_conn = Ralf(
        metric_dir=os.path.join(args.exp_dir, args.exp), log_wandb=False, exp_id=args.exp
    )

    # create pipeline
    source = from_file(args.send_rate, os.path.join(args.data_dir, args.file))
    user_vectors = source.map(UserOperator, args, num_replicas=1).as_queryable("user_vectors")
    #movies = source.join(user_vectors, MovieOperator).as_queryable("movie_vectors")
    #movie_vectors = user_vectors.map(MovieOperator).as_queryable("movie_vectors")
    # deploy
    ralf_conn.deploy(source, "source")

    return ralf_conn


def main():

    parser = argparse.ArgumentParser(description="Specify experiment config")
    parser.add_argument("--send-rate", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=10)

    # Experiment related
    # TODO: add wikipedia dataset
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/amitnarang/Downloads/ml-latest-small",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="/Users/amitnarang/ralf-experiments",
    )

    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--exp", type=str)  # experiment id
    args = parser.parse_args()
    # create experiment directory
    ex_id = args.exp
    ex_dir = os.path.join(args.exp_dir, ex_id)
    os.mkdir(ex_dir)

    # create stl pipeline
    ralf_conn = create_doc_pipeline(args)
    ralf_conn.run()

    # snapshot stats
    run_duration = 120
    snapshot_interval = 10
    start = time.time()
    while time.time() - start < run_duration:
        snapshot_time = ralf_conn.snapshot()
        remaining_time = snapshot_interval - snapshot_time
        if remaining_time < 0:
            print(
                f"snapshot interval is {snapshot_interval} but it took {snapshot_time} to perform it!"
            )
            time.sleep(0)
        else:
            print("writing snapshot", snapshot_time)
            time.sleep(remaining_time)


if __name__ == "__main__":
    main()
