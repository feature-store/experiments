import pandas as pd
import json
import pickle
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import dask.dataframe as dd

import sys 
sys.path.insert(1, "../")
from workloads.util import use_results, use_dataset, read_config, log_dataset

data_dir = use_dataset("ml-25m")
#data_dir = use_dataset("ml-latest-small")
ratings_path = f"{data_dir}/ratings.csv"
tags_path = f"{data_dir}/tags.csv" 
movies_path = f"{data_dir}/movies.csv"

# read files
tags = dd.read_csv(tags_path)
tags.columns = ['user_id', 'movie_id', 'tag', 'timestamp']
ratings = pd.read_csv(ratings_path)
ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
movies = dd.read_csv(movies_path)
movies.columns = ['movie_id', 'title', 'genres']


print("read CV")
users = list(set(ratings.user_id.tolist()))
print("Num users", len(users))
ratings = dd.from_pandas(ratings, npartitions=1000)
user_start_ts = {user: ratings[ratings["user_id"] == user].timestamp.min() for user in users}
ratings.timestamp = ratings.apply(lambda x: x["timestamp"] - user_start_ts[x["user_id"]], axis=1)
ratings.timestamp = ratings.timestamp.apply(lambda x: int(x/100))
ratings.compute()
cutoff = ratings.timestamp.median()

print("Saving train/test CSV")
train_df = ratings[ratings["timestamp"] < cutoff].sort_values("timestamp")
test_df = ratings[ratings["timestamp"] > cutoff].sort_values("timestamp")
# filter out test set movies not in train
train_movies = train_df.movie_id.tolist()
test_df = test_df[test_df.movie_id.apply(lambda x: x in train_movies)]

train_df.compute()
test_df.compute()
train_df.to_csv(f"{data_dir}/train.csv")
test_df.to_csv(f"{data_dir}/test.csv")


combined_df = train_df.join(movies, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
pivot_table = combined_df.pivot_table(columns=['movie_id'], index=['user_id'], values='rating')
A = pivot_table.fillna(0).values
pivot_table.compute()
A.compute()
movie_to_index = {int(pivot_table.columns[i]): int(i) for i in range(len(pivot_table.columns))}
user_to_index = {int(pivot_table.index[i]): int(i) for i in range(len(pivot_table.index))}
R = A>0.5; R[R == True] = 1; R[R == False] = 0; R = R.astype(np.float64, copy=False)

print("saving user/movie index maps")
print(user_to_index)
json.dump(movie_to_index, open(f"{data_dir}/movie_to_index.json", "w"))
json.dump(user_to_index, open(f"{data_dir}/user_to_index.json", "w"))

print("Saving A/R CSV")
pickle.dump(A, open(f"{data_dir}/A.pkl", "wb"))
pickle.dump(R, open(f"{data_dir}/R.pkl", "wb"))

# Train ALS Model 
print("Training ALS model")
spark = SparkSession.builder.master('local').appName('als').getOrCreate()
spark_als_df = spark.createDataFrame(train_df) 
als = ALS(
         userCol="user_id", 
         itemCol="movie_id",
         ratingCol="rating", 
         nonnegative = True, 
         implicitPrefs = False,
         coldStartStrategy="drop",
         rank=150,
         maxIter=10,
         regParam=.1
)
model=als.fit(spark_als_df)

user_matrix = model.userFactors.toPandas().sort_values("id").set_index("id").features.to_list()
movie_matrix = model.itemFactors.toPandas().sort_values("id").set_index("id").features.to_list()
user_matrix = np.array(user_matrix)
movie_matrix = np.array(movie_matrix)

pickle.dump(user_matrix, open(f"{data_dir}/user_matrix.pkl", "wb"))
pickle.dump(movie_matrix, open(f"{data_dir}/movie_matrix.pkl", "wb"))

