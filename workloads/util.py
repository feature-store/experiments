import os
# TODO: Common source operator to ingest events.csv

def upload_dataset(directory, name): 

    s3_folder = f"s3://feature-store-datasets/vldb/datasets/{name}"

    print(f"aws s3 sync {directory} {s3_folder}")
    os.system(f"aws s3 sync {directory} {s3_folder}")

def download_dataset(name, directory):
    s3_folder = f"s3://feature-store-datasets/vldb/datasets/{name}"
    os.system(f"aws s3 sync {s3_folder} {directory}/{name}")



# TODO: peridic snapshoting code
