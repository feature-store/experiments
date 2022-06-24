import boto3
import json
import itertools
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import sqlite3
import os
import json
import time
from multiprocessing import Pool

def read_data(keys, stream_name): 
    global stream_conn

    boto3.setup_default_session(profile_name='sarah')
    client = boto3.client('kinesis', region_name="us-west-2")

if __name__ == "__main__":

    #num_shards = 250 # sends num_shards*4MB/s 
    num_threads = 32
    num_keys = 10000
    #stream_name = "azure-stream"

    # read terraform config
    resource_config_file = "/home/ubuntu/workspace/ralf/terraform/config.json"
    resource_config = json.load(open(resource_config_file))
    s3_bucket = resource_config["s3_bucket"]["value"]
    s3_arn = resource_config["s3_arn"]["value"]
    stream_name = resource_config["kinesis_name"]["value"]
    stream_arn = resource_config["kinesis_arn"]["value"]
    num_shards = resource_config["kinesis_shard_count"]["value"]

    print("STREAM", stream_name)
    print(stream_arn)


    # list shards
    client = boto3.client('kinesis')
    response = client.list_shards(
        StreamName=stream_name,
        MaxResults=1000,
    )
    shards = response["Shards"]
    print(shards)

    print(stream_arn)
    response = client.list_stream_consumers(
        StreamARN=stream_arn,
    )
    print(response)
    for consumer in response["Consumers"]:
        consumer_arn = consumer["ConsumerARN"]
        consumer_name = consumer["ConsumerName"]
        response = client.deregister_stream_consumer(
            StreamARN=stream_arn,
            ConsumerName=consumer_name,
            ConsumerARN=consumer_arn
        )
        status = "DELETING"
        while status == "DELETING": 
            try:
                response = client.describe_stream_consumer(
                    StreamARN=stream_arn,
                    ConsumerName=consumer_name,
                    ConsumerARN=consumer_arn
                )
                status = response["ConsumerDescription"]["ConsumerStatus"]
            except Exception as e:
                print(e)


        print(f"Deregistered {consumer_name}")

    num_replicas = 5

    for i in range(num_replicas): 
            
        try:
            response = client.register_stream_consumer(
                StreamARN=stream_arn,
                ConsumerName=f"source_replica_{i}"
            )
            print(response)
            consumer_name = response["Consumer"]["ConsumerName"] 
            consumer_arn = response["Consumer"]["ConsumerARN"]
        except Exception as e:
            print(e)

        status = None
        while status != "ACTIVE": 
            response = client.describe_stream_consumer(
                StreamARN=stream_arn,
                ConsumerName=consumer_name,
                ConsumerARN=consumer_arn
            )
            status = response["ConsumerDescription"]["ConsumerStatus"]

        print("Finished creating stream consumer")
        response = client.subscribe_to_shard(
            ConsumerARN=consumer_arn,
            ShardId=shards[0]["ShardId"],
            StartingPosition={
                'Type': 'AT_SEQUENCE_NUMBER',
                'SequenceNumber': shards[0]['SequenceNumberRange']['StartingSequenceNumber']
            }
        )

        for event in response["EventStream"]:
            print(event)




    # get records



