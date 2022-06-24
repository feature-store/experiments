import boto3
from tqdm import tqdm 
import pandas as pd
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

def read_data_sqlite(keys, stream_name): 
    global stream_conn

    boto3.setup_default_session(profile_name='sarah')
    client = boto3.client('kinesis', region_name="us-west-2")

    keys_selection_clause = ",".join(map(str, keys))
    # read DB data
    print("Running query")
    st = time.time()
    azure_database = "/home/ubuntu/cleaned_sqlite_3_days_min_ts.db"
    conn = sqlite3.connect(azure_database)
    conn = sqlite3.connect(azure_database)
    cursor = conn.execute(
        "SELECT timestamp, int_id, avg_cpu FROM readings WHERE "
        f"int_id in ({keys_selection_clause})"
    )
    print("Query completed", time.time() - st)

    data_buffer = defaultdict(list)
    for ts, int_id, avg_cpu in cursor:
        data_buffer[ts].append((int_id, avg_cpu, ts))


    for ts, data in tqdm(data_buffer.items()): 

        with ThreadPoolExecutor(max_workers=32) as executor:
            
            def send_data_helper(records):
                response = client.put_records(
                    StreamName=stream_name,
                    Records = [
                        {
                            "Data": json.dumps({"key": int_id, "value": avg_cpu, "ts": ts}).encode('utf-8'),
                            "PartitionKey": str(int_id)
                        }
                        for (int_id, avg_cpu, ts) in records
                    ]
                    #ExplicitHashKey='string', # TODO: use ralf hash function
                )
            executor.submit(send_data_helper, data)

def encode_record(record): 
    return json.dumps(record).encode('utf-8')

def send_ordered_data_process(keys, groups, last_seq_nos):

    client = boto3.client('kinesis')
    def send_single_key(key, group, seq_no): 
        # send data for a single key
        for i, row in group.iterrows(): 
            try:
                row = row.to_json()
                record_bytes = encode_record(row)
                # send row 
                response = client.put_record(
                    StreamName=stream_name, 
                    Data=record_bytes,
                    PartitionKey=key,
                    SequenceNumberForOrdering='0' if seq_no is None else seq_no
                )
                shard_id = response["ShardId"]
                seq_no = response["SequenceNumber"]
            except Exception as e:
                print(e)
        return seq_no  

    futures = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        for key, group, seq_no in zip(keys, groups, last_seq_nos): 
            futures.append(executor.submit(send_single_key, key, group, seq_no))
    last_seq_no = []
    for key, f in tqdm(zip(keys, futures)): 
        last_seq_no.append([key, f.result()])
    #print(last_seq_no)
    return last_seq_no


def send_ordered_data(data_dir, stream_name, max_index = 195): 

    last_seq_no_all = defaultdict(lambda: "0") # map each key to last observed sequence no 
    print(last_seq_no_all["rere"])
    for i in range(1, max_index + 1, 1): 
        f = f"{data_dir}/vm_cpu_readings-file-{i}-of-195.csv"
        df = pd.read_csv(f, header=None)
        df.columns=["timestamp", "key", "min_cpu", "max_cpu", "avg_cpu"]
        df = df.groupby("key")

        groups = [group for key, group in  df]
        keys = [key for key, group in df]
        seq_nos = [last_seq_no_all[key] for key, group in df]
        num_threads = 32
        shard_size = int(len(groups) / num_threads)
        shards = [
            (
                keys[i:min(i+shard_size, len(groups))],
                groups[i:min(i+shard_size, len(groups))],
                seq_nos[i:min(i+shard_size, len(groups))] 
            ) 
            for i in range(num_threads)
        ]
        print(f"{f}: Created {len(shards)} shards with length {shard_size}") 
        p = Pool(num_threads)
        last_seq_no = p.starmap(send_ordered_data_process, shards)
        p.close()

        # collect last seq no
        for res in last_seq_no:
            for res_pair in res:
                last_seq_no_all[res_pair[0]] = res_pair[1]

        # write last seq no
        open(f"last_seq_no_{i}.json", "w").write(json.dumps(last_seq_no_all))

    print("Done!")
    
        ## encode data in parallel 
        #p = Pool(32)
        #encoded_data = p.map(encode_record, data)
        #p.close()
        #print("Finished encoding data")

        #def send_data_helper(record_bytes, key):
            #seq_no = key_to_last_seq_no[key]
            #response = client.put_record(
                #StreamName=stream_name, 
                #Data=record_bytes,
                #PartitionKey=key,
                #SequenceNumberForOrdering='' if seq_no is None else seq_no
            #)
            #shard_id = response["ShardId"]
            #seq_no = response["SequenceNumber"]
            #key_to_last_seq_no[key] = seq_no
            #if shard_id not in start_seq_no: 
                #start_seq_no[shard_id] = start_seq_no


        #with ThreadPoolExecutor(max_workers=128) as executor:
            ## send batch of records together ? 
            #for record_bytes, record in tqdm(zip(encoded_data, data)): 
                #executor.submit(send_data_helper, record_bytes, record)
            ## wait for all tasks to finish
            #executor.shutdown(wait=True, cancel_futures=False)
            #print(len(list(key_to_last_seq_no.keys())))





def wait_status_complete(stream_name, status):
    stream_status = None
    stream_arn = None
    while stream_status is None or stream_status == status:
        try:
            stream_desc = client.describe_stream(
                StreamName=stream_name,
                Limit=num_shards,
            )["StreamDescription"]
            stream_status = stream_desc["StreamStatus"]
            stream_arn = stream_desc["StreamARN"]
            print(f"Stream {stream_name} status: {stream_status}")
            time.sleep(1)
        except Exception as e:
            print(e)
            stream_status = "DELETED"
            break

    return stream_status, stream_arn


def read_keys(num_keys): 
    print(f"Reading keys {num_keys}")
    cache_file = f"query_cache_{num_keys}.json"
    azure_database = "/home/ubuntu/cleaned_sqlite_3_days_min_ts.db"
    conn = sqlite3.connect(azure_database)
    conn = sqlite3.connect(azure_database)

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
            keys = cache["keys"]
            all_timestamps = cache["all_timestamps"]
            return keys, all_timestamps
    else:
        print(f"Genering query cache for num_keys={num_keys}")
        keys = list(
            itertools.chain.from_iterable(
                conn.execute(
                    f"SELECT int_id FROM readings GROUP BY int_id LIMIT {num_keys};"
                )
            )
        )
        all_timestamps = list(
            itertools.chain.from_iterable(
                conn.execute(
                    "SELECT timestamp FROM readings GROUP BY timestamp ORDER BY timestamp"
                ).fetchall()
            )
        )
        with open(cache_file, "w") as f:
            json.dump({"keys": keys, "all_timestamps": all_timestamps}, f)
        return keys, all_timestamps

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
    print("shards", num_shards)
    print(stream_arn)

    data_dir = "/home/ubuntu/workspace/experiments/datasets/azure"

    send_ordered_data(data_dir, stream_name, max_index = 2)

#    client = boto3.client('kinesis')
    ## delete stream if exists
    #try: 
        #response = client.delete_stream(
            #StreamName=stream_name,
            #EnforceConsumerDeletion=True
        #)
        #stream_status, stream_arn = wait_status_complete(stream_name, "DELETING")
        #print(f"Finished deleting {stream_name}")
    #except Exception as e:
        #print(e)

    ## create stream 
    #stream_status = None
    #stream_arn = None
    #try:
        #response = client.create_stream(
            #StreamName=stream_name,
            #ShardCount=num_shards,
            #StreamModeDetails={
                #'StreamMode': 'PROVISIONED'
            #}
        #)
        #response_id = response["ResponseMetadata"]["RequestId"]
    #except Exception as e:
        #print(e)

    ## wait for stream to become active
    #stream_status, stream_arn = wait_status_complete(stream_name, "CREATING")
    #print("Created stream")
    ##while stream_status != "ACTIVE": 
        ##stream_desc = client.describe_stream(
            ##StreamName=stream_name,
            ##Limit=num_shards,
        ##)["StreamDescription"]
        ##stream_status = stream_desc["StreamStatus"]
        ##stream_arn = stream_desc["StreamARN"]


    ## enable metrics 
    #response = client.enable_enhanced_monitoring(
        #StreamName=stream_name,
        #ShardLevelMetrics=['ALL'],
    #)

    ## create firehose delivery stream
    #client = boto3.client('firehose')
    #response = client.create_delivery_stream(
        #DeliveryStreamName=delivery_stream_name,
        #DeliveryStreamType='KinesisStreamAsSource',
        #KinesisStreamSourceConfiguration={
            #'KinesisStreamARN': stream_arn,
            #'RoleARN': stream_arn,
        #},
        #S3DestinationConfiguration={
            #'RoleARN': s3_arn,
            #'BucketARN': s3_arn,
            #'Prefix': "azure",
            #'BufferingHints': {
                #'SizeInMBs': 4,
                #'IntervalInSeconds': 60
            #},
            #'CompressionFormat': 'UNCOMPRESSED',
        #}
    #)
    #print(response)



    #print(stream_arn)
    #print(stream_status)

#    keys, timestamps = read_keys(num_keys)

    ## create key shards for each thread
    #shard_size = int(len(keys) / num_threads)
    #key_shards = [keys[i:min(i+shard_size, len(keys))] for i in range(num_threads)]

    ## send data for each timestamp
    #p = Pool()
    #p.starmap(read_data, zip(key_shards, [stream_name]*num_threads))
    #p.close()

    ##global data_buffer 
    #data_buffer = read_data(keys, stream_name)

    #p = Pool(num_threads) #, initializer=read_data)
    #for ts in timestamps:
    #    results = p.starmap(send_data, zip(key_shards, [stream_name]*num_threads))
    #    #results = p.map(send_data, [ts]*num_threads)
    #    p.close() # wait for finish
