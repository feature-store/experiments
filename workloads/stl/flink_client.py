import pickle
from workloads.util import read_config, use_dataset, log_dataset
import sys
from workloads.util import log_results
import pandas as pd
import json
import subprocess
from absl import app, flags
import threading
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import sqlite3
import time
import multiprocessing
from kafka import KafkaConsumer, KafkaProducer
from multiprocessing import Queue

# Function for Process 1

FLAGS = flags.FLAGS
flags.DEFINE_integer("slide", default=1, required=False, help="Slide size")
flags.DEFINE_integer("keys", default=1, required=False, help="Slide size")


def send_to_kafka(queue, slide):
    #producer = KafkaProducer(security_protocol="SSL", bootstrap_servers=os.environ.get('KAFKA_HOST', 'localhost:9092'))
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    azure_database = "/home/ubuntu/cleaned_sqlite_3_days_min_ts.db"
    conn = sqlite3.connect(azure_database)
    conn = sqlite3.connect(azure_database)

    dataset_dir = use_dataset("azure")
    #key_file = f"{dataset_dir}/azure_{FLAGS.keys}/keys_{FLAGS.keys}.json"
    #print(key_file)
    #keys = list(json.load(open(key_file, "r")).values())
    #keys_selection_clause = ",".join(map(str, keys))
    #print(keys_selection_clause)

    st = time.time()
    #cursor = conn.execute(
    #    "SELECT timestamp, int_id, avg_cpu FROM readings WHERE "
    #    f"int_id in ({keys_selection_clause})"
    #)

    cursor = conn.execute(
        f"SELECT timestamp, int_id, avg_cpu FROM readings WHERE int_id IN ("
        f"SELECT int_id FROM ("
        f"SELECT int_id, COUNT(*) AS int_id_count FROM readings "
        f"GROUP BY int_id ORDER BY int_id_count DESC LIMIT {FLAGS.keys}"
        f") AS top_int_ids)"
    )
    print("Query completed", time.time() - st)

    data_buffer = defaultdict(list)
    for ts, int_id, avg_cpu in cursor:
        data_buffer[int(ts)].append((int_id, avg_cpu, ts))


    results = []
    sent_data = []
    for data in data_buffer.values(): 
        sent_data += [{"key": int_id, "cpu": avg_cpu, "ts": t} for (int_id, avg_cpu, t) in data]
    pd.DataFrame(sent_data).to_csv(f"sent_data_{slide}_{FLAGS.keys}.csv")

    # run experiment 
    for ts, data in tqdm(sorted(data_buffer.items())):  

        # timestamp is every 5 minutes in Azure

        for r in data: 
            assert r[2] == ts, f"Incorrect timestamp {data}"

        with ThreadPoolExecutor(max_workers=1) as executor:
            
            def send_data_helper(records):
                #for r in records:
                #    print(r)
                for (int_id, avg_cpu, t) in records:
                    try:
                        res = producer.send(
                            "records",
                            value = {"key": int_id, "cpu": avg_cpu, "ts": t}
                        )
                    except Exception as e:
                        print("ERROR", e, repr(e))
                        import traceback 
                        traceback.print_exc()
                        raise e
            #print(f"Sending {len(data)} records for {FLAGS.keys} keys")
            executor.submit(send_data_helper, data)


            wait_time = 0.3 # send at 1000x original speed
            st = time.time()

            # check queue for processed data
            while not queue.empty(): 
                item = queue.get()

                # calculate staleness
                r = {"key": item["key"], "output_ts": item["ts"], "curr_ts": ts, "staleness": ts - item["ts"]}
                print(r)
                results.append(r)

            # sleep 
            time.sleep(wait_time - (time.time() - st))

        # TODO: pop from queue and determine max size

    df = pd.DataFrame(results)
    df.to_csv(f"results_{slide}_{FLAGS.keys}.csv")
    print("finished", f"results_{slide}.csv")
    producer.close()


    #for _, row in window_data.iterrows():
    #    record = {
    #        "key": row["key"],
    #        "value": row["value"],
    #        "timestamp": row["timestamp"]
    #    }
    #    producer.send("records", value=record)


# Function for Process 2
def listen_results(queue, slide):
    # Start the Flink program as a subprocess
    flink_process = subprocess.Popen(
        ["python", "workloads/stl/test_flink.py", "--slide", f"{slide}"],  # Replace with the actual path to your Flink program
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True  # Ensure text mode for stdout
    )
    print(subprocess.STDOUT)
    # Process the output in real-time
    for line in flink_process.stdout:
        # Extract key and ts information from the line (adjust parsing as needed)
        parts = line.strip().split('|')
        if len(parts) == 5 and parts[1].strip() == '+I':
            key = int(parts[2].strip())
            ts = int(parts[3].strip())
            record = {"key": key, "ts": ts}

            # Send the record to Kafka
            queue.put(record)
            #kafka_producer.send("output", value=json.dumps(record).encode("utf-8"))

    # Wait for the Flink program to complete
    flink_process.wait()


    #def deserialize(data): 
    #    return json.loads(data.decode("utf-8"))

    #consumer = KafkaConsumer(
    #    "output",
    #    bootstrap_servers="localhost:9092",
    #    value_deserializer=deserialize,
    #)
    #for message in consumer:
    #    queue.put(message.value)

def main(argv):
    output_queue = Queue()

    p1 = multiprocessing.Process(target=send_to_kafka, args=(output_queue, FLAGS.slide))
    p2 = multiprocessing.Process(target=listen_results, args=(output_queue, FLAGS.slide))

    p1.start()
    p2.start()

    p1.join() # only wait for p1 
    p2.kill()

if __name__ == "__main__":
    app.run(main)

