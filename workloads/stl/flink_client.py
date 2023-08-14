import pickle
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

def send_to_kafka(queue):
    #producer = KafkaProducer(security_protocol="SSL", bootstrap_servers=os.environ.get('KAFKA_HOST', 'localhost:9092'))
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: pickle.dumps(v)
    )

    azure_database = "/home/ubuntu/cleaned_sqlite_3_days_min_ts.db"
    conn = sqlite3.connect(azure_database)
    conn = sqlite3.connect(azure_database)

    keys = list(json.load(open("/home/ubuntu/keys.json", "r")).values())
    keys_selection_clause = ",".join(map(str, keys))
    print(keys_selection_clause)
    st = time.time()
    cursor = conn.execute(
        "SELECT timestamp, int_id, avg_cpu FROM readings WHERE "
        f"int_id in ({keys_selection_clause})"
    )
    print("Query completed", time.time() - st)

    data_buffer = defaultdict(list)
    for ts, int_id, avg_cpu in cursor:
        data_buffer[ts].append((int_id, avg_cpu, ts))


    for ts, data in tqdm(data_buffer.items()): 

        with ThreadPoolExecutor(max_workers=1) as executor:
            
            def send_data_helper(records):
                print(f"Sending {len(records)} records")
                #for r in records:
                #    print(r)
                for (int_id, avg_cpu, ts) in records:
                    try:
                        res = producer.send(
                            "records",
                            value = {"key": int_id, "value": avg_cpu, "ts": ts}
                        )
                    except Exception as e:
                        print("ERROR", e, repr(e))
                        import traceback 
                        traceback.print_exc()
                        raise e
            executor.submit(send_data_helper, data)


            wait_time = 1
            st = time.time()

            # check queue for processed data
            max_processed_ts = 0
            while not queue.empty(): 
                item = queue.get()
                if item["ts"] > max_processed_ts: 
                    max_processed_ts = item["ts"]

            # calculate staleness
            print("Staleness", ts - max_processed_ts)

            # sleep 
            time.sleep(wait_time - (time.time() - st))

        # TODO: pop from queue and determine max size


    #for _, row in window_data.iterrows():
    #    record = {
    #        "key": row["key"],
    #        "value": row["value"],
    #        "timestamp": row["timestamp"]
    #    }
    #    producer.send("records", value=record)

    producer.close()

# Function for Process 2
def listen_results(queue):
    consumer = KafkaConsumer(
        "records",
        bootstrap_servers="localhost:9092",
        value_deserializer=pickle.loads
    )
    for message in consumer:
        queue.put(message.value)

if __name__ == "__main__":
    output_queue = Queue()

    p1 = multiprocessing.Process(target=send_to_kafka, args=(output_queue,))
    p2 = multiprocessing.Process(target=listen_results, args=(output_queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
