import pickle
import time
import multiprocessing
from kafka import KafkaConsumer
from multiprocessing import Queue

# Function for Process 1

def send_to_kafka(window_data):
    producer = KafkaProducer(
        bootstrap_servers="your_kafka_brokers",
        value_serializer=lambda v: pickle.dumps(v)
    )

    for _, row in window_data.iterrows():
        record = {
            "key": row["key"],
            "value": row["value"],
            "timestamp": row["timestamp"]
        }
        producer.send("records", value=record)

    producer.close()

def process_records():
    with open("records.pkl", "rb") as f:
        records_list = pickle.load(f)

    df = pd.DataFrame(records_list)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Group by 1-hour tumbling windows
    grouped = df.groupby(pd.Grouper(key="timestamp", freq="1H"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _, window_data in grouped:
            executor.submit(send_to_kafka, window_data)

# Function for Process 2
def listen_results(queue):
    consumer = KafkaConsumer(
        "records",
        bootstrap_servers="your_kafka_brokers",
        value_deserializer=lambda x: x.decode("utf-8")
    )

    for message in consumer:
        data = json.loads(message.value)
        timestamp = data["timestamp"]
        queue.put(timestamp)

if __name__ == "__main__":
    output_queue = Queue()

    p1 = multiprocessing.Process(target=process_1, args=(output_queue,))
    p2 = multiprocessing.Process(target=process_2, args=(output_queue,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
