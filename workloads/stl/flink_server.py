import numpy as np
from statsmodels.tsa.forecasting.stl import STLForecast
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.common.serialization import JsonRowSerializationSchema
from kafka import KafkaProducer
import json
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
import json

env = StreamExecutionEnvironment.get_execution_environment()

# Define the processing function
def stl(window):
    model = STLForecast(
        np.array([record["value"] for record in window]),
        ARIMA,
        model_kwargs=dict(order=(1, 1, 0), trend="t"),
        period=12 * 24  # 5 min timestamp interval, period of one day
    ).fit()
    return max([record["timestamp"] for record in window])

# Define input and output streams
input_stream = "records"
output_stream = "output"

# Define slide size and window size
slide_size = 1
window_size = 672

# Define Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Define output sink
def kafka_sink(value):
    producer.send(output_stream, value=value)

# Define Kafka source properties
kafka_source_props = {
    "bootstrap.servers": "localhost:9092",
}
 
# Create a FlinkKafkaConsumer instance
kafka_source = FlinkKafkaConsumer(
    "records",
    SimpleStringSchema(),
    properties=kafka_source_props
)

# Add the Kafka source to the execution environment
records_stream = env.add_source(kafka_source)

# Print the data stream
records_stream.print()

## Define the Kafka source
#records_stream = env.add_source(
#    "kafka",
#    "org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer",
#    "records",
#    json.dumps(kafka_source_props),
#    value_deserializer="org.apache.kafka.common.serialization.StringDeserializer"
#)

## Read records from Kinesis stream
#records_stream = env.from_collection(records_list)

# Group records by "key"
grouped_stream = records_stream.key_by(lambda record: record["key"])

# Compute sliding window and apply processing function
result_stream = grouped_stream.window(Slide.over(window_size * slide_size).every(slide_size)) \
    .apply(lambda key, window, records: stl([record["value"] for record in records]))

# Add the Kafka sink to the result stream
result_stream.add_sink(kafka_sink)

# Execute the program
env.execute("Processing Records and Writing to Kafka")
