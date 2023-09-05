from pyflink.table import EnvironmentSettings, TableEnvironment
from absl import app, flags
import pandas as pd
import time
import threading
import sys
from io import StringIO


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL

from collections import defaultdict
from pyflink.common import Row
from pyflink.table.udf import udaf
from pyflink.table import AggregateFunction, DataTypes, TableEnvironment, EnvironmentSettings, TableSink
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.common.typeinfo import Types
from pyflink.table import ListView
from pyflink.table.udf import udtaf, TableAggregateFunction

# Import the necessary Kafka dependencies
from kafka import KafkaProducer

#SEASONALITY = 24*60
SEASONALITY = 24*60
WINDOW_SIZE = SEASONALITY*3 

FLAGS = flags.FLAGS
flags.DEFINE_integer("slide", default=WINDOW_SIZE, required=False, help="Slide size")


class Top2(TableAggregateFunction):
     
    def __init__(self, slide): 
        self.slide_size = slide
        self.window = defaultdict(list)
        self.window_size = WINDOW_SIZE

    def emit_value(self, accumulator):
        for key in self.window.keys(): 
            window = self.window[key]
            if len(window) >= self.window_size: 
                # TODO: insert time consuming function 
                window = window[:WINDOW_SIZE]
                values = [row[1] for row in window]
                times = [row[2] for row in window]

                df = pd.DataFrame({"values": values, "ts": times})
                df = df.set_index("ts")
                data = df.values.flatten() #.to_numpy()

                model = STLForecast(
                    data, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=SEASONALITY
                ).fit()
                self.window[key] = self.window[key][self.slide_size:]
                yield Row(max(times))
            #else: 
            #    print("wrong size", len(window)) #%, self.keys, self.values)

    def create_accumulator(self):
        # (key, cpu, ts)
        return Row(None)

    def accumulate(self, accumulator, row):
        #if accumulator[0] is not None:
        #    assert accumulator[0] == row[0], f"Key mismatch {accumulator[0]}, {row[0]}"
        #accumulator[0] = row[0]
        self.window[row[0]].append(row)
        #print("row", row, accumulator, accumulator[0], row[0], len(self.window[row[0]]))

    def get_accumulator_type(self):
        return 'ROW<k BIGINT>'

    def get_result_type(self):
        return 'ROW<ts BIGINT>'

def log_processing(argv):

    print("Created thread")

    kafka_jar = "/home/ubuntu/experiments/flink-1.17.1/lib/flink-sql-connector-kafka-1.17.1.jar"

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(8)
    env.add_jars(f"file://{kafka_jar}")
    t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    t_env.get_config().set("pipeline.jars", f"file://{kafka_jar}")

    print("craeted env")

    #env_settings = EnvironmentSettings.in_streaming_mode()

    #t_env = TableEnvironment.create(env_settings)
    #t_env.get_config().set("parallelism.default", "1")
    #t_env.get_config().set("pipeline.jars", f"file://{kafka_jar}")

    #env = StreamExecutionEnvironment.get_execution_environment()
    #env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
    #kafka_jar = "/home/ubuntu/experiments/flink-1.17.1/lib/flink-sql-connector-kafka-1.17.1.jar"
    #env.add_jars(f"file://{kafka_jar}")
    #t_env = StreamTableEnvironment.create(env)
    #t_env.get_config().set("pipeline.jars", "file:///my/jar/path/connector.jar;file:///my/jar/path/json.jar")
    
    source_ddl = """
            CREATE TABLE source_table(
                key VARCHAR,
                cpu DOUBLE,
                ts BIGINT
            ) WITH (
              'connector' = 'kafka',
              'topic' = 'records',
              'properties.bootstrap.servers' = 'localhost:9092',
              'properties.group.id' = 'test_3',
              'scan.startup.mode' = 'earliest-offset',
              'properties.auto.offset.reset' = 'earliest',
              'format' = 'json'
            )
            """


    t_env.execute_sql(source_ddl)

    table = t_env.from_path("source_table")
    table.print_schema()
    weighted_avg = udtaf(Top2(FLAGS.slide))
    result = table.group_by(col('key')).flat_aggregate(weighted_avg).select(col('*')).execute().print()

if __name__ == '__main__':
    app.run(log_processing)
