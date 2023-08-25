import numpy as np
from statsmodels.tsa.forecasting.stl import STLForecast
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from kafka import KafkaProducer
import json
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer

from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.window import Tumble
from pyflink.table.udf import udf

from pyflink.common import Row
from pyflink.table import AggregateFunction, DataTypes, TableEnvironment, EnvironmentSettings
from pyflink.table.expressions import call
from pyflink.table.udf import udaf
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble

import json

class WeightedAvg(AggregateFunction):

    def create_accumulator(self):
        # Row(sum, count)
        return Row(0, 0)

    def get_value(self, accumulator):
        if accumulator[1] == 0:
            return None
        else:
            return accumulator[0] / accumulator[1]

    def accumulate(self, accumulator, value, weight):
        accumulator[0] += value * weight
        accumulator[1] += weight
    
    def retract(self, accumulator, value, weight):
        accumulator[0] -= value * weight
        accumulator[1] -= weight
        
    def get_result_type(self):
        return DataTypes.INT()
        
    def get_accumulator_type(self):
        return DataTypes.ARRAY(DataTypes.FLOAT()) 



#env = StreamExecutionEnvironment.get_execution_environment()
env = StreamExecutionEnvironment.get_execution_environment()
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
kafka_jar = "/home/ubuntu/experiments/flink-1.17.1/lib/flink-sql-connector-kafka-1.17.1.jar"
env.add_jars(f"file://{kafka_jar}")
t_env = StreamTableEnvironment.create(env)

# Define the processing function
@udf(input_types=[DataTypes.ARRAY(DataTypes.FLOAT())], result_type=DataTypes.INT())
def STL(window):
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


src_ddl = """
    CREATE TABLE sales_usd (
        key VARCHAR,
        cpu DOUBLE,
        ts BIGINT,
        proctime AS PROCTIME()
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'records',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
"""

t_env.execute_sql(src_ddl)
t = t_env.from_path('sales_usd')

weighted_avg = udaf(WeightedAvg())
#t = t_env.from_elements([(1, 2, "Lee"),
#                             (3, 4, "Jay"),
#                             (5, 6, "Jay"),
#                             (7, 8, "Lee")]).alias("ts", "value", "cpu")

# use the general Python aggregate function in GroupBy Window Aggregation
tumble_window = Tumble.over(lit(1).hours) \
            .on(col("proctime")) \
            .alias("w")

result = t.window(tumble_window) \
        .group_by(col('w'), col('key')) \
        .select(col('w').start, col('w').end, weighted_avg(col('cpu'), col('cpu'))) \
        .execute()
result.print()


### Define Kafka source properties
##kafka_props = {
##    "bootstrap.servers": "localhost:9092",
##    "group.id": "flink-group"
##}
##
##source = KafkaSource.builder() \
##    .set_bootstrap_servers(kafka_props["bootstrap.servers"]) \
##    .set_topics(input_stream) \
##    .build()
##
##tenv.from_source(source, WatermarkStrategy.no_watermarks(), "Kafka Source")
##
##
### Define the Kafka source table
##source_table = t_env.from_kafka_source(
##    "records",
##    kafka_props,
##    schema
##)
#
## Define the window operation
#windowed_table = source_table.window(
#    Tumble.over("672.seconds").on("timestamp").alias("window")
#)
#
## Apply the STL UDF to each window
#result_table = windowed_table.group_by("key, window") \
#                            .select("key, STL(value) as result")
#
## Define Kafka sink properties
#kafka_sink_props = {
#    "bootstrap.servers": "localhost:9092",
#    "topic": "output"
#}
#
## Write the results to the Kafka sink
#result_table.execute_insert("output", sink_properties=kafka_sink_props)
#
#env.execute("Flink Kafka Windowing Job")
#
