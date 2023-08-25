from pyflink.table import EnvironmentSettings, TableEnvironment
from pyflink.common import Row
from pyflink.table.udf import udaf
from pyflink.table import AggregateFunction, DataTypes, TableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.common.typeinfo import Types
from pyflink.table import ListView
from pyflink.table.udf import udtaf, TableAggregateFunction

class Top2(TableAggregateFunction):

    def emit_value(self, accumulator):
        if len(self.values) == self.window_size: 
            print(accumulator[0], self.keys)
            max_ts = max(self.ts)
            self.values = []
            self.keys = []
            self.ts = []
            yield Row(max_ts)

    def create_accumulator(self):
        # (key, cpu, ts)
        self.values = []
        self.ts = []
        self.keys = []
        self.window_size = 672
        self.slide_size = 672
        return Row(None)

    def accumulate(self, accumulator, row):
        #print("row", row)
        if accumulator[0] is not None:
            assert accumulator[0] == row[0], f"Key mismatch {accumulator[0]}, {row[0]}"
        accumulator[0] = row[0]
        self.keys.append(row[0])
        self.values.append(row[1])
        self.ts.append(row[2])

    def get_accumulator_type(self):
        return 'ROW<k BIGINT>'

    def get_result_type(self):
        return 'ROW<ts BIGINT>'

class WindowMap(TableAggregateFunction):

    def emit_value(self, accumulator):
        # the ListView is iterable
        if accumulator[2] >= self.window_size: 
            accumulator[0].clear()
            accumulator[2] = 0
            yield accumulator[1]

    def create_accumulator(self):
        self.window_size = 672
        self.slide_size = 672
        return Row(ListView(), 0, 0)

    def accumulate(self, accumulator, value, ts, key):
        accumulator[0].add(value)
        accumulator[1] = ts
        accumulator[2] += 1
        #if accumulator[3] != "":
        #    assert accumulator[3] == key, f"Key mismatch {accumulator[3]}, {key}"
        #accumulator[3] = key

    def get_accumulator_type(self):
        return DataTypes.ROW([
            # declare the first column of the accumulator as a string ListView.
            DataTypes.FIELD("f0", DataTypes.LIST_VIEW(DataTypes.DOUBLE())),
            DataTypes.FIELD("f1", DataTypes.BIGINT()),
            DataTypes.FIELD("f2", DataTypes.BIGINT())
            #DataTypes.FIELD("f3", DataTypes.VARCHAR(8))
        ])

    def get_result_type(self):
        #return DataTypes.BIGINT()
        return 'ROW<a BIGINT>'

class WeightedAvg(AggregateFunction):

    def create_accumulator(self):
        # Row(sum, count)
        self.window_size = 672
        return Row(0, 0)

    def get_value(self, accumulator):
        if accumulator[1] == 0:
            return None
        else:
            return accumulator[0] / accumulator[1]

    def accumulate(self, accumulator, value, weight):
        accumulator[0] += value 
        accumulator[1] += 1
    
    def retract(self, accumulator, value, weight):
        accumulator[0] -= value * weight
        accumulator[1] -= weight
        
    def get_result_type(self):
        return DataTypes.INT()
        
    def get_accumulator_type(self):
        return DataTypes.ARRAY(DataTypes.FLOAT()) 


def log_processing():


    kafka_jar = "/home/ubuntu/experiments/flink-1.17.1/lib/flink-sql-connector-kafka-1.17.1.jar"

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.add_jars(f"file://{kafka_jar}")
    t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    t_env.get_config().set("pipeline.jars", f"file://{kafka_jar}")

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

    sink_ddl = """
            CREATE TABLE sink_table(
                key VARCHAR
                ts BIGINT
            ) WITH (
              'connector' = 'kafka',
              'topic' = 'sink_topic',
              'properties.bootstrap.servers' = 'kafka:9092',
              'format' = 'json'
            )
            """

    t_env.execute_sql(source_ddl)

    table = t_env.from_path("source_table")
    source_table = table.group_by(col('key'))
    table.print_schema()
    #t_env.sql_query("SELECT a FROM source_table") \
    #    .execute_insert("sink_table").wait()
    # Apply the STL UDF to each window
    weighted_avg = udtaf(Top2())
    
    #source = t_env.to_data_stream(source_table)

    #result_table = source_table.select("key, SUM(cpu) as total_cpu")

    #t_env.to_data_stream(result_table).print()

    #result = source_table.select(col('key'), weighted_avg(col('cpu'), col('ts'), col('key'))) 
    table.group_by(col('key')).flat_aggregate(weighted_avg).select(col('*')).execute().print()
   # t_env.to_data_stream(result).print()
    env.execute()


if __name__ == '__main__':
    log_processing()
