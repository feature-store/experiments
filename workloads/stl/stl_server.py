from ralf.v2 import LIFO, FIFO, BaseTransform, RalfApplication, RalfConfig, Record
from ralf.v2.operator import OperatorConfig, SimpyOperatorConfig, RayOperatorConfig
from dataclasses import dataclass
from typing import List
import os
from collections import defaultdict
import pandas as pd
import simpy
from statsmodels.tsa.seasonal import STL


@dataclass 
class SourceValue: 
    key: str
    value: int

@dataclass 
class WindowValue: 
    key: str
    value: List[int]

@dataclass 
class TimeSeriesValue: 
    key: str
    trend: float 
    seasonality: List[float]


class YahooSource(BaseTransform): 
    def __init__(self, dataset: str, keys: List[int]) -> None:

        import wandb
        # download data 
        run = wandb.init()
        artifact = run.use_artifact('ucb-ralf/stl/yahoo:v0', type='dataset')
        artifact_dir = artifact.download()

        self.data = defaultdict(list)
        self.ts = 0

        # read data
        data_dir = f"{artifact_dir}/{dataset}/"
        for key in keys: 
            data_file = os.path.join(data_dir, f"{key}.csv")
            df = pd.read_csv(data_file)
            for index, row in df.iterrows():
                self.data[key].append(row.to_dict())

    def on_event(self, _: Record) -> List[Record[SourceValue]]:
        records = []
        for key in self.data.keys():
            if self.ts >= len(self.data[key]):
                raise StopIteration()
            
            records.append(Record(SourceValue(key=key, value=self.data[key][self.ts]["value"])))
        self.ts += 1
        return records


class CounterSource(BaseTransform):
    def __init__(self, up_to: int) -> None:
        self.count = 0
        self.up_to = up_to

    def on_event(self, _: Record) -> Record[SourceValue]:
        self.count += 1
        if self.count >= self.up_to:
            raise StopIteration()
        return Record(SourceValue(key=str(self.count%10), value=self.count))

class Window(BaseTransform):
    def __init__(self, window_size, slide_size=None) -> None:
        self._data = defaultdict(list)
        self.window_size = window_size
        self.slide_size = window_size if slide_size is None else slide_size 

    def on_event(self, record: Record):
        self._data[record.entry.key].append(record.entry.value)

        if len(self._data[record.entry.key]) >= self.window_size: 
            window = list(self._data[record.entry.key])
            self._data[record.entry.key] = self._data[record.entry.key][self.slide_size:]
            assert len(self._data[record.entry.key]) == self.window_size - self.slide_size, f"List length is wrong size {len(self._data[record.entry.key])}"

            # return window record
            #print("window", record.entry.key, window)
            return Record(WindowValue(key=record.entry.key, value=window))

class STLFit(BaseTransform): 
    def __init__(self): 
        self.seasonality = 12

    def on_event(self, record: Record): 
        stl_result = STL(record.entry.value, period=self.seasonality, robust=True).fit() 
        trend = stl_result.trend[-1]
        seasonality = list(stl_result.seasonal[-(self.seasonality + 1) : -1])
        print(record.entry.key, trend, seasonality)
        return Record(TimeSeriesValue(key=record.entry.key, trend=trend, seasonality=seasonality))


def main():
    print("Running STL pipeline on ralf...")

    deploy_mode = "local"
    #deploy_mode = "simpy"
    app = RalfApplication(RalfConfig(deploy_mode=deploy_mode))

    # create simulation env 
    if deploy_mode == "simpy": 
        env = simpy.Environment()
    else: 
        env = None

    # create feature frames
    window_ff = app.source(
        YahooSource("A4", keys=list(range(1, 101, 1))),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
                stop_after_s=10
            ),         
            ray_config=RayOperatorConfig(num_replicas=2)
    )
    ).transform(
        Window(window_size=200), 
        scheduler=FIFO(), 
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.01, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=2)
        )
    )
    stl_ff = window_ff.transform(
        STLFit(),
        scheduler=LIFO(),
        operator_config=OperatorConfig(
            simpy_config=SimpyOperatorConfig(
                shared_env=env, 
                processing_time_s=0.2, 
            ),         
            ray_config=RayOperatorConfig(num_replicas=2)
        )
    )


    app.deploy()

    env.run(100)
    app.wait()


if __name__ == "__main__":
    main()
