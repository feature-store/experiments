import pickle
import time

from statsmodels.tsa.seasonal import STL, DecomposeResult


def say_hi():
    # print("hi")
    return b"a string"


def fit_window(window, seasonality) -> bytes:
    start = time.time()

    stl_data = [d["value"] for d in window]
    stl_result = STL(stl_data, period=seasonality, robust=True).fit()
    # print(f"train @ {window[-1]['timestamp']}")

    model = dict(
        trend=stl_result.trend[-1],
        seasonality=list(stl_result.seasonal[-(seasonality + 1) : -1]),
        # create_time=record.create_time,
        # complete_time=time.time(),
    )

    result = pickle.dumps(model)

    fit_time = time.time() - start
    # print(f"fit time: {fit_time}")

    return result


def predict(model_dict, event_dict) -> bytes:
    if predict.start is None:
        predict.start = time.time()
        predict.count = 0

    staleness = int(event_dict["timestamp"] - model_dict["timestamp"])  # / (60 * 60))

    model = pickle.loads(bytes(model_dict["value"]))

    last_trend = model["trend"]
    seasonal = model["seasonality"][staleness % len(model["seasonality"])]

    # calculate residual
    residual = event_dict["value"] - last_trend - seasonal

    result = pickle.dumps((seasonal, residual))

    predict.count += 1
    if predict.count % 1000 == 0:
        print(f"xput: {predict.count / (time.time() - predict.start)}")

    return result


predict.start = None
