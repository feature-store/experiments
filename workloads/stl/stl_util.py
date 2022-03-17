from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

import pandas as pd

import warnings 

def predict_seasonality(seasonality, event_ts, model_ts, interval=1): 
    """
    Calculate predicted seasonality from last computed STL model 
    (based off different in timestamp) 
    """
    staleness = int(int(event_ts - model_ts) / interval)
    assert staleness >= 0
    assert isinstance(staleness, int)
    seasonal = seasonality[staleness % len(seasonality)]
    return seasonal 

def predict(value, trend, seasonality, event_ts, model_ts, interval=1):
    """
    Calculate predicted residual and staleness (compared to last model timestamp) given event, model
    """
    # TODO: BE CAREFUL - changes based off timestamp units
    staleness = int(int(event_ts - model_ts) / interval)
    #print(staleness, interval)
    assert staleness >= 0
    assert isinstance(staleness, int)
    seasonal = seasonality[staleness % len(seasonality)]

    # calculate residual
    residual = value - trend - seasonal
    return residual 


def remove_anomaly(df, window_size): 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for index, row in df.iterrows(): 
            if not row["is_anomaly"] or index < window_size: continue 
        
            chunk = df.iloc[index-window_size:index].value
            model = STLForecast(
                chunk, ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=24
            ).fit()
            row["value"] = model.forecast(1).tolist()[0]
            df.iloc[index] = pd.Series(row)
        return df

