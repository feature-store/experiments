
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


