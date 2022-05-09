import pickle
import datetime
import pandas as pd

# This is required only for Monitoring
# Note the data capture client requires at least 1 feature, but we are only capturing the prediction
from domino_data_capture.data_capture_client import DataCaptureClient
pred_client = DataCaptureClient(['dummy_y'], ['y'])

#bring in serialized model
with open('model.pkl', 'rb') as f:
    m = pickle.load(f)
    
    
#function that uses model to predict the power generation 
def predict(year, month, day):
    '''
    Input:
    year - integer
    month - integer
    day - integer

    Output:
    predicted generation in MW
    '''
    ds = pd.DataFrame({'ds': [datetime.datetime(year,month,day)]})
    prediction = m.predict(ds)['yhat'].values[0]
    pred_client.capturePrediction([prediction], [prediction]) # required only for Monitoring
    return prediction