#import the packages we need
import sys
import datetime
import pandas as pd
import requests
from fbprophet import Prophet
import matplotlib.pyplot as plt

#define some important variables based on the input parameters
#this script is designed to be used with a launcher accepting two parameters:
#1 start_date: Date formatted like 'Tue Feb 01 2022 00:00:00 GMT-0500 (Eastern Standard Time)'
#2 fuel_type: select from list below
#CCGT, OIL, COAL, NUCLEAR, WIND, PS, NPSHYD, OCGT, OTHER, INTFR, INTIRL, INTNED, INTEW, BIOMASS, INTEM
start_date_str = sys.argv[1]
fuel_type = sys.argv[2]
start_date = datetime.datetime.strptime(start_date_str.split(' (')[0], '%a %b %d %Y 00:00:00 GMT%z').strftime('%Y-%m-%d')
today = datetime.datetime.today().strftime('%Y-%m-%d')
new_data_path = f'results/launcher_{today}_{fuel_type}_from_{start_date}_data.csv'
new_data_url = f"https://www.bmreports.com/bmrs/?q=ajax/filter_csv_download/FUELHH/csv/FromDate%3D{start_date}%26ToDate%3D{today}/"
new_predictions_path = f'results/launcher_{today}_{fuel_type}_from_{start_date}_prediction.csv'
new_plot_path = f'results/launcher_{today}_{fuel_type}_from_{start_date}_prediction.png'

#Collect New Data for Combined Cycle Gas Turbine Generations
response = requests.get(new_data_url)
with open(new_data_path, 'wb') as f:
    f.write(response.content)

#read in our data
df = pd.read_csv(new_data_path, skiprows=1, skipfooter=1, header=None, engine='python')
df = df.iloc[:,0:18]

#rename the columns
df = df.iloc[:,0:18]
df.columns = ['HDF', 'date', 'half_hour_increment',
              'CCGT', 'OIL', 'COAL', 'NUCLEAR',
              'WIND', 'PS', 'NPSHYD', 'OCGT',
              'OTHER', 'INTFR', 'INTIRL', 'INTNED',
               'INTEW', 'BIOMASS', 'INTEM']

#Create a new column datetime that represents the starting datetime of the measured increment
df['datetime'] = pd.to_datetime(df['date'], format="%Y%m%d")
df['datetime'] = df.apply(lambda x:x['datetime']+ datetime.timedelta(minutes=30*(int(x['half_hour_increment'])-1)), 
                          axis = 1)

#Prep our data - for Facebook Prophet, the time series data needs to be in a DataFrame with 2 columns named ds and y
df_for_prophet = df[['datetime', fuel_type]].rename(columns = {'datetime':'ds', fuel_type:'y'})


#fit the model with Prophet
m = Prophet(yearly_seasonality=False)
m.fit(df_for_prophet);

#Make a DataFrame to hold the predictions and predict future values of CCGT power generation
future = m.make_future_dataframe(periods=24*7, freq='H')
forecast = m.predict(future)

#save the predictions
forecast.to_csv(new_predictions_path)

#Plot the fitted line and predictions
fig = m.plot(forecast)
plt.ylabel(fuel_type)
plt.savefig(new_plot_path)