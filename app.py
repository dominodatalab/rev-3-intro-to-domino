# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
from dash.dependencies import Input, Output
import requests
import datetime
import os

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.update({'requests_pathname_prefix': '/{}/{}/r/notebookSession/{}/'.format(
    os.environ.get("DOMINO_PROJECT_OWNER"),
    os.environ.get("DOMINO_PROJECT_NAME"),
    os.environ.get("DOMINO_RUN_ID"))})

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Plot configs
prediction_color = '#0072B2'
error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
actual_color = 'black'
cap_color = 'black'
trend_color = '#B23B00'
line_width = 2
marker_size = 4
uncertainty=True
plot_cap=True
trend=False
changepoints=False
changepoints_threshold=0.01
xlabel='ds'
ylabel='y'

app.layout = html.Div(style={'paddingLeft': '40px', 'paddingRight': '40px'}, children=[
        html.H1(children='Predictor for Power Generation in UK'),
        html.Div(children='''
        This is a web app developed in Dash and published in Domino.
        You can add more description here to describe the app.
    '''),
         html.Div([
            html.P('Select a Fuel Type:', className='fuel_type', id='fuel_type_paragraph'),
            dcc.Dropdown(
                    options=[
                        {'label': 'Combined Cycle Gas Turbine', 'value': 'CCGT'},
                        {'label': 'Oil', 'value': 'OIL'},
                        {'label': 'Coal', 'value': 'COAL'},
                        {'label': 'Nuclear', 'value': 'NUCLEAR'},
                        {'label': 'Wind', 'value': 'WIND'},
                        {'label': 'Pumped Storage', 'value': 'PS'},
                        {'label': 'Hydro (Non Pumped Storage', 'value': 'NPSHYD'},
                        {'label': 'Open Cycle Gas Turbine', 'value': 'OCGT'},
                        {'label': 'Other', 'value': 'OTHER'},
                        {'label': 'France (IFA)', 'value': 'INTFR'},
                        {'label': 'Northern Ireland (Moyle)', 'value': 'INTIRL'},
                        {'label': 'Netherlands (BritNed)', 'value': 'INTNED'},
                        {'label': 'Ireland (East-West)', 'value': 'INTEW'},
                        {'label': 'Biomass', 'value': 'BIOMASS'},
                        {'label': 'Belgium (Nemolink)', 'value': 'INTEM'}
                    ],
                    value='CCGT',
                    id='fuel_type',
                    style = {'width':'auto', 'min-width': '300px'}
                )
        ], style={'marginTop': 25}),
        html.Div([
                html.Div('Training data will end today.'),
                html.Div('Select the starting date for the training data:'),
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=dt(2022, 1, 1)
                )
        ], style={'marginTop': 25}),
        html.Div([
                dcc.Loading(
            id="loading",
            children=[dcc.Graph(id='prediction_graph',)],
            type="circle",
            ),
                ], style={'marginTop': 25})
])

@app.callback(
    # Output('loading', 'chhildren'),
    Output('prediction_graph', 'figure'),
    [Input('fuel_type', 'value'),
     Input('date-picker', 'date')])
def update_output(fuel_type, start_date):
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        start_date_reformatted = start_date.split('T')[0]
        url = 'https://www.bmreports.com/bmrs/?q=ajax/filter_csv_download/FUELHH/csv/FromDate%3D{start_date}%26ToDate%3D{today}/&filename=GenerationbyFuelType_20191002_1657'.format(start_date = start_date_reformatted, today = today)
        r = requests.get(url, allow_redirects=True)
        open('data.csv', 'wb').write(r.content)
        df = pd.read_csv('data.csv', skiprows=1, skipfooter=1, header=None, engine='python')
        df = df.iloc[:,0:18]
        df.columns = ['HDF', 'date', 'half_hour_increment',
                'CCGT', 'OIL', 'COAL', 'NUCLEAR',
                'WIND', 'PS', 'NPSHYD', 'OCGT',
                'OTHER', 'INTFR', 'INTIRL', 'INTNED', 'INTEW', 'BIOMASS', 'INTEM']
        df['datetime'] = pd.to_datetime(df['date'], format="%Y%m%d")
        df['datetime'] = df.apply(lambda x:
                          x['datetime']+ datetime.timedelta(
                              minutes=30*(int(x['half_hour_increment'])-1))
                          , axis = 1)
        df_for_prophet = df[['datetime', fuel_type]].rename(columns = {'datetime':'ds', fuel_type:'y'})
        m = Prophet()
        m.fit(df_for_prophet)
        future = m.make_future_dataframe(periods=72, freq='H')
        fcst = m.predict(future)
        # from https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py
        data = []
    # Add actual
        data.append(go.Scatter(
                name='Actual',
                x=m.history['ds'],
                y=m.history['y'],
                marker=dict(color=actual_color, size=marker_size),
                mode='markers'
        ))
    # Add lower bound
        if uncertainty and m.uncertainty_samples:
                data.append(go.Scatter(
                        x=fcst['ds'],
                        y=fcst['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        hoverinfo='skip'
                ))
    # Add prediction
        data.append(go.Scatter(
                name='Predicted',
                x=fcst['ds'],
                y=fcst['yhat'],
                mode='lines',
                line=dict(color=prediction_color, width=line_width),
                fillcolor=error_color,
                fill='tonexty' if uncertainty and m.uncertainty_samples else 'none'
        ))
    # Add upper bound
        if uncertainty and m.uncertainty_samples:
                data.append(go.Scatter(
                        x=fcst['ds'],
                        y=fcst['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        fillcolor=error_color,
                        fill='tonexty',
                        hoverinfo='skip'
                ))
    # Add caps
        if 'cap' in fcst and plot_cap:
                data.append(go.Scatter(
                    name='Cap',
                    x=fcst['ds'],
                    y=fcst['cap'],
                    mode='lines',
                    line=dict(color=cap_color, dash='dash', width=line_width),
        ))
        if m.logistic_floor and 'floor' in fcst and plot_cap:
            data.append(go.Scatter(
                name='Floor',
                x=fcst['ds'],
                y=fcst['floor'],
                mode='lines',
                line=dict(color=cap_color, dash='dash', width=line_width),
            ))
    # Add trend
        if trend:
            data.append(go.Scatter(
                name='Trend',
                x=fcst['ds'],
                y=fcst['trend'],
                mode='lines',
                line=dict(color=trend_color, width=line_width),
            ))
        # Add changepoints
        if changepoints:
            signif_changepoints = m.changepoints[
                np.abs(np.nanmean(m.params['delta'], axis=0)) >= changepoints_threshold
            ]
            data.append(go.Scatter(
                x=signif_changepoints,
                y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
                marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                            line=dict(width=line_width)),
                mode='markers',
                hoverinfo='skip'
            ))

        layout = dict(
            showlegend=False,
            yaxis=dict(
                title=ylabel
            ),
            xaxis=dict(
                title=xlabel,
                type='date',
                rangeselector=dict(
                    buttons=list([
                        dict(count=7,
                             label='1w',
                             step='day',
                             stepmode='backward'),
                        dict(count=1,
                             label='1m',
                             step='month',
                             stepmode='backward'),
                        dict(count=6,
                             label='6m',
                             step='month',
                             stepmode='backward'),
                        dict(count=1,
                             label='1y',
                             step='year',
                             stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(
                    visible=True 
                ),
            ),
        )
        return {
                'data': data,
                'layout': layout
        }

if __name__ == '__main__':
    app.run_server(port=8888, host='0.0.0.0', debug=False)