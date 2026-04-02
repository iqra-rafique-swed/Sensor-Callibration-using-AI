import numpy as np
import pandas as pd

def clean_data(data):
    cols = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']
    
    # Replace -1 with NaN
    data[cols] = data[cols].replace(-1, np.nan)
    
    # Drop missing
    data = data.dropna()
    
    # Convert time
    data['Measurement date'] = pd.to_datetime(data['Measurement date'])
    
    # Sort
    data = data.sort_values(by='Measurement date')

    return data


def remove_outliers(data):
    data = data[
        (data['PM2.5'] >= 0) & (data['PM2.5'] <= 200) &
        (data['PM10'] >= 0) & (data['PM10'] <= 300)
    ]

    data = data[
        data['PM2.5'] < data['PM2.5'].quantile(0.995)
    ]

    return data