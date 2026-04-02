import os
import pandas as pd

def load_pollution_data(dataset_path):
    csv_path = os.path.join(dataset_path, "Measurement_summary.csv")
    return pd.read_csv(csv_path)