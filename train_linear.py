from utils.data_loader import load_pollution_data
from utils.preprocessing import clean_data, remove_outliers
from utils.drift import add_drift
from utils.metrics import compute_rmse
from utils.metrics import compute_mae
from models.linear_model import LinearCalibrationModel
from utils.visualization import plot_calibration
from utils.visualization import plot_error

import numpy as np

DATASET_PATH = "data/"

def run():
    dataset = load_pollution_data(DATASET_PATH)

    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    print("\nAFTER CLEANING:")
    print(dataset['PM2.5'].describe())

    dataset['PM2.5_drifted'] = add_drift(dataset['PM2.5'].values)

    X = dataset[['PM2.5_drifted']].values
    y = dataset['PM2.5'].values

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearCalibrationModel()
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = compute_rmse(y_test, y_pred)
    print("RMSE:", rmse)

    mae = compute_mae(y_test, y_pred)
    print("Mae: ", mae)

    plot_calibration(
        y_test,
        X_test.flatten(),
        y_pred,
        save_path="results/linear_plot.png"
    )

    plot_error(
        y_test,
        y_pred,
        save_path="results/linear_error.png"
    )



if __name__ == "__main__":
    run()

"""
RMSE: 3.4756833811426318
Mae:  3.089159442437663
"""