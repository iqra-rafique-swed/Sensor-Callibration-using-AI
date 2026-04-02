import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_pollution_data
from utils.preprocessing import clean_data, remove_outliers
from utils.drift import add_drift
from utils.sequence import create_sequences
from utils.metrics import compute_rmse
from utils.visualization import (
    plot_calibration,
    plot_error,
    plot_learning_curve
)

from models.lstm_model import LSTMCalibrationModel

DATASET_PATH = "data/"


# ==============================
# DATA PREPARATION (UPGRADED)
# ==============================
def prepare_data(dataset):
    # Add drift
    dataset['PM2.5_drifted'] = add_drift(dataset['PM2.5'].values)

    # 🔥 MULTI-FEATURE INPUT
    features = ['PM2.5_drifted', 'PM10', 'NO2', 'O3', 'CO']
    X = dataset[features].values

    # 🔥 RESIDUAL LEARNING
    y_true = dataset['PM2.5'].values.reshape(-1, 1)
    drifted = dataset['PM2.5_drifted'].values.reshape(-1, 1)

    y_residual = y_true - drifted  # learn correction

    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_residual)

    return X_scaled, y_scaled, scaler_X, scaler_y, drifted


# ==============================
# TRAINING
# ==============================
def train_model(model, train_loader, X_test, y_test, criterion, optimizer, epochs=25):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()

        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")

    return train_losses, val_losses


# ==============================
# EVALUATION (UPDATED)
# ==============================
def evaluate(model, X_test, y_test, scaler_y, drifted_test):
    model.eval()
    with torch.no_grad():
        preds_residual = model(X_test).cpu().numpy()

    # Inverse scaling (residual)
    preds_residual = scaler_y.inverse_transform(preds_residual)

    # 🔥 RECONSTRUCT FINAL OUTPUT
    preds = drifted_test + preds_residual

    y_true = drifted_test + scaler_y.inverse_transform(y_test.cpu().numpy())

    rmse = compute_rmse(y_true, preds)
    mae = np.mean(np.abs(y_true - preds))

    return preds, y_true, rmse, mae


# ==============================
# MAIN
# ==============================
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load + clean
    dataset = load_pollution_data(DATASET_PATH)
    dataset = clean_data(dataset)
    dataset = remove_outliers(dataset)

    # Prepare
    X_scaled, y_scaled, scaler_X, scaler_y, drifted = prepare_data(dataset)

    # 🔥 SHORTER SEQUENCE
    seq_length = 5
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

    split = int(len(X_seq) * 0.8)

    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    drifted_test = drifted[split + seq_length:]

    # Torch
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Loader (NO SHUFFLE for time series)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # 🔥 SIMPLIFIED + STRONGER MODEL
    model = LSTMCalibrationModel(
        input_size=X_train.shape[2],
        hidden_size=128,
        num_layers=1,
        dropout=0.0
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_losses, val_losses = train_model(
        model,
        train_loader,
        X_test,
        y_test_tensor,
        criterion,
        optimizer,
        epochs=25
    )

    # Evaluate
    preds, y_true, rmse, mae = evaluate(
        model,
        X_test,
        y_test_tensor,
        scaler_y,
        drifted_test
    )

    print("\n===== IMPROVED LSTM RESULTS =====")
    print("RMSE:", rmse)
    print("MAE: ", mae)

    # Visualization
    plot_calibration(
        y_true=y_true.flatten(),
        y_drifted=drifted_test.flatten(),
        y_pred=preds.flatten(),
        title="LSTM Calibration",
        save_path="results/lstm_calibration.png"
    )

    plot_error(
        y_true.flatten(),
        preds.flatten(),
        save_path="results/lstm_error.png"
    )

    plot_learning_curve(
        train_losses,
        val_losses,
        save_path="results/lstm_learning_curve.png"
    )


if __name__ == "__main__":
    run()

"""
===== IMPROVED LSTM RESULTS =====
RMSE: 2.656676592288319
MAE:  2.1913134671126375
"""