import numpy as np

def create_sequences(data, targets, seq_length=10):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(targets[i+seq_length])

    return np.array(X), np.array(y)