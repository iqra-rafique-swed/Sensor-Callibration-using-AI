import numpy as np

def add_drift(signal):
    t = np.arange(len(signal))

    drift = 5 * np.sin(t / 200) 
    
    noise = np.random.normal(0, 0.5, len(signal))

    return signal + drift + noise