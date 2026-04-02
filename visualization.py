import matplotlib.pyplot as plt

def plot_calibration(y_true, y_drifted, y_pred, title="Calibration Result", num_points=500, save_path=None):
    plt.figure(figsize=(10, 5))
    
    plt.plot(y_true[:num_points], label="True")
    plt.plot(y_drifted[:num_points], label="Drifted")
    plt.plot(y_pred[:num_points], label="Predicted")
    
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("PM2.5")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_error(y_true, y_pred, save_path=None):
    errors = y_true - y_pred

    plt.hist(errors, bins=50)
    plt.title("Error Distribution")

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_learning_curve(train_losses, val_losses=None, save_path=None):

    plt.figure(figsize=(8, 5))

    # Plot training loss
    plt.plot(train_losses, label="Train Loss")

    # Plot validation loss (if available)
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()    