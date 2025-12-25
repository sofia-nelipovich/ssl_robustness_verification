import matplotlib.pyplot as plt

def plot_results(train_log, val_log, title, ylabel):
    epochs = range(1, len(train_log) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_log, label=f'Train {ylabel}')
    plt.plot(epochs, val_log, label=f'Val {ylabel}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
