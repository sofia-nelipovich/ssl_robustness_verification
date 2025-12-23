import matplotlib.pyplot as plt


def plot_all_experiments(all_histories, save_path='hyperparameter_search_results.png'):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    for name, history in all_histories.items():
        epochs = range(1, len(history['val_acc']) + 1)
        best_val = max(history['val_acc'])
        plt.plot(epochs, history['val_acc'], label=f"{name} (Max: {best_val:.1f}%)")

    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, history in all_histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label=name)

    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()
