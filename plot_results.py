import json
import matplotlib.pyplot as plt

def plot_simclr(log_file="simclr_baseline_logs.json", ax_loss=None, ax_acc=None, label="SimCLR Baseline"):
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    losses = logs["loss"]
    epochs_loss = range(1, len(losses) + 1)
    
    accs = logs["knn_acc"]
    epochs_acc = [x[0] for x in accs]
    acc_vals = [x[1] for x in accs]
    
    if ax_loss:
        ax_loss.plot(epochs_loss, losses, label=label)
    if ax_acc:
        ax_acc.plot(epochs_acc, acc_vals, label=label, marker='o')

def plot_supervised(log_file="sl_baseline_logs.json", ax_loss=None, ax_acc=None, label="Supervised Baseline"):
    with open(log_file, "r") as f:
        logs = json.load(f)
        
    losses = logs["loss"]
    epochs_loss = range(1, len(losses) + 1)
    
    accs = logs["test_acc"]
    epochs_acc = [x[0] for x in accs]
    acc_vals = [x[1] for x in accs]
    
    if ax_loss:
        ax_loss.plot(epochs_loss, losses, label=label)
    if ax_acc:
        ax_acc.plot(epochs_acc, acc_vals, label=label)

def main():
    # 1. Main loss and kNN curve for SimCLR Baseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_simclr("simclr_baseline_logs.json", ax1, ax2, "Baseline SimCLR (temp=0.5, bs=256)")
    ax1.set_title("Training NT-Xent Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.set_title("kNN Monitor Accuracy (k=20)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("simclr_learning_curves.png", dpi=300)
    print("Saved simclr_learning_curves.png")

    # 2. Supervised Learning loss and accuracy curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_supervised("sl_baseline_logs.json", ax1, ax2, "Supervised ResNet-18")
    ax1.set_title("Training Cross-Entropy Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.set_title("Test Set Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("supervised_learning_curves.png", dpi=300)
    print("Saved supervised_learning_curves.png")

    # 3. Ablations Plot (Optional)
    # This will assume the optional logs exist. If not, it skips gracefully.
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_simclr("simclr_baseline_logs.json", ax1, ax2, "temp=0.5 (Baseline)")
        try: plot_simclr("simclr_temp5_logs.json", ax1, ax2, "temp=5.0")
        except: pass
        try: plot_simclr("simclr_temp01_logs.json", ax1, ax2, "temp=0.1")
        except: pass
        
        ax1.set_title("Loss with Temperature Variations")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2.set_title("kNN Accuracy with Temperature Variations")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("ablation_temperature.png", dpi=300)
        print("Saved ablation_temperature.png")
    except Exception as e:
        print("Skipped ablation plotting:", e)

if __name__ == "__main__":
    main()
