import json
import matplotlib.pyplot as plt
import os

def plot_simclr(log_file, ax_loss=None, ax_acc=None, label="SimCLR"):
    if not os.path.exists(log_file): return False
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    losses = logs["loss"]
    epochs_loss = range(1, len(losses) + 1)
    
    accs = logs["knn_acc"]
    epochs_acc = [x[0] for x in accs]
    acc_vals = [x[1] for x in accs]
    
    if ax_loss: ax_loss.plot(epochs_loss, losses, label=label)
    if ax_acc: ax_acc.plot(epochs_acc, acc_vals, label=label, marker='o')
    return True

def plot_supervised(log_file, ax_loss=None, ax_acc=None, label="Supervised"):
    if not os.path.exists(log_file): return False
    with open(log_file, "r") as f:
        logs = json.load(f)
        
    losses = logs["loss"]
    epochs_loss = range(1, len(losses) + 1)
    
    accs = logs["test_acc"]
    epochs_acc = [x[0] for x in accs]
    acc_vals = [x[1] for x in accs]
    
    if ax_loss: ax_loss.plot(epochs_loss, losses, label=label)
    if ax_acc: ax_acc.plot(epochs_acc, acc_vals, label=label)
    return True

def get_final_acc(log_file):
    if not os.path.exists(log_file): return 0.0
    with open(log_file, "r") as f:
        logs = json.load(f)
    return logs["test_acc"][-1][1]

def main():
    # 1. Main loss and kNN curve for SimCLR Baseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_simclr("simclr_baseline_logs.json", ax1, ax2, "Baseline SimCLR (bs=256, temp=0.5)")
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

    # 3. Baseline Comparisons Bar Chart (CIFAR-10)
    acc_rand = get_final_acc("linprobe_random_baseline_logs.json")
    acc_simclr = get_final_acc("linprobe_simclr_logs.json")
    acc_sl = get_final_acc("sl_baseline_logs.json")
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Random Backbone", "SimCLR SSL", "Supervised"], [acc_rand, acc_simclr, acc_sl], color=['gray', 'blue', 'green'])
    plt.title("CIFAR-10 Classification Accuracy")
    plt.ylabel("Accuracy (%)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("baseline_comparisons.png", dpi=300)
    print("Saved baseline_comparisons.png")

    # 4. Temperature Ablations Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_simclr("simclr_baseline_logs.json", ax1, ax2, "temp=0.5 (Baseline)")
    plot_simclr("simclr_temp5_logs.json", ax1, ax2, "temp=5.0")
    plot_simclr("simclr_temp01_logs.json", ax1, ax2, "temp=0.1")
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

    # 5. Batch Size Ablations Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_simclr("simclr_baseline_logs.json", ax1, ax2, "BS=256 (Baseline)")
    plot_simclr("simclr_bs128_logs.json", ax1, ax2, "BS=128")
    plot_simclr("simclr_bs64_logs.json", ax1, ax2, "BS=64")
    plot_simclr("simclr_bs32_logs.json", ax1, ax2, "BS=32")
    ax1.set_title("Loss with Batch Size Variations")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.set_title("kNN Accuracy with Batch Size Variations")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("ablation_batch_size.png", dpi=300)
    print("Saved ablation_batch_size.png")

    # 6. Projector Ablations Bar Chart
    acc_simclr = get_final_acc("linprobe_simclr_logs.json") # Baseline backbone
    acc_noproj = get_final_acc("linprobe_simclr_noproj_logs.json") # Trained without proj
    acc_projout = get_final_acc("linprobe_simclr_projout_logs.json") # Linprobe on proj outputs
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(["Standard (Encoder)", "Trained w/o Projector", "Linear Probe on Projector"], 
                   [acc_simclr, acc_noproj, acc_projout], color=['blue', 'orange', 'purple'])
    plt.title("Projector Ablations on CIFAR-10")
    plt.ylabel("Accuracy (%)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("ablation_projector.png", dpi=300)
    print("Saved ablation_projector.png")

    # 7. Transfer Learning (CIFAR-100) Bar Chart
    acc_t_rand = get_final_acc("linprobe_random_cifar100_logs.json")
    acc_t_sl = get_final_acc("linprobe_sl_cifar100_logs.json")
    acc_t_simclr = get_final_acc("linprobe_simclr_cifar100_logs.json")
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Random Backbone", "Supervised (CIFAR-10)", "SimCLR (CIFAR-10)"], 
                   [acc_t_rand, acc_t_sl, acc_t_simclr], color=['gray', 'green', 'blue'])
    plt.title("Transfer Learning Accuracy on CIFAR-100")
    plt.ylabel("Accuracy (%)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("transfer_learning_cifar100.png", dpi=300)
    print("Saved transfer_learning_cifar100.png")

if __name__ == "__main__":
    main()
