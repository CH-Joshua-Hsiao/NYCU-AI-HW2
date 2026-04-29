import subprocess
import sys

def run_cmd(cmd):
    print(f"========== RUNNING: {cmd} ==========")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # 1. Train SimCLR Baseline (using batch_size 256 as requested)
    run_cmd("python train_simclr.py --epochs 2 --batch_size 256 --save_name simclr_baseline.pth")
    
    # 2. Linear Probing on SimCLR Baseline
    run_cmd("python train_linprobe.py --epochs 2 --pretrained simclr_baseline.pth --save_name linprobe_simclr.pth")
    
    # 3. Supervised Learning Baseline
    run_cmd("python train_supervised.py --epochs 1 --batch_size 256 --save_name sl_baseline.pth")
    
    # 4. Optional Ablations (Richer Report):
    # Lower-Bound Random Baseline
    run_cmd("python train_linprobe.py --epochs 2 --random")
    
    # Temperature Variations
    run_cmd("python train_simclr.py --epochs 5 --batch_size 256 --temperature 5.0 --save_name simclr_temp5.pth")
    run_cmd("python train_simclr.py --epochs 5 --batch_size 256 --temperature 0.1 --save_name simclr_temp01.pth")
    
    # Plot everything
    run_cmd("python plot_results.py")
    print("ALL EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()
