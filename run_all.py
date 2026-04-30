import subprocess
import sys

def run_cmd(cmd):
    print(f"\n========== RUNNING: {cmd} ==========\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # 1. Train Supervised Baseline (from scratch, 100 epochs)
    run_cmd("python train_supervised.py --epochs 100 --batch_size 256 --save_name sl_baseline.pth")

    # 2. Train SimCLR Baseline (using batch_size 256, 200 epochs)
    run_cmd("python train_simclr.py --epochs 200 --batch_size 256 --save_name simclr_baseline.pth")
    
    # 3. Linear Probing on SimCLR Baseline (100 epochs as requested in PDF)
    run_cmd("python train_linprobe.py --epochs 100 --pretrained simclr_baseline.pth --save_name linprobe_simclr.pth")
    
    # 4. Lower-Bound Random Baseline
    run_cmd("python train_linprobe.py --epochs 100 --random --save_name linprobe_random_baseline.pth")

    # 5. Temperature Ablations (200 epochs for fair comparison with baseline)
    run_cmd("python train_simclr.py --epochs 200 --batch_size 256 --temperature 5.0 --save_name simclr_temp5.pth")
    run_cmd("python train_simclr.py --epochs 200 --batch_size 256 --temperature 0.1 --save_name simclr_temp01.pth")
    
    # 6. Batch Size Ablations
    run_cmd("python train_simclr.py --epochs 200 --batch_size 128 --save_name simclr_bs128.pth")
    run_cmd("python train_simclr.py --epochs 200 --batch_size 64 --save_name simclr_bs64.pth")
    run_cmd("python train_simclr.py --epochs 200 --batch_size 32 --save_name simclr_bs32.pth")

    # 7. Projector Ablations
    # Train without projector
    run_cmd("python train_simclr.py --epochs 200 --batch_size 256 --save_name simclr_noproj.pth --no_projector")
    run_cmd("python train_linprobe.py --epochs 100 --pretrained simclr_noproj.pth --save_name linprobe_simclr_noproj.pth")
    # Linprobe WITH projector outputs from baseline
    run_cmd("python train_linprobe.py --epochs 100 --pretrained simclr_baseline.pth --save_name linprobe_simclr_projout.pth --use_projector")

    # 8. Transfer Learning on CIFAR-100
    run_cmd("python train_linprobe.py --epochs 100 --random --dataset cifar100 --save_name linprobe_random_cifar100.pth")
    run_cmd("python train_linprobe.py --epochs 100 --pretrained_sl sl_baseline.pth --dataset cifar100 --save_name linprobe_sl_cifar100.pth")
    run_cmd("python train_linprobe.py --epochs 100 --pretrained simclr_baseline.pth --dataset cifar100 --save_name linprobe_simclr_cifar100.pth")

    # 9. Plot everything
    run_cmd("python plot_results.py")
    print("ALL EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()
