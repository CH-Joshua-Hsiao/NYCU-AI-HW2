import os
import tempfile
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import json

from dataset import get_cifar10_transforms
from models import SimCLRNet
from loss import NTXentLoss
from utils import knn_monitor

def train_simclr(batch_size=256, epochs=200, lr=3e-4, wd=1e-6, temperature=0.5, save_name="simclr_model.pth", no_projector=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    simclr_transform, _, test_transform = get_cifar10_transforms()
    
    train_dataset = CIFAR10(root='./data', train=True, transform=simclr_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    memory_dataset = CIFAR10(root='./data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SimCLRNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = NTXentLoss(temperature=temperature)
    
    logs = {"loss": [], "knn_acc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for (img1, img2), _ in pbar:
            img1, img2 = img1.to(device), img2.to(device)
            
            optimizer.zero_grad()
            h1, z1 = model(img1)
            h2, z2 = model(img2)
            
            if no_projector:
                loss = criterion(h1, h2)
            else:
                loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logs["loss"].append(avg_loss)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            knn_acc = knn_monitor(model.backbone, memory_loader, test_loader, device, k=20)
            logs["knn_acc"].append((epoch, knn_acc))
            print(f"Epoch {epoch} | kNN Monitor Accuracy: {knn_acc:.2f}%")
            
    torch.save(model.state_dict(), save_name)
    with open(save_name.replace(".pth", "_logs.json"), "w") as f:
        json.dump(logs, f)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--save_name", type=str, default="simclr_baseline.pth")
    parser.add_argument("--no_projector", action="store_true")
    args = parser.parse_args()
    
    train_simclr(batch_size=args.batch_size, epochs=args.epochs, temperature=args.temperature, save_name=args.save_name, no_projector=args.no_projector)
