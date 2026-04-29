import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import json

from dataset import get_cifar10_transforms
from models import SimCLRNet, LinearClassifier, ModifiedResNet18

def train_linprobe(pretrained_path=None, batch_size=256, epochs=100, lr=1e-3, wd=1e-6, save_name="linprobe_model.pth", is_random_baseline=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, train_transform, test_transform = get_cifar10_transforms()
    
    train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if not is_random_baseline and pretrained_path:
        simclr_model = SimCLRNet()
        simclr_model.load_state_dict(torch.load(pretrained_path))
        backbone = simclr_model.backbone
    else:
        backbone = ModifiedResNet18()
        
    backbone = backbone.to(device)
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    classifier = LinearClassifier(in_features=512, num_classes=10).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    logs = {"loss": [], "test_acc": []}
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            
            with torch.no_grad():
                features = backbone(imgs)
                
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        logs["loss"].append(avg_loss)
        
        # Eval
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                features = backbone(imgs)
                outputs = classifier(features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        acc = 100. * correct / total
        logs["test_acc"].append((epoch, acc))
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        
    torch.save(classifier.state_dict(), save_name)
    with open(save_name.replace(".pth", "_logs.json"), "w") as f:
        json.dump(logs, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="simclr_baseline.pth")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    
    if args.random:
        train_linprobe(is_random_baseline=True, save_name="linprobe_random_baseline.pth")
    else:
        train_linprobe(pretrained_path=args.pretrained, save_name="linprobe_simclr.pth")
