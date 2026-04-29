import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        # Modify conv1: 3x3 kernel, stride=1, padding=1
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Modify maxpool to identity
        resnet.maxpool = nn.Identity()
        
        # Remove fc layer to get 512 dim output (before flatten, it's global avg pooling)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], -1) # Flatten to (B, 512)
        return h

class ProjectorHead(nn.Module):
    def __init__(self, in_features=512, hidden_features=512, out_features=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )
        
    def forward(self, x):
        return self.net(x)

class SimCLRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ModifiedResNet18()
        self.projector = ProjectorHead()
        
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

class LinearClassifier(nn.Module):
    def __init__(self, in_features=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)
