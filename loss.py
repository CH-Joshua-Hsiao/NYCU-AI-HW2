import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        """
        z_i and z_j are B x 128 normalized feature maps
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        z = torch.cat([z_i, z_j], dim=0) # 2N x 128
        
        # Pairwise cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) # 2N x 2N
        sim_matrix = sim_matrix / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Labels: the mate of i is i + N, mate of i + N is i
        positives = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(device)
        
        loss = self.criterion(sim_matrix, positives)
        return loss
