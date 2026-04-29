import torch
import torch.nn.functional as F

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_monitor(net, memory_data_loader, test_data_loader, device, k=20, t=0.1, num_classes=10):
    net.eval()
    classes = num_classes
    total_top1 = 0
    total_num = 0
    feature_bank = []
    
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            # For SimCLR, memory loader returns (data, target)
            feature = net(data.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=device)
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            
    return total_top1 / total_num * 100
