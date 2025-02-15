import torch
from torch_geometric.data import Data
import torch.optim as optim
from models.gcn_detector import ImprovedGCNDetector
from models.negative_sampler import HardNegativeSampler
from models.focal_loss import FocalLoss
from sklearn.metrics import roc_auc_score
import numpy as np
from config import ModelConfig
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/weibo_config.yaml',
                       help='Path to the config file.')
    parser.add_argument('--dataset', type=str, choices=['weibo', 'reddit'],
                       help='Dataset to use. Will override config file if specified.')
    return parser.parse_args()

def train(model, data, optimizer, criterion, config):
    model.train()
    optimizer.zero_grad()
    
    # 获取正样本索引
    positive_indices = torch.where(data.y == 1)[0]
    
    # 采样硬负样本
    sampler = HardNegativeSampler(
        num_samples=config.num_neg_samples,
        k_hop=config.k_hop,
        similarity_threshold=config.similarity_threshold,
        batch_size=config.batch_size if hasattr(config, 'batch_size') else 1000
    )
    hard_negative_indices = sampler(data.x, data.edge_index, positive_indices)
    
    # 构建训练子图
    train_indices = torch.cat([positive_indices, hard_negative_indices])
    train_mask = torch.zeros_like(data.y, dtype=torch.bool)
    train_mask[train_indices] = True
    
    # 前向传播
    out = model(data.x, data.edge_index)
    loss = calculate_loss(out[train_mask], data.y[train_mask], 
                         data.edge_index, config, criterion)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return loss.item()

def calculate_loss(out, labels, edge_index, config, criterion):
    # BCE损失
    bce_loss = criterion(out.squeeze(), labels.float())
    
    # 结构正则化损失
    # 只考虑训练子图中的边
    train_nodes = torch.where(labels >= 0)[0]  # 获取训练节点的索引
    train_nodes_set = set(train_nodes.cpu().numpy())
    
    row, col = edge_index
    # 筛选出训练子图中的边
    edge_mask = torch.tensor([
        i for i, (r, c) in enumerate(zip(row.cpu(), col.cpu()))
        if r.item() in train_nodes_set and c.item() in train_nodes_set
    ], device=edge_index.device)
    
    if len(edge_mask) > 0:
        row_masked = row[edge_mask]
        col_masked = col[edge_mask]
        edge_loss = torch.mean(torch.abs(out[row_masked] - out[col_masked]))
    else:
        edge_loss = torch.tensor(0.0, device=out.device)
    
    # 总损失
    total_loss = bce_loss + config.edge_loss_weight * edge_loss
    
    return total_loss

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.squeeze().cpu().numpy()
        labels = data.y.cpu().numpy()
        auc = roc_auc_score(labels, pred)
    return auc

def main():
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 如果指定了数据集，使用对应的配置
    if args.dataset:
        config = ModelConfig.load_dataset_config(args.dataset)
    else:
        config = ModelConfig.load_yaml(args.config)
    
    # 加载数据
    data = torch.load(config.data_path)
    # 根据实际数据更新配置的输入维度
    config.in_channels = data.x.size(1)
    device = torch.device(config.device)
    data = data.to(device)
    
    # 初始化模型
    model = ImprovedGCNDetector(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 训练循环
    best_auc = 0
    patience = config.early_stopping
    counter = 0
    
    for epoch in range(config.epochs):
        loss = train(model, data, optimizer, criterion, config)
        
        if epoch % 10 == 0:
            auc = test(model, data)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f'checkpoints/{config.dataset_name}_best.pth')
                counter = 0
            else:
                counter += 1
                
            if counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
    
    print(f'Best AUC: {best_auc:.4f}')

if __name__ == '__main__':
    main() 