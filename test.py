import torch
from models.gcn_detector import ImprovedGCNDetector
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from config import ModelConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/weibo_config.yaml',
                       help='Path to the config file.')
    parser.add_argument('--dataset', type=str, choices=['weibo', 'reddit'],
                       help='Dataset to use. Will override config file if specified.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 如果指定了数据集，使用对应的配置
    if args.dataset:
        config = ModelConfig.load_dataset_config(args.dataset)
    else:
        config = ModelConfig.load_yaml(args.config)
    
    # 加载数据和模型
    data = torch.load(config.data_path)
    device = torch.device(config.device)
    data = data.to(device)
    
    # 初始化模型并加载最佳权重
    model = ImprovedGCNDetector(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers
    ).to(device)
    model.load_state_dict(torch.load(f'checkpoints/{config.dataset_name}_best.pth'))
    
    # 测试
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.squeeze().cpu().numpy()
        labels = data.y.cpu().numpy()
        
        # 计算各种指标
        auc = roc_auc_score(labels, pred)
        ap = average_precision_score(labels, pred)
        precision, recall, _ = precision_recall_curve(labels, pred)
        
    print(f'Test Results for {config.dataset_name}:')
    print(f'AUC: {auc:.4f}')
    print(f'AP: {ap:.4f}')

if __name__ == '__main__':
    main() 