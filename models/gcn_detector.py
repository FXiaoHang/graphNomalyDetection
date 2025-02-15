import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ImprovedGCNDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ImprovedGCNDetector, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # 最后一层
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 异常分数预测层
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, edge_index):
        # 特征提取
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if i != len(self.convs) - 1:  # 除最后一层外都添加非线性和dropout
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        # 预测异常分数
        scores = self.pred_layer(x)
        
        # 使用sigmoid得到最终的异常概率
        return torch.sigmoid(scores) 