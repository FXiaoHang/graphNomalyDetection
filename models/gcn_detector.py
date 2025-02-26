import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AdaptiveGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        
        # 邻居选择网络
        self.neighbor_selector = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 特征归一化层
        self.norm = nn.LayerNorm(in_channels * 2)
        
    def select_neighbors(self, x, edge_index):
        row, col = edge_index
        
        # 获取源节点和目标节点的特征
        x_i = x[row]
        x_j = x[col]
        
        # 归一化特征
        x_combined = torch.cat([x_i, x_j], dim=1)
        x_combined = self.norm(x_combined)
        
        # 计算选择概率
        selection_scores = self.neighbor_selector(x_combined)
        
        # 使用概率进行采样
        mask = torch.bernoulli(selection_scores)
        return mask.squeeze()
    
    def forward(self, x, edge_index):
        # 选择重要的邻居
        neighbor_mask = self.select_neighbors(x, edge_index)
        
        # 应用邻居选择
        row, col = edge_index
        selected_edges = torch.where(neighbor_mask > 0)[0]
        selected_edge_index = edge_index[:, selected_edges]
        
        # 执行GCN传播
        out = self.gcn(x, selected_edge_index)
        return out

class ImprovedGCNDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ImprovedGCNDetector, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(AdaptiveGCNLayer(in_channels, hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(AdaptiveGCNLayer(hidden_channels, hidden_channels))
            
        # 最后一层
        self.convs.append(AdaptiveGCNLayer(hidden_channels, hidden_channels))
        
        # 改进预测层，添加更多的特征提取能力
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index):
        # 特征提取
        features = []
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                features.append(x)
        
        # 多尺度特征融合
        if features:
            attention_weights = [self.attention(feat) for feat in features]
            attention_weights = torch.softmax(torch.cat(attention_weights, dim=1), dim=1)
            multi_scale_feature = sum([w * f for w, f in zip(attention_weights.split(1, dim=1), features)])
            x = x + multi_scale_feature
        
        # 预测异常分数
        scores = self.pred_layer(x)
        
        return torch.sigmoid(scores) 