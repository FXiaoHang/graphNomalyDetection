import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.guidance_node import GuidanceNodeGenerator
from models.rl_neighbor_selector import RLNeighborSelector

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
        
        # 引导节点生成器
        self.guidance_generator = GuidanceNodeGenerator(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        )
        
        # RL邻居选择器
        self.neighbor_selector = RLNeighborSelector(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        )
        
        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # 预测层
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # 包含原始特征和引导特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, edge_index, batch_nodes=None, labels=None):
        if batch_nodes is None:
            batch_nodes = torch.arange(x.size(0), device=x.device)
            
        # 1. 生成引导节点特征
        guidance_features, guidance_loss = self.guidance_generator(
            x, edge_index, batch_nodes, labels
        )
        
        # 2. RL邻居选择
        selected_edge_index, value_loss, _ = self.neighbor_selector(
            x, edge_index, guidance_features, labels
        )
        
        # 3. 消息传递
        h = x
        for conv in self.convs:
            h = conv(h, selected_edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
        
        # 4. 特征融合与预测
        # 确保特征维度匹配
        if guidance_features is not None and h is not None:
            final_features = torch.cat([h, guidance_features], dim=1)
            scores = self.pred_layer(final_features)
            
            if self.training:
                aux_loss = guidance_loss + value_loss if guidance_loss is not None else value_loss
                return scores, aux_loss
            else:
                return scores
        else:
            # 如果特征生成失败，只使用GCN特征
            scores = self.pred_layer(h)
            if self.training:
                return scores, torch.tensor(0.0, device=x.device)
            else:
                return scores