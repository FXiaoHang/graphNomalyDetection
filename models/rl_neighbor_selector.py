import torch
import torch.nn as nn
import torch.nn.functional as F

class RLNeighborSelector(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(RLNeighborSelector, self).__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(in_channels * 3, hidden_channels),  # 目标节点、邻居节点和引导节点特征
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 价值预测器
        self.value_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, edge_index, guidance_features, labels, epsilon=0.1):
        row, col = edge_index
        
        # 构建状态表示
        target_features = x[row]
        neighbor_features = x[col]
        guidance_neighbor_features = guidance_features[col]
        
        state_features = torch.cat([
            target_features,
            neighbor_features,
            guidance_neighbor_features
        ], dim=1)
        
        # 编码状态
        encoded_states = self.state_encoder(state_features)
        
        # 预测邻居选择概率
        selection_probs = self.policy_net(encoded_states)
        
        if self.training:
            # 训练时使用ε-贪心策略
            random_actions = torch.rand_like(selection_probs) < epsilon
            actions = torch.where(
                random_actions,
                (torch.rand_like(selection_probs) > 0.5).float(),
                (selection_probs > 0.5).float()
            )
        else:
            # 测试时直接使用策略输出
            actions = (selection_probs > 0.5).float()
        
        # 计算价值预测损失
        value_preds = self.value_predictor(encoded_states)
        value_targets = labels[row].float().unsqueeze(1)
        value_loss = F.binary_cross_entropy_with_logits(value_preds, value_targets)
        
        # 选择邻居
        selected_edges = torch.where(actions.squeeze() > 0)[0]
        selected_edge_index = edge_index[:, selected_edges]
        
        return selected_edge_index, value_loss, actions 