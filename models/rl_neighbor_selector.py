import torch
import torch.nn as nn
import torch.nn.functional as F

class ValuePredictor(nn.Module):
    def __init__(self, hidden_channels):
        super(ValuePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x):
        return self.mlp(x)

class RLPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(RLPolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.net(state)

class RLNeighborSelector(nn.Module):
    def __init__(self, in_channels, hidden_channels, epsilon=0.1):
        super(RLNeighborSelector, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.epsilon = epsilon  # 容忍参数
        
        # 特征转换层
        self.node_transform = nn.Linear(in_channels, hidden_channels)
        self.neighbor_transform = nn.Linear(in_channels, hidden_channels)
        self.guidance_transform = nn.Linear(hidden_channels, hidden_channels)  # 引导特征已经是hidden_channels维度
        
        # 状态编码器 - 输入是三个hidden_channels维的特征拼接
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 价值预测器
        self.value_predictor = ValuePredictor(hidden_channels)
        
        # RL策略网络
        self.policy_net = RLPolicyNet(hidden_channels, hidden_channels)
        
        # 记录历史贡献值
        self.prev_contribution = None
        
    def construct_state(self, x, edge_index, guidance_features):
        row, col = edge_index
        
        # 转换特征到相同的维度
        node_features = self.node_transform(x[row])  # [num_edges, hidden_channels]
        neighbor_features = self.neighbor_transform(x[col])  # [num_edges, hidden_channels]
        guidance_neighbor_features = self.guidance_transform(guidance_features[col])  # [num_edges, hidden_channels]
        
        # 构建状态表示
        state = torch.cat([
            node_features,
            neighbor_features,
            guidance_neighbor_features
        ], dim=1)
        
        return state
    
    def compute_reward(self, contribution, action):
        if self.prev_contribution is None:
            self.prev_contribution = contribution.mean().detach()
            return torch.zeros_like(contribution)
        
        reward = torch.where(
            action > 0,
            contribution - (self.prev_contribution - self.epsilon),
            (self.prev_contribution - self.epsilon) - contribution
        )
        
        self.prev_contribution = contribution.mean().detach()
        return reward
    
    def forward(self, x, edge_index, guidance_features, labels=None):
        # 构建状态
        state = self.construct_state(x, edge_index, guidance_features)
        encoded_state = self.state_encoder(state)
        
        # 价值预测
        value_preds = self.value_predictor(encoded_state)
        
        # 策略网络预测动作概率
        action_probs = self.policy_net(encoded_state)
        
        if self.training:
            # ε-贪心策略
            random_actions = torch.rand_like(action_probs) < self.epsilon
            actions = torch.where(
                random_actions,
                (torch.rand_like(action_probs) > 0.5).float(),
                (action_probs > 0.5).float()
            )
        else:
            actions = (action_probs > 0.5).float()
        
        # 计算价值预测损失
        value_loss = torch.tensor(0.0, device=x.device)
        if self.training and labels is not None:
            row, _ = edge_index
            value_targets = labels[row].float().unsqueeze(1)
            value_loss = F.binary_cross_entropy_with_logits(value_preds, value_targets)
            
            # 计算贡献值和奖励
            with torch.no_grad():
                contribution = torch.abs(value_preds.detach() - value_targets)
                rewards = self.compute_reward(contribution, actions.detach())
            
            # 将奖励纳入价值损失，避免重复计算梯度
            policy_loss = -0.1 * (rewards.detach() * action_probs).mean()
            value_loss = value_loss + policy_loss
        
        # 选择邻居
        selected_edges = torch.where(actions.squeeze() > 0)[0]
        if len(selected_edges) == 0:  # 确保至少选择一些边
            selected_edges = torch.randint(0, edge_index.size(1), (edge_index.size(1) // 10,), device=edge_index.device)
        
        selected_edge_index = edge_index[:, selected_edges]
        
        return selected_edge_index, value_loss, actions 