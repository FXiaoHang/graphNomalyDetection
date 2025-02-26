import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GuidanceNodeGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, alpha=0.5, beta=0.5):
        super(GuidanceNodeGenerator, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha  # 正样本权重
        self.beta = beta    # 上下文级与局部级损失的平衡参数
        
        # GNN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # 双线性层
        self.bilinear_ctx = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.bilinear_local = nn.Bilinear(hidden_channels, hidden_channels, 1)
        
    def forward(self, x, edge_index, batch_nodes, labels):
        # 生成引导节点特征
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
        
        guidance_features = h[batch_nodes]
        
        # 计算上下文级损失
        pos_mask = labels[batch_nodes] == 1
        neg_mask = ~pos_mask
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            # 构造正负样本
            pos_proto = torch.mean(guidance_features[pos_mask], dim=0, keepdim=True)
            neg_proto = torch.mean(guidance_features[neg_mask], dim=0, keepdim=True)
            
            # 计算相似度
            pos_sim = self.bilinear_ctx(guidance_features, pos_proto.repeat(len(guidance_features), 1))
            neg_sim = self.bilinear_ctx(guidance_features, neg_proto.repeat(len(guidance_features), 1))
            
            # 上下文级损失
            ctx_loss = F.binary_cross_entropy_with_logits(
                pos_sim, 
                labels[batch_nodes].float().unsqueeze(1)
            )
            
            # 计算局部级损失
            # 屏蔽目标节点
            masked_x = x.clone()
            masked_x[batch_nodes] = 0
            
            # 获取环境表示
            env_h = masked_x
            for conv in self.convs:
                env_h = conv(env_h, edge_index)
                env_h = F.relu(env_h)
            
            env_features = env_h[batch_nodes]
            local_sim = self.bilinear_local(env_features, guidance_features)
            
            # 局部级损失
            local_targets = (1 - labels[batch_nodes]).float().unsqueeze(1)  # 欺诈节点应与环境不同
            local_loss = F.binary_cross_entropy_with_logits(local_sim, local_targets)
            
            # 总损失
            total_loss = self.beta * ctx_loss + (1 - self.beta) * local_loss
        else:
            total_loss = torch.tensor(0.0, device=x.device)
            
        return guidance_features, total_loss 