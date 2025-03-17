import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GuidanceNodeGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, alpha=0.5, beta=0.5):
        super(GuidanceNodeGenerator, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        
        # GNN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # 投影层，确保输出维度正确
        self.proj = nn.Linear(hidden_channels, hidden_channels)
            
        # 双线性层
        self.bilinear_ctx = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.bilinear_local = nn.Bilinear(hidden_channels, hidden_channels, 1)
        
    def forward(self, x, edge_index, batch_nodes, labels):
        device = x.device
        num_nodes = x.size(0)
        
        # 生成引导节点特征
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
        
        # 投影到正确的维度
        guidance_features = self.proj(h)
        
        # 初始化损失为0
        total_loss = torch.tensor(0.0, device=device)
        
        # 如果在训练模式下，计算损失
        if self.training and labels is not None:
            try:
                # 确保batch_nodes和labels的维度匹配
                batch_labels = labels[batch_nodes]
                batch_features = guidance_features[batch_nodes]
                
                pos_mask = batch_labels == 1
                neg_mask = batch_labels == 0
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    # 构造正负样本
                    pos_proto = torch.mean(batch_features[pos_mask], dim=0, keepdim=True)
                    neg_proto = torch.mean(batch_features[neg_mask], dim=0, keepdim=True)
                    
                    # 计算相似度
                    pos_sim = self.bilinear_ctx(batch_features, pos_proto.repeat(len(batch_features), 1))
                    neg_sim = self.bilinear_ctx(batch_features, neg_proto.repeat(len(batch_features), 1))
                    
                    # 上下文级损失
                    ctx_loss = F.binary_cross_entropy_with_logits(
                        pos_sim - neg_sim, 
                        batch_labels.float().unsqueeze(1)
                    )
                    
                    # 计算局部级损失
                    masked_x = x.clone()
                    masked_x[batch_nodes] = 0
                    
                    env_h = masked_x
                    for conv in self.convs:
                        env_h = conv(env_h, edge_index)
                        env_h = F.relu(env_h)
                    
                    env_h = self.proj(env_h)  # 确保环境特征也经过投影
                    
                    local_sim = self.bilinear_local(batch_features, env_h[batch_nodes])
                    local_targets = (1 - batch_labels).float().unsqueeze(1)
                    local_loss = F.binary_cross_entropy_with_logits(local_sim, local_targets)
                    
                    total_loss = self.beta * ctx_loss + (1 - self.beta) * local_loss
            except Exception as e:
                print(f"Warning in guidance loss computation: {e}")
                # 即使损失计算失败，仍然返回特征
        
        return guidance_features, total_loss 