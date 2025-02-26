import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification
        Args:
            alpha: 平衡正负样本的权重
            gamma: 调制因子，降低易分样本的权重
            reduction: 损失计算方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 获取 sigmoid 后的预测概率
        probs = torch.sigmoid(inputs)
        # 计算二元交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算预测概率
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # 计算alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # 计算调制因子
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 计算最终的focal loss
        loss = alpha_t * modulating_factor * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 