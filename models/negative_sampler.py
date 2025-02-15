import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, negative_sampling
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

class HardNegativeSampler:
    def __init__(self, num_samples=5, k_hop=2, similarity_threshold=0.5, batch_size=1000):
        self.num_samples = num_samples
        self.k_hop = k_hop
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
    
    def get_k_hop_neighbors_batch(self, edge_index, k, node_indices):
        """批量获取k跳邻居"""
        neighbors = {idx.item() for idx in node_indices}
        current_neighbors = neighbors.copy()
        
        edge_index_np = edge_index.cpu().numpy()
        for _ in range(k):
            next_neighbors = set()
            for node in current_neighbors:
                mask = edge_index_np[0] == node
                next_neighbors.update(edge_index_np[1][mask].tolist())
            current_neighbors = next_neighbors - neighbors
            neighbors.update(current_neighbors)
            
        return list(neighbors - set(node_indices.cpu().numpy().tolist()))
    
    def compute_similarity_batch(self, x, indices, all_indices):
        """批量计算相似度"""
        x_indices = x[indices]
        x_all = x[all_indices]
        
        # 使用矩阵乘法计算余弦相似度
        norm_x = F.normalize(x_indices, p=2, dim=1)
        norm_all = F.normalize(x_all, p=2, dim=1)
        similarity = torch.mm(norm_x, norm_all.t())
        
        return similarity
    
    def sample_hard_negatives(self, x, edge_index, positive_indices):
        device = x.device
        hard_negatives = []
        
        # 将正样本分批处理
        for i in tqdm(range(0, len(positive_indices), self.batch_size), desc="Sampling negatives"):
            batch_indices = positive_indices[i:i + self.batch_size]
            
            # 获取批次的k跳邻居
            k_hop_neighbors = self.get_k_hop_neighbors_batch(edge_index, self.k_hop, batch_indices)
            if not k_hop_neighbors:
                continue
                
            # 计算与邻居节点的相似度
            neighbor_indices = torch.tensor(k_hop_neighbors, device=device)
            similarities = self.compute_similarity_batch(
                x, batch_indices, neighbor_indices
            )
            
            # 选择最相似的负样本
            topk = min(self.num_samples, len(k_hop_neighbors))
            if topk > 0:
                _, top_indices = similarities.topk(topk, dim=1)
                selected_neighbors = neighbor_indices[top_indices.view(-1)].cpu().numpy()
                hard_negatives.extend(selected_neighbors.tolist())
        
        # 去重
        hard_negatives = list(set(hard_negatives))
        return torch.tensor(hard_negatives, device=device)

    def __call__(self, x, edge_index, positive_indices):
        return self.sample_hard_negatives(x, edge_index, positive_indices) 