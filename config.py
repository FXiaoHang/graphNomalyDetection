import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import yaml
import os

@dataclass
class ModelConfig:
    # 模型架构参数
    in_channels: int
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    
    # 训练参数
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    early_stopping: int = 20
    
    # 损失函数参数
    edge_loss_weight: float = 0.1
    aux_loss_weight: float = 0.5    # 新增：辅助损失权重
    guidance_alpha: float = 0.25    # 新增：引导节点正样本权重
    guidance_beta: float = 0.5      # 新增：上下文级与局部级损失平衡参数
    
    # 数据集参数
    dataset_name: str = 'weibo'
    data_path: str = 'weibo.pt'
    
    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 负样本采样参数
    num_neg_samples: int = 5
    k_hop: int = 2
    similarity_threshold: float = 0.5
    batch_size: int = 1000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)
    
    @classmethod
    def load_yaml(cls, yaml_path: str) -> 'ModelConfig':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        config_dict = {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'early_stopping': self.early_stopping,
            'edge_loss_weight': self.edge_loss_weight,
            'aux_loss_weight': self.aux_loss_weight,    # 新增
            'guidance_alpha': self.guidance_alpha,      # 新增
            'guidance_beta': self.guidance_beta,        # 新增
            'dataset_name': self.dataset_name,
            'data_path': self.data_path,
            'num_neg_samples': self.num_neg_samples,
            'k_hop': self.k_hop,
            'similarity_threshold': self.similarity_threshold,
            'batch_size': self.batch_size,
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False) 
    
    @classmethod
    def get_dataset_configs(cls) -> Dict[str, str]:
        """返回所有可用的数据集配置文件"""
        configs = {}
        config_dir = 'configs'
        for file in os.listdir(config_dir):
            if file.endswith('_config.yaml'):
                dataset_name = file.replace('_config.yaml', '')
                configs[dataset_name] = os.path.join(config_dir, file)
        return configs
    
    @classmethod
    def load_dataset_config(cls, dataset_name: str) -> 'ModelConfig':
        """根据数据集名称加载对应配置"""
        config_path = f'configs/{dataset_name}_config.yaml'
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration for dataset {dataset_name} not found")
        return cls.load_yaml(config_path)
    
    def update_for_dataset(self, dataset_name: str):
        """更新配置以适应特定数据集"""
        if dataset_name == 'reddit':
            self.hidden_channels = 128
            self.num_layers = 4
            self.dropout = 0.6
            self.learning_rate = 0.005
            self.weight_decay = 0.001
            self.epochs = 300
            self.early_stopping = 30
            self.edge_loss_weight = 0.15
        # 可以添加其他数据集的特定配置 