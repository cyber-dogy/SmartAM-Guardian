"""
SmartAM Dataset 实现
支持多模态数据加载和按Condition划分
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings


class SmartAMDataset(Dataset):
    """
    SLM增材制造多模态异常识别数据集
    
    特点:
    - 多模态输入: RGB_V1, RGB_V2, IR_Before, IR_After
    - 按Condition划分: 同一Condition的样本必须整体划分
    - 支持二分类和异常类型分类
    """
    
    # 角色定义
    SOURCE_TRAIN_ROLES = ['source_train', 'source_repeat', 'causal_intervention_high', 'causal_intervention_low']
    UNSEEN_TEST_ROLES = ['unseen_strategy_shift', 'unseen_texture_shift', 'unseen_param_shift', 'unseen_composite_shift']
    CAUSAL_EQUIV_ROLES = ['causal_equiv_test']
    
    # 标签映射
    LABEL_TYPE_MAP = {
        'NOR': 0,
        'HEW': 1,
        'LEL': 2
    }
    
    def __init__(
        self,
        metadata_path: Union[str, Path],
        data_root: Union[str, Path],
        condition_ids: Optional[List[str]] = None,
        role_filter: Optional[List[str]] = None,
        modalities: List[str] = ['rgb_v1', 'rgb_v2', 'ir_before'],
        transform = None,
        task: str = 'binary',  # 'binary' or 'multiclass'
        load_images: bool = True
    ):
        """
        Args:
            metadata_path: metadata.csv 路径
            data_root: 数据根目录 (Causal_Image_Data)
            condition_ids: 指定要加载的Condition列表，None则加载全部
            role_filter: 按角色过滤，None则不过滤
            modalities: 要加载的模态列表
            transform: 数据变换
            task: 'binary' (0/1) 或 'multiclass' (NOR/HEW/LEL)
            load_images: 是否实际加载图像（可用于仅统计）
        """
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.task = task
        self.load_images = load_images
        self.modalities = modalities
        
        # 验证模态名称
        valid_modalities = ['rgb_v1', 'rgb_v2', 'ir_before', 'ir_after']
        for m in modalities:
            if m not in valid_modalities:
                raise ValueError(f"Unknown modality: {m}. Valid: {valid_modalities}")
        
        # 加载元数据
        self.df = pd.read_csv(self.metadata_path)
        
        # 应用过滤
        if condition_ids is not None:
            self.df = self.df[self.df['condition_id'].isin(condition_ids)]
        
        if role_filter is not None:
            self.df = self.df[self.df['role'].isin(role_filter)]
        
        if len(self.df) == 0:
            warnings.warn("Filtered dataset is empty!")
        
        # 重置索引
        self.df = self.df.reset_index(drop=True)
        
        # 获取唯一的conditions
        self.conditions = self.df['condition_id'].unique().tolist()
        
        print(f"Dataset loaded: {len(self.df)} samples from {len(self.conditions)} conditions")
        print(f"Modalities: {modalities}")
        print(f"Task: {task}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            dict: {
                'images': {modality: tensor, ...},
                'label': int,
                'label_type': int (if task='multiclass'),
                'metadata': dict,
                'condition_id': str
            }
        """
        row = self.df.iloc[idx]
        
        # 加载图像
        images = {}
        if self.load_images:
            condition = row['condition_id']
            layer_id = int(row['layer_id'])
            img_dir = self.data_root / 'images' / condition / f'L{layer_id:04d}'
            
            # 映射模态名称到文件名
            modality_file_map = {
                'rgb_v1': 'rgb_view1.jpg',
                'rgb_v2': 'rgb_view2.jpg',
                'ir_before': 'ir_before.jpg',
                'ir_after': 'ir_after.jpg'
            }
            
            for modality in self.modalities:
                img_path = img_dir / modality_file_map[modality]
                if img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    images[modality] = img
                else:
                    warnings.warn(f"Image not found: {img_path}")
                    # 创建空白图像作为占位
                    images[modality] = Image.new('RGB', (224, 224), color='black')
            
            # 应用变换
            if self.transform is not None:
                images = self.transform(images)
        
        # 获取标签
        label = int(row['defect_label_binary'])
        
        result = {
            'images': images,
            'label': torch.tensor(label, dtype=torch.long),
            'condition_id': row['condition_id'],
            'metadata': {
                'condition_id': row['condition_id'],
                'layer_id': int(row['layer_id']),
                'power_w': int(row['power_w']),
                'speed_mms': int(row['speed_mms']),
                'spacing_mm': float(row['spacing_mm']),
                'scan_strategy': row['scan_strategy'],
                'energy_density': float(row['energy_density']),
                'role': row['role'],
            }
        }
        
        # 多分类标签
        if self.task == 'multiclass':
            label_type = row['defect_label_type']
            result['label_type'] = torch.tensor(
                self.LABEL_TYPE_MAP.get(label_type, 0),
                dtype=torch.long
            )
        
        return result
    
    def get_condition_distribution(self):
        """获取各Condition的样本分布"""
        return self.df.groupby('condition_id').size().to_dict()
    
    def get_label_distribution(self):
        """获取标签分布"""
        return self.df['defect_label_binary'].value_counts().to_dict()
    
    def get_role_distribution(self):
        """获取角色分布"""
        return self.df['role'].value_counts().to_dict()


def split_by_condition(
    metadata_path: Union[str, Path],
    train_roles: Optional[List[str]] = None,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    按Condition划分训练/验证/测试集
    
    策略:
    - 训练集: source_train + source_repeat + causal_intervention_* 的随机部分
    - 验证集: 从训练角色中随机抽部分Condition
    - 测试集: unseen_* 角色 + causal_equiv_test
    
    Args:
        metadata_path: metadata.csv 路径
        train_roles: 训练集角色，默认使用 SOURCE_TRAIN_ROLES
        val_ratio: 验证集占训练Condition的比例
        random_seed: 随机种子
    
    Returns:
        (train_conditions, val_conditions, test_conditions)
    """
    df = pd.read_csv(metadata_path)
    
    if train_roles is None:
        train_roles = SmartAMDataset.SOURCE_TRAIN_ROLES
    
    # 获取各角色的Condition
    train_conditions = df[df['role'].isin(train_roles)]['condition_id'].unique()
    test_conditions = df[df['role'].isin(SmartAMDataset.UNSEEN_TEST_ROLES + SmartAMDataset.CAUSAL_EQUIV_ROLES)]['condition_id'].unique()
    
    # 从训练集中划分验证集
    np.random.seed(random_seed)
    n_val = max(1, int(len(train_conditions) * val_ratio))
    val_indices = np.random.choice(len(train_conditions), size=n_val, replace=False)
    val_conditions = [train_conditions[i] for i in val_indices]
    train_conditions = [c for c in train_conditions if c not in val_conditions]
    
    print("=" * 50)
    print("数据集划分结果")
    print("=" * 50)
    print(f"训练集: {len(train_conditions)} conditions")
    print(f"  {train_conditions}")
    print(f"验证集: {len(val_conditions)} conditions")
    print(f"  {val_conditions}")
    print(f"测试集: {len(test_conditions)} conditions")
    print(f"  {test_conditions}")
    
    return train_conditions.tolist(), val_conditions, test_conditions.tolist()


def create_data_loaders(
    metadata_path: Union[str, Path],
    data_root: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    val_ratio: float = 0.1,
    modalities: List[str] = ['rgb_v1', 'rgb_v2', 'ir_before'],
    task: str = 'binary',
    random_seed: int = 42
) -> Dict[str, DataLoader]:
    """
    创建训练/验证/测试 DataLoader
    
    Args:
        metadata_path: metadata.csv 路径
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        val_ratio: 验证集比例
        modalities: 使用的模态
        task: 'binary' 或 'multiclass'
        random_seed: 随机种子
    
    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader}
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    # 划分Condition
    train_conditions, val_conditions, test_conditions = split_by_condition(
        metadata_path, val_ratio=val_ratio, random_seed=random_seed
    )
    
    # 创建数据集
    train_dataset = SmartAMDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        condition_ids=train_conditions,
        modalities=modalities,
        transform=get_train_transforms(img_size),
        task=task
    )
    
    val_dataset = SmartAMDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        condition_ids=val_conditions,
        modalities=modalities,
        transform=get_val_transforms(img_size),
        task=task
    )
    
    test_dataset = SmartAMDataset(
        metadata_path=metadata_path,
        data_root=data_root,
        condition_ids=test_conditions,
        modalities=modalities,
        transform=get_val_transforms(img_size),
        task=task
    )
    
    # 创建DataLoader
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders


def collate_multimodal(batch):
    """
    多模态数据的collate函数
    将列表的dict转换为dict的列表/张量
    """
    images = {key: [] for key in batch[0]['images'].keys()}
    labels = []
    labels_type = []
    condition_ids = []
    metadata = []
    
    for item in batch:
        for key, img in item['images'].items():
            images[key].append(img)
        labels.append(item['label'])
        condition_ids.append(item['condition_id'])
        metadata.append(item['metadata'])
        if 'label_type' in item:
            labels_type.append(item['label_type'])
    
    # 堆叠张量
    for key in images:
        if images[key][0] is not None:
            images[key] = torch.stack(images[key])
        else:
            images[key] = None
    
    result = {
        'images': images,
        'labels': torch.stack(labels),
        'condition_ids': condition_ids,
        'metadata': metadata
    }
    
    if labels_type:
        result['labels_type'] = torch.stack(labels_type)
    
    return result
