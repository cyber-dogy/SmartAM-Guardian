"""
数据增强和预处理变换
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class MultiModalTransform:
    """
    多模态数据变换器
    对RGB和IR图像应用不同的变换
    """
    
    def __init__(self, rgb_transform, ir_transform):
        self.rgb_transform = rgb_transform
        self.ir_transform = ir_transform
    
    def __call__(self, images_dict):
        """
        Args:
            images_dict: dict with keys 'rgb_v1', 'rgb_v2', 'ir_before', 'ir_after'
        Returns:
            dict with transformed tensors
        """
        result = {}
        for key, img in images_dict.items():
            if img is None:
                result[key] = None
                continue
            if 'rgb' in key:
                result[key] = self.rgb_transform(img)
            else:  # ir
                result[key] = self.ir_transform(img)
        return result


def get_train_transforms(img_size=224):
    """
    训练集数据增强
    
    Args:
        img_size: 输出图像尺寸
    
    Returns:
        MultiModalTransform 实例
    """
    # RGB图像变换（更强的增强）
    rgb_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # IR图像变换（更保守的增强，保留温度信息）
    ir_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    return MultiModalTransform(rgb_transform, ir_transform)


def get_val_transforms(img_size=224):
    """
    验证/测试集数据变换（无增强）
    
    Args:
        img_size: 输出图像尺寸
    
    Returns:
        MultiModalTransform 实例
    """
    rgb_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    ir_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    return MultiModalTransform(rgb_transform, ir_transform)


class MixUp:
    """
    MixUp数据增强
    参考: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Args:
            batch: dict with 'images', 'labels', 'labels_type'
        Returns:
            mixed batch
        """
        if self.alpha <= 0:
            return batch
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch['labels'].size(0)
        index = torch.randperm(batch_size)
        
        mixed_batch = {
            'images': {},
            'labels': batch['labels'],
            'labels_mixed': batch['labels'][index],
            'labels_type': batch['labels_type'],
            'labels_type_mixed': batch['labels_type'][index],
            'lam': lam,
            'condition_ids': batch['condition_ids'],
        }
        
        # Mix images
        for key, img in batch['images'].items():
            if img is not None:
                mixed_batch['images'][key] = lam * img + (1 - lam) * img[index]
            else:
                mixed_batch['images'][key] = None
        
        return mixed_batch
