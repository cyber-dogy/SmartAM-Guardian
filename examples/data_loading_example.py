"""
数据加载使用示例
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import SmartAMDataset, create_data_loaders
from data.dataset import split_by_condition
import matplotlib.pyplot as plt


def example_basic_loading():
    """基本数据加载示例"""
    print("=" * 60)
    print("示例1: 基本数据加载")
    print("=" * 60)
    
    # 直接创建DataLoader（自动划分）
    loaders = create_data_loaders(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        batch_size=4,
        num_workers=0,  # Windows建议用0
        img_size=224,
        modalities=['rgb_v1', 'rgb_v2', 'ir_before'],
        task='binary'
    )
    
    print(f"\nTrain batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # 获取一个batch
    batch = next(iter(loaders['train']))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Images keys: {batch['images'].keys()}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Condition IDs: {batch['condition_ids']}")
    
    return loaders


def example_manual_split():
    """手动划分数据集示例"""
    print("\n" + "=" * 60)
    print("示例2: 手动划分数据集")
    print("=" * 60)
    
    # 手动划分Condition
    train_conditions, val_conditions, test_conditions = split_by_condition(
        metadata_path="Causal_Image_Data/metadata.csv",
        val_ratio=0.15,
        random_seed=42
    )
    
    # 只加载特定condition
    dataset = SmartAMDataset(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        condition_ids=['Base-01', 'Base-02'],  # 只加载这两个condition
        modalities=['rgb_v1', 'ir_before'],
        task='multiclass'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Conditions: {dataset.conditions}")
    
    # 查看标签分布
    print(f"\nLabel distribution: {dataset.get_label_distribution()}")
    print(f"Role distribution: {dataset.get_role_distribution()}")
    
    # 获取单个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shapes:")
    for key, img in sample['images'].items():
        if img is not None:
            print(f"  {key}: {img.shape}")
    print(f"Label: {sample['label']}")
    print(f"Metadata: {sample['metadata']}")
    
    return dataset


def example_role_filter():
    """按角色过滤示例"""
    print("\n" + "=" * 60)
    print("示例3: 按角色过滤")
    print("=" * 60)
    
    # 只加载源域训练数据
    source_dataset = SmartAMDataset(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        role_filter=['source_train', 'source_repeat'],
        modalities=['rgb_v1'],
        task='binary'
    )
    
    print(f"Source train samples: {len(source_dataset)}")
    print(f"Role distribution: {source_dataset.get_role_distribution()}")
    
    # 只加载未见域测试数据
    unseen_dataset = SmartAMDataset(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        role_filter=['unseen_strategy_shift', 'unseen_texture_shift'],
        modalities=['rgb_v1'],
        task='binary'
    )
    
    print(f"\nUnseen test samples: {len(unseen_dataset)}")
    print(f"Conditions: {unseen_dataset.conditions}")
    
    return source_dataset, unseen_dataset


def example_visualization():
    """可视化示例"""
    print("\n" + "=" * 60)
    print("示例4: 可视化样本")
    print("=" * 60)
    
    # 创建数据集（无变换以获取原始图像）
    dataset = SmartAMDataset(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        condition_ids=['Base-01'],
        modalities=['rgb_v1', 'rgb_v2', 'ir_before', 'ir_after'],
        transform=None,  # 不应用变换
        load_images=True
    )
    
    # 获取第一个样本
    sample = dataset[0]
    
    # 创建图像展示
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    modality_names = {
        'rgb_v1': 'RGB View 1',
        'rgb_v2': 'RGB View 2',
        'ir_before': 'IR Before',
        'ir_after': 'IR After'
    }
    
    for idx, (key, img) in enumerate(sample['images'].items()):
        if img is not None:
            axes[idx].imshow(img)
            axes[idx].set_title(f"{modality_names[key]}\n{sample['condition_id']}-L{sample['metadata']['layer_id']:04d}")
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to sample_visualization.png")
    plt.close()


def example_statistics():
    """数据统计示例"""
    print("\n" + "=" * 60)
    print("示例5: 数据统计")
    print("=" * 60)
    
    dataset = SmartAMDataset(
        metadata_path="Causal_Image_Data/metadata.csv",
        data_root=".",
        load_images=False  # 不加载图像，只统计
    )
    
    print(f"\n总样本数: {len(dataset)}")
    print(f"Condition数量: {len(dataset.conditions)}")
    print(f"Conditions: {dataset.conditions}")
    
    print(f"\n标签分布:")
    for label, count in dataset.get_label_distribution().items():
        label_name = 'Normal' if label == 0 else 'Abnormal'
        print(f"  {label_name} ({label}): {count}")
    
    print(f"\n角色分布:")
    for role, count in dataset.get_role_distribution().items():
        print(f"  {role}: {count}")
    
    print(f"\n各Condition样本数:")
    cond_dist = dataset.get_condition_distribution()
    for cond, count in sorted(cond_dist.items()):
        print(f"  {cond}: {count}")


if __name__ == '__main__':
    # 运行所有示例
    try:
        example_basic_loading()
    except Exception as e:
        print(f"示例1失败: {e}")
    
    try:
        example_manual_split()
    except Exception as e:
        print(f"示例2失败: {e}")
    
    try:
        example_role_filter()
    except Exception as e:
        print(f"示例3失败: {e}")
    
    try:
        example_visualization()
    except Exception as e:
        print(f"示例4失败: {e}")
    
    try:
        example_statistics()
    except Exception as e:
        print(f"示例5失败: {e}")
