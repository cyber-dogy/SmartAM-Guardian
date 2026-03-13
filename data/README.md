# SmartAM 数据加载模块

针对SLM增材制造多模态异常识别数据集的 PyTorch 数据加载实现。

## 特性

- **按 Condition 划分**: 严格遵守同一 Condition 不可拆分的约束
- **多模态支持**: RGB_V1, RGB_V2, IR_Before, IR_After 灵活组合
- **双任务支持**: 二分类 (0/1) 和多分类 (NOR/HEW/LEL)
- **数据增强**: 针对 RGB 和 IR 分别优化的增强策略
- **角色过滤**: 支持按 source/unseen/causal 等角色筛选数据

## 快速开始

### 1. 自动创建 DataLoader

```python
from data import create_data_loaders

loaders = create_data_loaders(
    metadata_path="Causal_Image_Data/metadata.csv",
    data_root=".",
    batch_size=16,
    num_workers=4,
    img_size=224,
    modalities=['rgb_v1', 'rgb_v2', 'ir_before'],  # 选择使用的模态
    task='binary'  # 或 'multiclass'
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### 2. 手动划分 Condition

```python
from data.dataset import split_by_condition

train_conditions, val_conditions, test_conditions = split_by_condition(
    metadata_path="Causal_Image_Data/metadata.csv",
    val_ratio=0.1,  # 10% 的训练 condition 用于验证
    random_seed=42
)
```

### 3. 自定义 Dataset

```python
from data import SmartAMDataset

dataset = SmartAMDataset(
    metadata_path="Causal_Image_Data/metadata.csv",
    data_root=".",
    condition_ids=['Base-01', 'High-02'],  # 只加载特定 condition
    role_filter=['source_train', 'source_repeat'],  # 或按角色过滤
    modalities=['rgb_v1', 'ir_before'],
    task='binary',
    transform=None  # 自定义变换
)
```

## 数据划分策略

| 数据集 | 来源 | 用途 |
|--------|------|------|
| 训练集 | `source_train`, `source_repeat`, `causal_intervention_*` | 训练模型 |
| 验证集 | 从训练 condition 中随机抽取 | 调参和早停 |
| 测试集 | `unseen_*`, `causal_equiv_test` | 域泛化测试 |

**重要**: 划分单位为 `condition_id`，同一 condition 的所有 layer 必须在同一集合。

## 模态说明

| 模态键 | 文件名 | 说明 |
|--------|--------|------|
| `rgb_v1` | `rgb_view1.jpg` | 正面斜上方可见光 (推荐) |
| `rgb_v2` | `rgb_view2.jpg` | 侧后上方可见光 |
| `ir_before` | `ir_before.jpg` | 打印后制件红外图 |
| `ir_after` | `ir_after.jpg` | 铺粉后粉层红外图 (慎用) |

## 使用示例

详见 `examples/data_loading_example.py`:

```bash
cd examples
python data_loading_example.py
```

包含以下示例:
1. 基本数据加载
2. 手动划分数据集
3. 按角色过滤
4. 可视化样本
5. 数据统计

## 返回格式

```python
sample = dataset[0]
# sample 包含:
{
    'images': {
        'rgb_v1': tensor([3, 224, 224]),
        'rgb_v2': tensor([3, 224, 224]),
        'ir_before': tensor([3, 224, 224]),
    },
    'label': tensor(0),  # 0=正常, 1=异常
    'label_type': tensor(0),  # 0=NOR, 1=HEW, 2=LEL (multiclass任务)
    'condition_id': 'Base-01',
    'metadata': {
        'condition_id': 'Base-01',
        'layer_id': 1,
        'power_w': 120,
        'speed_mms': 1000,
        'spacing_mm': 0.08,
        'scan_strategy': 'Stripe',
        'energy_density': 1.5,
        'role': 'source_train',
    }
}
```
