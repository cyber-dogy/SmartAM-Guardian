# SmartAM Guardian — 智能增材制造质量管控系统

## 项目背景

我正在准备2026年12月的校招，目标是AI应用方向（RAG/Agent/MLOps）。我的研究背景是增材制造（3D打印）多模态数据分析，希望结合这个背景做一个差异化的工业级AI项目。

**核心差异化**：基于已有SLM增材制造多模态异常识别数据集，构建端到端的AI应用系统，强调**因果推理**、**域泛化**和**可解释性**。

**项目目标**：构建一个工业级增材制造质量管控Agent系统，涵盖多模态异常检测、RAG知识库、Multi-Agent协作、MLOps管线和边缘设备部署。

---

## 数据集概况（已有）

**SLM增材制造多模态异常识别数据集**

```
📊 总样本数: ~300+ 层级样本
🎨 模态数: 4 (RGB_V1, RGB_V2, IR_Before, IR_After)
🔧 Condition数: 23 个不同工况组
🏷️ 标签: Binary (0/1) + Type (NOR/HEW/LEL)
⚡ 关键参数: P (功率), V (速度), S (间距), Ed (能量密度)
📁 划分单位: condition_id（不可拆分）
🎯 核心挑战: 域泛化、因果稳定性、多模态融合
```

**数据组织**：
```
metadata.csv              # 主元数据
├── condition_id          # 工况唯一标识
├── layer_id              # 层号
├── power_w               # 激光功率
├── speed_mms             # 扫描速度
├── spacing_mm            # 填充间距
├── energy_density        # 面能量密度 Ed = P/(V×S)
├── defect_label_binary   # 0=正常, 1=异常
├── defect_label_type     # NOR/HEW/LEL
└── role                  # 工况角色（划分依据）
```

**工况角色（数据划分策略）**：
- `source_train` / `source_repeat`：训练集基础分布
- `causal_intervention_high`：高能异常（HEW）
- `causal_intervention_low`：低能异常（LEL）
- `causal_equiv_test`：等效能量验证（Ed相同但参数不同）
- `unseen_strategy_shift`：未见扫描策略（Zigzag vs Stripe）
- `unseen_texture_shift`：未见纹理偏移（仅间距S变化）
- `unseen_param_shift` / `unseen_composite_shift`：复合偏移

**核心约束**：
- ⚠️ **必须按condition_id整体划分**，同一condition的所有layer必须在同一集合
- 层间存在时序相关性（热积累效应）
- 等效能量组用于验证模型是否理解物理机制而非记忆表面特征

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    SmartAM Guardian 架构                     │
├─────────────────────────────────────────────────────────────┤
│  数据采集层                                                   │
│  ├── RGB View 1 (CH1): 正面斜上方可见光                      │
│  ├── RGB View 2 (CH2): 侧后上方可见光                        │
│  ├── IR Before (CH3): 打印后制件表面热图                      │
│  └── IR After (CH3): 铺粉后粉层表面热图（激光干扰，慎用）      │
│                         ↓                                    │
│  多模态融合层 (Cross-Modal Attention / Feature Concat)        │
│                         ↓                                    │
│  缺陷检测模型 (基于Condition-level划分训练)                    │
│  ├── 二分类头: 正常 vs 异常                                   │
│  └── 三分类头: NOR vs HEW vs LEL                             │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Multi-Agent诊断系统                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │ 视觉    │  │ 工艺    │  │ 因果    │  │ 知识    │ │   │
│  │  │ Agent   │  │ Agent   │  │ Agent   │  │ Agent   │ │   │
│  │  │图像分析  │  │参数分析  │  │根因推理  │  │RAG检索  │ │   │
│  │  │缺陷定位  │  │Ed计算   │  │物理机制  │  │案例匹配  │ │   │
│  │  │置信度   │  │异常识别  │  │因果解释  │  │解决方案  │ │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │   │
│  │       └─────────────┴─────────────┴─────────────┘    │   │
│  │                     ↓                                 │   │
│  │              Orchestrator Agent                       │   │
│  │              (结果整合 + 报告生成 + 可解释性)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                    │
│  RAG知识库 (FAISS + BGE-small + Qwen2-1.5B-Chat)             │
│  ├── 工艺参数知识 (P/V/S/Ed关系)                              │
│  ├── 缺陷类型-成因-解决方案映射                                │
│  ├── 因果推理规则 (Ed阈值与缺陷类型关联)                        │
│  └── 历史案例库 (按condition索引)                            │
│                         ↓                                    │
│  输出:                                                        │
│  ├── 缺陷类型 + 置信度                                       │
│  ├── 根因分析 (物理机制解释)                                  │
│  ├── 调整建议 (基于因果推理)                                  │
│  └── 生成报告 (含可解释性说明)                                │
└─────────────────────────────────────────────────────────────┘
│                         ↓                                    │
┌─────────────────────────────────────────────────────────────┐
│  部署层                                                       │
│  ├── 云端: FastAPI + Docker + MLflow监控                     │
│  │          └── 域泛化测试 (unseen_* conditions)              │
│  └── 边缘: NVIDIA Jetson TX2 + TensorRT量化优化              │
│             └── 实时推理 (<100ms延迟)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心技术亮点

### 1. 因果推理能力
- **物理机制建模**：模型理解 Ed = P/(V×S) 与缺陷的因果关系
- **等效能量验证**：Ed相同但参数不同的组合应被识别为正常
- **反事实推理**："如果功率提高20W，结果会怎样"

### 2. 域泛化能力
- **未见策略迁移**：训练(Stripe) → 测试(Zigzag)
- **未见纹理判别**：识别仅S变化导致的视觉差异（非真实缺陷）
- **复合工况适应**：多维度参数同时变化的场景

### 3. 可解释性
- **Attention可视化**：多模态融合权重
- **因果链追溯**：从缺陷到根因的推理路径
- **物理一致性检查**：模型预测是否符合物理规律

---

## 技术栈

| 层级 | 技术 | 说明 |
|-----|------|------|
| 数据处理 | Pandas, PyTorch Dataset | 按condition划分的数据加载器 |
| 多模态融合 | Cross-Attention / Concat + MLP | 三模态输入（V1+V2+IR_Before） |
| 骨干网络 | EfficientNet-B0 / ResNet18 | 轻量级，适合TX2 |
| 域泛化 | Mixup / CutMix / Domain Generalization | 提升泛化能力 |
| 因果学习 | Causal Inference / SCM | 结构因果模型 |
| Embedding | BGE-small | 工艺知识编码 |
| 向量数据库 | FAISS (HNSW) | 快速检索 |
| LLM | Qwen2-1.5B-Chat | 本地部署 |
| RAG框架 | LlamaIndex | 知识库管理 |
| Agent框架 | AutoGen / LangGraph | Multi-Agent编排 |
| API服务 | FastAPI | RESTful API |
| 前端界面 | Streamlit | 可视化演示 |
| 推理优化 | ONNX Runtime, TensorRT, INT8 | TX2边缘部署 |
| MLOps | MLflow, DVC, Prometheus | 实验跟踪与监控 |
| 部署 | Docker, Docker Compose | 容器化 |

---

## 90天执行计划（基于已有数据集）

### Phase 1: 基础模型与RAG（第1-30天）

**目标**：训练多模态异常检测模型，搭建基础RAG系统

**Week 1: 数据工程（已有数据集接入）**
- Day 1-2: 项目初始化，GitHub仓库搭建
- Day 3-4: 数据加载器实现（按condition_id划分）
- Day 5-6: 数据增强与预处理（归一化、Resize）
- Day 7: 数据集划分验证（确保condition-level正确）

**Week 2: 多模态模型**
- Day 8-10: 多模态融合网络搭建（3模态输入）
- Day 11-12: 双任务头（二分类 + 三分类）
- Day 13-14: 基础训练循环实现

**Week 3: 模型训练与优化**
- Day 15-17: 基线模型训练（ResNet18 + Concat）
- Day 18-19: 域泛化技术（Mixup/CutMix）
- Day 20-21: Attention-based融合实验

**Week 4: RAG知识库搭建**
- Day 22-24: 工艺知识库构建（YAML格式）
- Day 25-26: FAISS + BGE-small集成
- Day 27-28: Qwen2本地部署与Prompt工程
- Day 29-30: 基础RAG Demo + 测试

**阶段产出**：
- 多模态异常检测模型（支持二分类/三分类）
- 基础RAG问答系统
- 技术博客 × 2篇

---

### Phase 2: Agent系统与因果推理（第31-60天）

**目标**：构建Multi-Agent诊断系统，实现因果推理能力

**Week 5: 单Agent实现**
- Day 31-33: 视觉Agent（缺陷定位 + Attention可视化）
- Day 34-35: 工艺Agent（Ed计算 + 参数异常识别）
- Day 36-37: 知识Agent（RAG检索 + 案例匹配）

**Week 6: 因果推理Agent**
- Day 38-40: 因果规则引擎（Ed阈值 → 缺陷类型映射）
- Day 41-42: 反事实推理实现
- Day 43-44: 因果一致性检查（等效能量验证）

**Week 7: Multi-Agent编排**
- Day 45-47: AutoGen框架集成
- Day 48-49: Orchestrator Agent设计
- Day 50-51: Agent间通信与结果整合

**Week 8: 域泛化测试**
- Day 52-54: unseen_strategy_shift测试
- Day 55-56: unseen_texture_shift测试
- Day 57-58: unseen_composite_shift测试
- Day 59-60: 因果稳定性评估（causal_equiv_test）

**阶段产出**：
- Multi-Agent协作系统（含因果推理）
- 诊断报告自动生成（含可解释性）
- 域泛化测试报告
- 技术博客 × 2篇

---

### Phase 3: MLOps与监控（第61-75天）

**目标**：工程化 + 可维护性

**Week 9-10: MLOps管线**
- Day 61-63: MLflow实验跟踪（记录condition-level指标）
- Day 64-65: DVC数据版本管理
- Day 66-67: 模型版本管理与自动测试
- Day 68-70: 监控Dashboard（Prometheus + Grafana）

**Week 11: 反馈闭环**
- Day 71-73: 用户反馈收集机制
- Day 74-75: 知识库自动更新流程

**阶段产出**：
- MLOps完整管线
- 监控Dashboard
- 技术博客 × 1篇

---

### Phase 4: 产品化与部署（第76-90天）

**目标**：可演示的完整产品

**Week 12: Web界面**
- Day 76-78: FastAPI后端API
- Day 79-81: Streamlit前端（实时诊断界面）
- Day 82-84: 可视化功能（Attention热力图、因果链展示）

**Week 13: TX2边缘部署**
- Day 85-87: 模型量化（INT8）+ TensorRT转换
- Day 88-90: TX2部署优化（<100ms延迟目标）

**Week 14-15: 文档与展示**
- 项目README
- 技术文档（重点强调因果推理和域泛化）
- 演示视频
- 面试话术（准备回答"如何证明模型理解物理机制"）

**阶段产出**：
- 完整Web应用
- TX2部署版本
- 项目文档
- 技术博客 × 1篇

---

## 简历描述模板

**项目名称**：SmartAM Guardian — 智能增材制造质量管控系统

**要点**：
- 基于多模态数据（可见光+红外+工艺参数）构建增材制造实时缺陷诊断系统，处理23种工况、300+样本的工业级数据集
- 设计Multi-Agent协作架构（视觉/工艺/因果/知识Agent），实现端到端的质量诊断与根因分析
- 引入因果推理机制，通过等效能量验证确保模型理解物理机制（Ed = P/(V×S)），而非仅记忆表面特征
- 实现域泛化能力，模型在未见扫描策略（Zigzag vs Stripe）和未见纹理条件下仍保持高准确率
- 搭建完整MLOps管线（MLflow + DVC + Prometheus），支持condition-level实验跟踪与模型版本管理
- 优化并部署至NVIDIA Jetson TX2边缘设备，实现<100ms延迟的实时推理
- 技术栈：PyTorch/ONNX/TensorRT, Cross-Modal Attention, LlamaIndex/AutoGen, FastAPI/Streamlit, MLflow/Docker

---

## 面试重点准备

**高频问题**：

1. **"如何证明模型理解物理机制而非仅记忆？"**
   - 答：等效能量验证（causal_equiv_test）——Ed相同但参数不同的组合被正确识别为正常

2. **"如何处理域迁移问题？"**
   - 答：condition-level划分 + Mixup/CutMix + 未见工况测试（unseen_* roles）

3. **"RAG在这个场景中的作用？"**
   - 答：工艺知识库支持根因分析，案例匹配提供调整建议

4. **"因果Agent具体怎么实现？"**
   - 答：基于Ed阈值的规则引擎 + 反事实推理 + 物理一致性检查

---

## 当前状态

**我正在执行**：Phase 1 / Phase 2 / Phase 3 / Phase 4（请勾选）

**当前遇到的具体问题**：
（在此处描述）

**今天的目标**：
（描述今天希望完成的任务）
