# SmartAM Guardian — 压缩版项目提示词（适合校招落地）

## 项目定位

我正在准备 2026 年 12 月校招，目标岗位是 **AI 应用 / RAG / Agent / MLOps**。
我的研究背景是 **SLM 增材制造多模态异常识别**，希望基于已有真实数据集，做一个 **可演示、可解释、能讲清技术价值** 的工业级 AI 项目。

这个项目不追求“大而全”，而是强调：

1. **真实工业问题**：基于 SLM 多模态数据进行异常诊断
2. **正确评估方式**：严格按 `condition_id` 划分，避免数据泄漏
3. **物理一致性验证**：利用 `Ed = P / (V × S)` 做等效能量与工艺一致性分析
4. **知识增强诊断**：结合 RAG/规则引擎输出根因分析与调整建议
5. **工程化落地**：包含 API、前端演示、实验跟踪和轻量部署

---

## 一、项目背景

### 数据集概况

已有一个 **SLM 增材制造多模态异常识别数据集**，包含约 300+ 层级样本，4 种模态：

- `RGB_V1`
- `RGB_V2`
- `IR_Before`
- `IR_After`（慎用，可作为补充，不作为首版核心输入）

元数据包括：

- `condition_id`
- `layer_id`
- `power_w`
- `speed_mms`
- `spacing_mm`
- `energy_density`
- `defect_label_binary`
- `defect_label_type`
- `role`

其中 `role` 用于表示工况性质，包括：

- `source_train`
- `source_repeat`
- `causal_intervention_high`
- `causal_intervention_low`
- `causal_equiv_test`
- `unseen_strategy_shift`
- `unseen_texture_shift`
- `unseen_param_shift`
- `unseen_composite_shift`

### 核心约束

- **必须按 `condition_id` 整体划分**
- 同一 `condition` 下各层之间存在热积累与纹理相似性，不能随机拆分
- `causal_equiv_test` 用于验证模型是否关注物理机制而非表面纹理
- 项目目标是做一个 **校招强叙事项目**，而不是单纯追求学术最优精度

---

## 二、压缩后的项目目标

构建一个 **工业多模态质量诊断系统**，具备以下能力：

### 1. 多模态异常检测

输入多模态图像和工艺参数，输出：

- 正常 / 异常（二分类）
- 缺陷类型（可扩展到 NOR / HEW / LEL）

### 2. 物理一致性验证

围绕 `Ed = P / (V × S)` 构建验证机制：

- 等效能量组的一致性分析
- 参数变化与风险方向的一致性检查
- 证明模型并非只记忆纹理特征

### 3. 知识增强诊断

基于工艺知识库和历史案例，实现：

- 根因分析
- 类似案例检索
- 调整建议生成

### 4. 工程化演示

形成一个可展示的完整系统，包括：

- 模型服务 API
- Web 可视化页面
- 实验跟踪
- 轻量推理部署

---

## 三、压缩后的系统定位

该项目不是“大而全”的多智能体平台，而是一个：

> **以多模态异常检测为核心，以物理一致性验证为亮点，以知识增强报告为应用价值的工业 AI 诊断系统**

其中：

- **核心主线**：多模态检测 + condition-level 泛化评估
- **亮点增强**：工艺规则 + RAG + 报告生成
- **工程加分项**：FastAPI + MLflow + Docker + ONNX/TensorRT

---

## 四、技术架构（压缩版）

### 1. 输入层

- RGB_V1
- RGB_V2
- IR_Before
- 工艺参数：P / V / S / Ed

### 2. 模型层

- Backbone：ResNet18 或 EfficientNet-B0
- 多模态融合：先用 `feature concat + MLP`
- 首版任务：二分类
- 扩展任务：三分类（NOR / HEW / LEL）

### 3. 规则与知识层

- 工艺规则：Ed、P、V、S 合理性检查
- 缺陷知识：缺陷类型 → 可能机理 → 调整建议
- 案例库：按 condition 检索类似样本

### 4. 诊断编排层

- Visual Analyzer：输出异常概率与视觉证据
- Process Analyzer：输出工艺风险与参数解释
- Knowledge Retriever：输出相关案例与建议
- Report Generator：生成最终诊断报告

### 5. 工程与部署层

- 模型训练：PyTorch
- 向量检索：FAISS
- LLM：Qwen2-1.5B / API LLM（二选一）
- 服务：FastAPI
- 前端：Streamlit
- 实验跟踪：MLflow
- 部署：Docker / ONNX Runtime / TensorRT（可选）

---

## 五、最小可行版本（MVP）

为了在 **30 天左右做出可展示项目**，首版必须收敛为：

### 必做

1. condition-level 数据划分与验证
2. 三模态输入数据加载器
3. 一个可靠 baseline：
   - ResNet18
   - feature concat
   - binary classification
4. unseen role 测试
5. Ed 工艺规则模块
6. 基础 RAG/规则知识库
7. FastAPI + Streamlit Demo
8. README + 演示材料

### 选做

1. Attention 融合
2. 三分类任务
3. Agent 框架化
4. MLflow 完整接入
5. ONNX / TensorRT
6. Jetson TX2 实测

---

## 六、项目亮点表达（校招版）

### 亮点 1：正确的数据划分

严格按 `condition_id` 划分训练/验证/测试集，避免层间泄漏，保证模型评估更接近真实工业部署场景。

### 亮点 2：不只做分类，还验证“物理一致性”

通过等效能量组与参数变化分析，验证模型是否真正捕捉到工艺与缺陷风险之间的关系，而非仅依赖视觉纹理。

### 亮点 3：从“预测结果”升级为“诊断报告”

结合工艺知识库、缺陷机理和历史案例，实现异常类型判断、根因分析与参数调整建议生成。

### 亮点 4：具备完整应用工程链路

覆盖模型训练、知识检索、服务封装、Web 演示和轻量部署，贴近 AI 应用 / RAG / Agent / MLOps 岗位需求。

---

## 七、面试叙事主线

面试时应重点强调：

### 1. 为什么不能随机划分 layer？

因为同一 condition 下层间具有强相关性，随机划分会导致信息泄漏，离线分数虚高，但泛化能力不真实。

### 2. 如何证明模型不是在记表面纹理？

通过 `causal_equiv_test`、`unseen_texture_shift` 和参数一致性分析，观察模型在等效能量与纹理变化场景下的稳定性。

### 3. 为什么需要知识库 / RAG？

分类模型只能输出“是什么”，知识库可以补充“为什么”和“怎么调”，更符合工业诊断系统的使用方式。

### 4. 为什么这个项目适合 AI 应用岗位？

因为它不仅有模型，还有检索增强、服务化封装、评估体系和工程部署，体现的是完整 AI 系统能力，而不是单点算法实验。

---

## 八、你可以直接复用的“项目要求提示词”

请基于以下要求帮助我继续设计和实现项目，输出内容优先面向校招落地，而非纯学术论文：

### 项目名称

SmartAM Guardian — 智能增材制造质量诊断系统

### 项目目标

围绕已有 SLM 多模态异常识别数据集，构建一个可演示、可解释、可工程化落地的工业 AI 应用项目，用于校招展示。

### 核心要求

1. 必须按 `condition_id` 划分数据，严禁 layer 随机拆分
2. 首版以 **30 天内可落地** 为目标，只保留必要模块
3. 核心主线应是：
   - 多模态异常检测
   - condition-level 泛化验证
   - 基于 Ed 的物理一致性分析
   - 知识增强诊断报告
4. 项目叙事必须适合 AI 应用 / RAG / Agent / MLOps 岗位面试
5. 结果表达要强调：
   - 工业真实问题
   - 正确评估方式
   - 物理一致性而非纯黑盒分类
   - 系统工程能力
6. 避免夸大“强因果推理”，更合适的表述包括：
   - physics-informed reasoning
   - 物理一致性验证
   - 因果启发式诊断
   - 基于结构先验的根因分析

### 首版建议技术栈

- PyTorch
- ResNet18 / EfficientNet-B0
- feature concat + MLP
- FAISS
- FastAPI
- Streamlit
- MLflow（可后置）
- Docker
- ONNX Runtime（可后置）

### 首版交付要求

1. 训练好的 baseline 模型
2. unseen condition 测试结果
3. 物理一致性验证示例
4. 知识增强诊断 Demo
5. Web 演示界面
6. README 首页
7. 面试讲解材料

### 输出风格要求

- 以工程落地和校招表达为主
- 不要把项目写成过于庞杂的研究平台
- 保持技术可信，不夸张
- 优先给出可执行方案、目录结构、代码骨架、评估设计、README 文案和面试表述
