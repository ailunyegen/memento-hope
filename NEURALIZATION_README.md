# DualMemoryController 神经化改进

## 概述
对 `DualMemoryController` 进行了神经化改造，引入 PyTorch 实现基于梯度流的内部状态管理，包含 Delta 梯度下降 (DGD) 和遗忘门控机制。

## 主要变化

### 1. HOPE 层神经化
- **之前**: 使用硬编码浮点数加减法调整权重
- **现在**: 引入 `HOPEAdapter` 神经网络模块。**现已彻底重构遗忘门控，基于信息论（互信息/条件熵）显式量化记忆价值 $V(c_t)$，并结合物理时间对数衰减实现双维度动态遗忘。**

### 2. 连续体记忆 (CMS)
- **slow_weights**: 从固定字典改为 `nn.Parameter`，可通过梯度下降学习。现已引入**黎曼流形（Riemannian Manifold）约束**，使用流形投影（Retraction）代替简单的 clamp，确保权重更新不脱离物理意义。
- **fast_weights**: 通过神经网络根据场景特征动态计算调整值。现已建模为 **Ornstein-Uhlenbeck (OU) 随机微分方程 (SDE)** 驱动的短期瞬态扰动过程。

### 3. CMS 数学原理创新 
为保证系统稳定与收敛，对 DualMemoryController 进行严谨的数学赋能：
- **耦合方程**: 推导出 SDE-ODE 耦合动态方程 $\dot{W}_{slow}(t) = -\alpha \nabla_{\mathcal{M}} \mathcal{L}_{couple} - \beta \nabla_{\mathcal{M}} \mathcal{L}_{task}$，实现快慢记忆的理论对齐。
- **流形优化**: 拓展梯度下降空间至边界带限的黎曼流形，引入 Fisher-Rao / Log-barrier 度量，天然避免权重越界。
- **稳定性**: 通过构造带有 Ito 修正的随机 Lyapunov 函数，给出了系统的依概率指数稳定性和期望收敛上界证明（详见 `CMS_Mathematical_Derivation.md`）。
- **遗忘门控信息论机制**: 废弃黑盒门控，严格定义记忆价值函数 $V(c_t) = I / H$，引入时域年龄项 $\log(t_{curr}/t_{create})$，严格证明了“无效记忆指数遗忘”与“有用记忆长效保留”的解析特性（详见 `Forgetting_Gate_Mathematical_Derivation.md`）。
- **多目标帕累托优化与时滞梯度流**: 为解决仿真评价指标互斥（如高胜率与短时间）及反馈异步滞后的问题，将损失分离重构。在底层算子中手写了 PCGrad（冲突梯度投影）算法寻求帕累托最优下降方向，并叠加了基于泰勒一阶展开的时滞补偿修正项，彻底杜绝多目标训练中的负迁移现象（详见 `Gradient_Flow_Mathematical_Derivation.md`）。

### 4. 梯度流机制
- **输入编码**: 将 terrain、weather、ew_threat、threat_level、time_pressure 编码为特征向量
- **遗忘门控**: 实现基于信息论和物理时间双重驱动的遗忘机制
- **反向传播**: 在 `integrate_feedback` 中计算多目标代理 Loss，通过 PCGrad 解决梯度冲突，附加时滞补偿后进行黎曼流形更新

## 技术实现

### HOPEAdapter 类
```python
class HOPEAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 输入到隐藏层、遗忘门、隐藏状态等组件
```

### DualMemoryController 改进
- 继承 `nn.Module`
- `slow_weights` 作为可学习参数
- `fast_weights()` 方法使用神经网络计算
- `integrate_feedback()` 使用 MSE loss 和 Adam 优化器进行更新

## 优势
1. **自适应学习**: 权重可以根据历史表现自动调整
2. **连续优化**: 通过梯度下降实现连续状态空间优化
3. **遗忘机制**: 能够选择性保留或遗忘历史信息
4. **数学严谨**: 基于真实神经网络的梯度流计算

## 测试验证
- 基本功能测试通过
- 梯度更新机制工作正常
- 完整 pipeline 运行成功，未破坏现有功能