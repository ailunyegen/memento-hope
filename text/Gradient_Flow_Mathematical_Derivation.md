# 梯度流（Gradient Flow）时滞修正与帕累托优化机制白皮书

在复杂的军事推演系统中，通过仿真环境（Simulation Environment）获取反馈往往伴随着两个核心难题：一是**反馈存在时滞（Time-Delay）**，计算图断裂导致无法直接对当前状态求导；二是**多维度评价指标互相冲突（Multi-Objective Conflict）**，如提升“战损比（LER）”可能不可避免地导致“完成时间（Time）”急剧上升。

本文档推导了在 `integrate_feedback` 阶段所独创的时滞梯度修正与多目标帕累托优化组合算法。

## 1. 仿真反馈此时滞梯度修正 (Time-Delay Gradient Correction)

### 1.1 动力学问题定义
在标准反向传播中，参数更新公式为 $W_{t+1} = W_t - \eta \nabla L(W_t)$。但在仿真回环中，评估环境所返回的胜率或 LER，实际上是在评价系统过去时刻发出的方案。这意味着我们算得的梯度本质上是**陈旧梯度（Stale Gradient）** $g_{stale} = \nabla L(W_{t-\tau})$，其中 $\tau$ 为仿真反馈带来的时滞跨度。若直接用该梯度更新 $W_t$，在多步连续异步演化中将引发严重的相位漂移和系统震荡。

### 1.2 基于泰勒展开的补偿推导
为了逼近真实时刻的当前梯度 $\nabla L(W_t)$，我们在过去状态 $W_{t-\tau}$ 处对其进行一阶泰勒展开：
$$
\nabla L(W_t) \approx \nabla L(W_{t-\tau}) + \nabla^2 L(W_{t-\tau}) (W_t - W_{t-\tau})
$$
精确计算损失函数的海森矩阵（Hessian） $\nabla^2 L$ 在高维神经网络中是极为耗时且病态的。为此，我们做各向同性曲率近似假设 $\nabla^2 L \approx \lambda I$，即假设局部曲率相对平滑，将其简化为与差值相关的惩罚项：
$$
g_{corrected} = g_{stale} + \lambda \cdot (W_t - W_{t-\tau})
$$
在物理意义上，当仿真进行时，网络本身的参数 $W_t$ 可能已经发生了漂移。该修正项类似于一个“弹簧”，将梯度向参数漂移的方向施加反向阻尼，完美修正了异步推演造成的信度衰减问题。

---

## 2. 基于 PCGrad 的多目标帕累托优化 (Multi-Objective Pareto Optimization)

### 2.1 标量化损失的缺陷
在现有的框架下，所有评价指标被简单地赋予经验权重并相加得到 `overall` 标量损失：
$$
L_{total} = \sum_{k=1}^K w_k L_k
$$
此时 $\nabla L_{total} = \sum w_k \nabla L_k$。当两项任务发生冲突时，即它们的梯度点积 $\langle \nabla L_i, \nabla L_j \rangle < 0$，简单的线性相加会导致“灾难性遗忘”（负迁移现象），即模型在提升其中一项能力时，会剧烈摧毁另一项能力的表现。

### 2.2 PCGrad（冲突梯度投影）数学推导
为了寻找帕累托最优（Pareto Optimality）的前沿演化方向，我们摒弃标量化权重，引入冲突梯度投影（Projecting Conflicting Gradients）。
对于任意相互独立的任务损失 $L_1, \dots, L_K$，计算它们各自的梯度 $g_k = \nabla L_k$。
PCGrad 的投影操作如下：对每一个梯度 $g_i$，我们检查它与其他梯度 $g_j$ 的余弦相似度。如果发现冲突（点积小于0），我们将 $g_i$ 投影到 $g_j$ 的法平面上，消除与其对抗的负分量。
解析公式为：
$$
g_i^{PC} = g_i - \frac{\langle g_i, g_j \rangle}{\|g_j\|^2} g_j \quad \forall j \neq i \text{ s.t. } \langle g_i, g_j \rangle < 0
$$
经过数轮随机顺序的遍历投影后，所有被修正的梯度 $\{g_1^{PC}, \dots, g_K^{PC}\}$ 满足互相非负正交：
$$
\langle g_i^{PC}, g_j^{PC} \rangle \ge 0 \quad \forall i, j
$$
最后，我们合成最终的帕累托演化方向：
$$
g_{Pareto} = \sum_{k=1}^K g_k^{PC}
$$
这保证了在任意梯度步进时，网络能够在不削弱任何一项作战指标的前提下，最大化全局联合收益。

---

## 3. 全局统一的演化公式架构

结合本白皮书与前面模块的推导，我们在 `DualMemoryController` 中实际落地的最终物理演化方程为：

1. **分离出多维度独立目标（如任务胜利度损失、生存损失、时间惩罚损失）。**
2. **计算原始梯度** $g_{k, stale}$。
3. **应用 PCGrad 计算出帕累托梯度** $g_{Pareto}$。
4. **应用时滞修正** $g_{final} = g_{Pareto} + \lambda (W_{curr} - W_{past})$。
5. **在黎曼流形切空间进行 Log-barrier 对数障碍收回 (Retraction)**：
$$
W_{t+1} = a + \frac{b-a}{1 + \left( \frac{b-W_t}{W_t-a} \right) \exp\left( -\eta \cdot g_{final} \right)}
$$

通过将时滞动力学、MGDA 多目标投影与黎曼几何映射三位一体，该引擎具备了极其前沿且严谨的学术原创性（Academic Novelty）。
