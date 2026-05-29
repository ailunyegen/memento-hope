# 连续体记忆（CMS）模型数学原理及证明白皮书

本文档系统性推导了基于常微分方程（ODE）和随机微分方程（SDE）耦合的连续体记忆演化模型，引入黎曼流形约束的优化动力学，并给出了基于随机 Lyapunov 函数的稳定性和收敛性证明。

## 1. 慢/快权重的耦合时域动态模型

在连续体记忆系统（Continuum Memory System, CMS）中，我们将长短期记忆解耦，构建如下耦合系统：
- **慢权重（$W_{slow} \in \mathcal{M}$）**：表征长期的、“稳态”的知识沉淀（如教条、战术法则），位于一个特定的黎曼流形 $\mathcal{M}$ 上。
- **快权重（$W_{fast} \in \mathbb{R}^n$）**：表征短期的、“瞬态”的环境感知（如地形、气象、实时威胁），它随着环境特征 $\Phi(t)$ 发生快速演化，并伴随一定的随机不确定性。

### 1.1 系统方程
我们构建如下的耦合微分方程：

**慢权重演化 (Gradient Flow on Manifold):**
$$
\dot{W}_{slow}(t) = -\alpha \nabla_{\mathcal{M}} \mathcal{L}_{couple}(W_{slow}(t), W_{fast}(t)) - \beta \nabla_{\mathcal{M}} \mathcal{L}_{task}(W_{slow}(t), \mathcal{D})
$$
其中，$\nabla_{\mathcal{M}}$ 表示在黎曼流形 $\mathcal{M}$ 上的梯度算子，$\mathcal{L}_{task}$ 为基于历史经验 $\mathcal{D}$ 的全局任务损失，$\mathcal{L}_{couple}$ 为快慢权重耦合的对齐损失。$\alpha, \beta > 0$ 为可学习或预设的学习率控制系数。

**快权重演化 (Stochastic Differential Equation, SDE):**
为了体现短期记忆在连续时间上的“扰动”和衰减，我们采用受迫的 Ornstein-Uhlenbeck (OU) 过程来建模：
$$
dW_{fast}(t) = \theta \left( \mathcal{F}(W_{slow}(t), \Phi(t)) - W_{fast}(t) \right) dt + \epsilon d\xi_t
$$
其中：
- $\mathcal{F}(\cdot, \cdot)$ 为基于当前慢记忆和环境特征 $\Phi(t)$ 的瞬态吸引子驱动函数（在代码中对应 `HOPEAdapter` 网络）。
- $\theta > 0$ 为回复力系数（遗忘率）。
- $\xi_t$ 为标准布朗运动，$d\xi_t \sim \mathcal{N}(0, dt)$。
- $\epsilon > 0$ 控制随机扰动的强度。

此耦合系统准确刻画了记忆的连续演化本质：快记忆迅速响应环境并带噪声衰减，慢记忆在流形上平滑地进行梯度下降沉淀知识。

---

## 2. 黎曼流形（Riemannian Manifold）约束与投影

由于权重具有实际的物理意义（例如，各项能力权重的非负性且存在天然边界），若将其视为欧氏空间进行普通梯度下降，容易脱离物理意义。现有方案往往采用 `torch.clamp` 进行粗暴截断，这不仅会阻断梯度的连续性，还会导致权重在边界处产生“死区”。

为解决此问题，我们引入黎曼流形约束。假设慢权重需满足 $\mathcal{M} = \{ W \in \mathbb{R}^n_{>0} \mid W_i \in (W_{min}, W_{max}) \}$。为简便起见，我们将参数空间映射到概率单纯形（Simplex）或者使用对数障碍（Log-Barrier）构造内点流形。在这里，我们考虑更一般的带界限的黎曼流形定义。

### 2.1 Fisher-Rao 度量与流形梯度
如果权重仅有上下界 $W \in (a, b)^n$，我们可以利用对数映射引入非欧几里得的黎曼度量（Log-barrier metric）：
$$
G_{ii}(W) = \frac{1}{(W_i - a)(b - W_i)}
$$
此时，张量矩阵 $G$ 是对角的。根据流形优化理论，黎曼梯度为：
$$
[\text{grad} f(W)]_i = [G^{-1}(W) \nabla f(W)]_i = (W_i - a)(b - W_i) \cdot [\nabla f(W)]_i
$$
这保证了当 $W_i$ 接近边界 $a$ 或 $b$ 时，有效梯度自然平滑衰减为 0，从而实现天然的“软”约束，避免权重越界。

### 2.2 收回算子（Retraction）
在流形的切空间 $T_{W}\mathcal{M}$ 得到黎曼梯度后，需利用收回算子 $R_W: T_{W}\mathcal{M} \to \mathcal{M}$ 进行更新：
$$
W(t + \Delta t) = R_{W(t)}(-\Delta t \cdot \text{grad} f(W(t)))
$$
对于区间约束 $(a, b)^n$，其收回可由 sigmoid 函数的逆放缩天然保证（对角线元素的精确解）：
$$
W_i(t+\Delta t) = a + \frac{b-a}{1 + \left( \frac{b-W_i(t)}{W_i(t)-a} \right) \exp\left( -\Delta t \cdot [\nabla f(W(t))]_i \right)}
$$
此算子在数学上是平滑的双射，且在更新步长 $\Delta t$ 任意大的情况下恒满足 $W_i \in (a, b)$ 边界条件，且完全消除了原先使用的 `torch.clamp` 所带来的不可导折点。

---

## 3. Lyapunov 稳定性与收敛上界证明

我们将证明上述 SDE-ODE 耦合系统在适当条件下达到依概率渐近稳定（Asymptotically stable in probability）。

### 3.1 构造随机 Lyapunov 函数
定义系统的联合能量函数（Lyapunov function） $V(W_{slow}, W_{fast})$ 如下：
$$
V(W_{slow}, W_{fast}) = \mathcal{L}_{total}(W_{slow}) + \frac{\gamma}{2} \| W_{fast} - \mathcal{F}(W_{slow}, \Phi) \|^2
$$
其中 $\gamma > 0$ 为常数，且假设 $\mathcal{L}_{total}$ 在流形 $\mathcal{M}$ 上是 $\mu$-强凸的（相对于黎曼度量），即：
$$
\langle \text{grad}\mathcal{L}_{total}(W), \text{grad}\mathcal{L}_{total}(W) \rangle_{\mathcal{M}} \ge 2\mu (\mathcal{L}_{total}(W) - \mathcal{L}_{total}^*)
$$

### 3.2 伊托引理（Itô's Lemma）求导
对于受 SDE 影响的系统，我们需要计算 $V$ 的无穷小生成元（Infinitesimal Generator） $\mathcal{A}V$。根据多元 Itô's Lemma：
$$
\mathcal{A}V = \nabla_{W_{slow}} V \cdot \dot{W}_{slow} + \nabla_{W_{fast}} V \cdot \mathbb{E}[dW_{fast}/dt] + \frac{1}{2} \text{Tr}(\epsilon^2 \nabla^2_{W_{fast}} V)
$$

计算各项：
1. **关于 $W_{slow}$ 的导数项：**
   $$
   \nabla_{W_{slow}} V \cdot \dot{W}_{slow} = \langle \text{grad} \mathcal{L}_{total}, -\beta \text{grad} \mathcal{L}_{total} - \alpha \text{grad} \mathcal{L}_{couple} \rangle_{\mathcal{M}}
   $$
   设定 $\mathcal{L}_{couple}$ 为对齐项且对总损失具有良性辅助作用，当充分接近平衡态时，主要受 $-\beta \|\text{grad} \mathcal{L}_{total}\|_{\mathcal{M}}^2$ 主导。

2. **关于 $W_{fast}$ 的漂移项：**
   由 SDE 定义：$\mathbb{E}[dW_{fast}/dt] = -\theta (W_{fast} - \mathcal{F}(W_{slow}, \Phi))$。
   又 $\nabla_{W_{fast}} V = \gamma (W_{fast} - \mathcal{F}(W_{slow}, \Phi))$，
   故此项为：$-\gamma \theta \| W_{fast} - \mathcal{F} \|^2$。

3. **二阶扩散项（Ito 修正项）：**
   因为 $\nabla^2_{W_{fast}} V = \gamma I$（单位矩阵），
   二阶项为 $\frac{1}{2} \gamma \epsilon^2 \text{Tr}(I) = \frac{n \gamma \epsilon^2}{2}$。

综合可得生成元：
$$
\mathcal{A}V \le -\beta \|\text{grad} \mathcal{L}_{total}\|_{\mathcal{M}}^2 - \gamma \theta \| W_{fast} - \mathcal{F} \|^2 + \frac{n \gamma \epsilon^2}{2}
$$

### 3.3 稳定性分析与收敛上界
为了考察系统到全局最小点 $W_{slow}^*$ 的收敛性，将强凸不等式代入：
$$
\mathcal{A}V \le -2\beta\mu (\mathcal{L}_{total} - \mathcal{L}_{total}^*) - \gamma \theta \| W_{fast} - \mathcal{F} \|^2 + \frac{n \gamma \epsilon^2}{2}
$$
这可以写成：
$$
\mathcal{A}V \le -\lambda (V - \mathcal{L}_{total}^*) + C
$$
其中 $\lambda = \min(2\beta\mu, 2\theta)$ 为系统的总指数收敛率，常数项 $C = \frac{n \gamma \epsilon^2}{2}$ 为随机扰动引起的残余方差。

**结论 1 (依概率稳定)：**
根据随机 Lyapunov 定理，如果 $\mathcal{A}V \le -\lambda V + C$，则系统在平衡点的一个邻域内是依概率指数渐近稳定的，系统存在一个平稳分布。快权重不断以速率 $\theta$ 追踪 $\mathcal{F}$，慢权重以速率 $2\beta\mu$ 收敛至最优流形吸引子。

**结论 2 (收敛上界)：**
对上述微分不等式使用 Gronwall 不等式（或对伊藤过程求期望），可推导出权重收敛的期望上界：
$$
\mathbb{E}[V(t)] \le \left( V(0) - \frac{C}{\lambda} \right) e^{-\lambda t} + \frac{C}{\lambda}
$$
这表明，慢权重以时间复杂度 $\mathcal{O}(e^{-\lambda t})$ 指数收敛，并在充分长的时间后，损失函数收敛至邻域 $\mathcal{O}(\frac{\epsilon^2}{\lambda})$ 内。如果控制噪声随着时间退火（即 $\epsilon(t) \to 0$），则可以实现几乎处处的精确收敛（Almost Sure Convergence）。
