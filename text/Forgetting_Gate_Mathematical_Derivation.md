# 遗忘门控（Forgetting Gating）信息论机制证明白皮书

在连续体记忆模块（CMS）中，原有的遗忘门控（基于类似 LSTM 的全连接层加 Sigmoid 激活）仅仅是一个黑盒式的经验结构，缺乏对其“应该遗忘什么”的显式数学定义。本文档推导了基于信息论（互信息与条件熵）的记忆价值评价函数，重构了遗忘门控公式，并给出了其物理遗忘特性的解析证明。

## 1. 记忆价值函数定义

我们定义隐状态（记忆单元）在时刻 $t$ 的内容为 $c_t$。过去所有的观测特征序列为 $\mathcal{X}_{1:t}$，未来时刻需要预测的决策目标或场景演化为 $\mathcal{Y}_{t+1}$。

我们通过信息论来量化 $c_t$ 的“记忆价值” $V(c_t)$。一个好的记忆应当具备两个特点：
1. **高预测能力**：它必须包含关于未来的有用信息。
2. **低冗余度**：它的编码应该尽量精简，占用最少的有效状态空间。

因此，定义价值量化公式如下：
$$
V(c_t) = \frac{I(c_t; \mathcal{Y}_{t+1} \mid \mathcal{X}_{1:t})}{H(c_t \mid \mathcal{X}_{1:t})}
$$
- $I(c_t; \mathcal{Y}_{t+1} \mid \mathcal{X}_{1:t})$：**条件互信息 (Conditional Mutual Information)**。它表示在已知过去观测的前提下，$c_t$ 能够为预测未来 $\mathcal{Y}_{t+1}$ 额外消除的不确定性（即提供的新增预测比特数）。
- $H(c_t \mid \mathcal{X}_{1:t})$：**条件熵 (Conditional Entropy)**。它表示该记忆本身的状态复杂度或冗余度。

该公式构成了记忆的“信息投入产出比”（单位比特的预测增益），以此作为衡量记忆是否值得保留的绝对基准。

---

## 2. 基于价值函数的遗忘门控重构

原有的 LSTM 遗忘门公式为 $f_t = \sigma(W_{fx} x_t + W_{fh} h_{t-1} + b_f)$，其门控系数的物理意义模糊。我们将其重构为受“记忆价值”和“物理时间”双维度驱动的形式：

$$
f_t = \sigma \left( \gamma \cdot V(c_t) + \delta \cdot \log\left(\frac{t_{current}}{t_{create}}\right) + \beta \right)
$$

其中：
- $\sigma(\cdot)$ 为 Sigmoid 函数，将输出约束在 $(0, 1)$，代表保留比例。
- $t_{create}$ 为该特定维度记忆特征被激活创造的时间戳；$t_{current}$ 为当前仿真时间步。
- $\log(t_{current} / t_{create})$ 为时间衰减项，表征记忆随时间流逝的自然老化（由于取对数，它符合生物心理学中的 Ebbinghaus 遗忘曲线假说）。
- $\gamma > 0$ 为价值灵敏度系数。
- $\delta < 0$ 为时间衰减率常数。
- $\beta$ 为基础保留偏置。

此公式将网络从“黑盒参数拟合”变为了“显式规则物理约束”的学习。

---

## 3. 遗忘特性的数学验证与解析解

我们需要证明：此门控机制能够满足“有用记忆保留、无效记忆指数遗忘”的特性。

考虑离散时间步下的记忆传递方程：
$$
c_{t+1} = f_t \cdot c_t
$$
设某维度的记忆寿命 $\tau = t_{current} - t_{create}$，且 $\tau \ge 1$。

### 3.1 情况一：无效记忆的指数衰减 (Invalid Memory)
当某项记忆对未来预测毫无帮助时，互信息 $I \to 0$，从而 $V(c_t) \to 0$。
代入遗忘门公式：
$$
f_\tau = \sigma \left( \delta \log \tau + \beta \right) = \frac{1}{1 + \exp(-\delta \log \tau - \beta)}
$$
由于 $\delta < 0$，随着时间 $\tau$ 增大，$-\delta \log \tau \to +\infty$，此时可采用泰勒级数近似 $\sigma(x) \approx e^x$ (当 $x \ll 0$)：
$$
f_\tau \approx \exp(\delta \log \tau + \beta) = e^\beta \cdot \tau^\delta
$$

经过 $T$ 步的时间演化，总的记忆保留量 $C_T$ 为：
$$
C_T = c_0 \prod_{\tau=1}^T f_\tau \approx c_0 \cdot \prod_{\tau=1}^T \left( e^\beta \cdot \tau^\delta \right) = c_0 \cdot e^{\beta T} \cdot (T!)^\delta
$$
根据 Stirling 近似公式 $T! \sim \sqrt{2\pi T} (T/e)^T$，代入上式得：
$$
C_T \propto e^{\beta T} \cdot \left( \frac{T}{e} \right)^{\delta T} = \exp\left( (\beta - \delta) T + \delta T \log T \right)
$$
由于 $\delta < 0$，$\delta T \log T \to -\infty$ 的速度远快于线性 $T$。因此，**无效记忆将以超指数（Super-Exponential）的极速速率衰减至零**，彻底被系统遗忘，释放状态空间。

### 3.2 情况二：高价值记忆的长效保留 (High-Value Memory)
当某项记忆提供了大量不可替代的未来预测信息时，$V(c_t) \gg 0$。
此时项 $\gamma \cdot V(c_t)$ 会在数值上完全压制时间衰减项 $\delta \log \tau$：
$$
\gamma \cdot V(c_t) + \delta \log \tau + \beta \gg 0
$$
此时 $\sigma(\cdot) \to 1 - \epsilon$（$\epsilon \approx 0$）。
无论时间 $\tau$ 经过多久，只要 $V(c_t)$ 维持高位，$f_\tau \approx 1$ 恒成立，记忆单元 $c_t$ 的信息内容无损耗地传递给 $c_{t+1}$，完美克服了梯度消失，实现了有用信息的长效固化（Long-term Retention）。

**结论**：该数学机制能够自适应地对信息进行筛分，完美匹配了连续体记忆（CMS）的物理时域演化要求。
