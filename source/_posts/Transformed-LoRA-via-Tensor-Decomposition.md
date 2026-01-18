---
title: Transformed LoRA via Tensor Decomposition
tags:
  - 毕业论文
  - TR 分解
  - 张量
  - 论文复现
categories:
  - 统计机器学习
cover: /img/tlora_cover.jpg
top_img: /img/default_top_img.jpg
mathjax: true
abbrlink: 41219a96
date: 2025-12-18 15:53:03
---

此篇博客来源于毕设课题的参考文献，复现论文[Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](https://arxiv.org/abs/2501.08727)，将简要记录论文第三节提出的模型。

## 准备工作
### 符号说明
我们使用方括号来表示数组的元素或切片。例如，给定一个张量 $\mathbf{X} \in \mathbb{R}^{I \times J \times K}$，我们定义其矩阵切片为 $\boldsymbol{X}\_{i::} = \mathbf{X}[i, :, :] \in \mathbb{R}^{J \times K}$，向量切片为 $\boldsymbol{x}\_{ij:} = \mathbf{X}[i, j, :] \in \mathbb{R}^{K}$，标量元素为 $x_{ijk} = \mathbf{X}[i, j, k] \in \mathbb{R}$。我们用 $\boldsymbol{I}_I$ 表示形状为 $I \times I$ 的单位矩阵。Kronecker 积（克罗内克积）用 $\otimes$ 表示。矩阵的迹用 $\text{tr}(\cdot)$ 表示。$\text{diag}(\boldsymbol{m})$ 表示对角元素为 $\boldsymbol{m}$ 的对角矩阵。

{% note info no-icon %}
**Little Endian 约定:**
对于张量化的索引，采用小端序（little-endian）约定。对于向量 $\boldsymbol{x} \in \mathbb{R}^I$（其中 $I = \prod_{d=1}^D I_d$），若将其张量化为 $\mathbf{X} \in \mathbb{R}^{I_1 \times \cdots \times I_D}$，则有 $\boldsymbol{x}[\overline{i_1 \dots i_D}] = \mathbf{X}[i_1, \dots, i_D]$，其中：

$$
\overline{i_1 \dots i_D} = i_1 + (i_2 - 1)I_1 + (i_3 - 1)I_1I_2 + \cdots + (i_D - 1)I_1 \cdots I_{D-1}
$$
{% endnote %}

### LoRA

考虑一个线性层 $\boldsymbol{y} = \boldsymbol{W}_0 \boldsymbol{x}$，其中权重的形状为 $I \times I$。

{% note success no-icon %}
**Low-rank Adaptation：** 
给定预训练权重 $\boldsymbol{W}_0$，LoRA 旨在学习一个具有相同形状的加性适应项 $\boldsymbol{\Delta}$。
$$
\boldsymbol{y}' = (\boldsymbol{W}_0 + \boldsymbol{\Delta})\boldsymbol{x}, \quad \text{s.t. } \boldsymbol{\Delta} = \boldsymbol{B}\boldsymbol{A},
$$
{% endnote %}

其中，该适应项通过低秩矩阵分解进行参数化，以减少可训练参数。低秩矩阵 $\boldsymbol{B} \in \mathbb{R}^{I \times R}$ 和 $\boldsymbol{A} \in \mathbb{R}^{R \times I}$ 通过给定的微调任务进行优化，它们总共有 $2IR$ 个可训练参数；如果 $R \ll I$，这一数量远小于原始大小 $I^2$。然而，理想的微调权重可能并不是低秩的。

### OFT

{% note success no-icon %}
**Orthogonal fine-tunning：** 
引入了预训练权重的正交变换来进行适应：

$$
\boldsymbol{y}' = (\boldsymbol{W}_0 \boldsymbol{T})\boldsymbol{x}, 
$$
其中 $\boldsymbol{T}$ 是一个形状为 $I \times I$ 的可训练正交矩阵。
{% endnote %}

然而，由于 $\boldsymbol{T}$ 尺寸巨大，直接优化它在计算上是不可行的。在一些研究中中，$\boldsymbol{T}$ 被参数化为块对角矩阵，这对于较小的参数预算来说极其稀疏。这种稀疏性被认为减少了神经元之间的信息传递和连接，这对微调是有害的。

### 本文方法
LoRA 背后的假设是差异 $\boldsymbol{\Delta}\_{\ast} = \boldsymbol{W}\_{\ast} - \boldsymbol{W}\_0$ 是低秩的。然而，在实际的大型基础模型中，这一假设往往难以成立。

为了解决这个问题，我们首先对 $\boldsymbol{W}\_0$ 应用一个可学习的线性变换 $\boldsymbol{T}$，使 $\boldsymbol{W}\_0$ 与 $\boldsymbol{W}\_*$ 对齐，然后用另一个紧凑结构来近似残差部分。此时，差异变为 $\boldsymbol{\Delta}'\_{\ast} = \boldsymbol{W}\_{\ast} - \boldsymbol{W}\_0\boldsymbol{T}$。变换后，$\boldsymbol{\Delta}'\_{\ast}$ 的秩应小于原始 $\boldsymbol{\Delta}\_{\ast}$ 的秩，从而使我们可以使用更紧凑的结构来近似该残差部分。

整体的微调结构变为：

{% note info no-icon %}
$$
\boldsymbol{y}' = (\boldsymbol{W}_0 \boldsymbol{T} + \boldsymbol{\Delta})\boldsymbol{x},
$$
{% endnote %}

其中 $\boldsymbol{T}$ 和 $\boldsymbol{\Delta}$ 是可学习的紧凑参数化形式。为了满足不同的需求和期望属性，我们分别为变换 $\boldsymbol{T}$ 设计了 TRM 形式，并为残差 $\boldsymbol{\Delta}$ 设计了 TR 形式。

## 张量分解
### 对变化矩阵的TRM分解

#### TRM的原理
由于目标权重空间（例如微调任务的最优权重）不太可能是低秩的，我们假设 $\boldsymbol{T}$ 也具有**满秩结构 (full-rank structure)**。此外，为了解决 OFT 中的稀疏性问题，$\boldsymbol{T}$ 期望具有**密集元素 (dense entries)**。由于该变换矩阵尺寸巨大，它必须由参数高效的结构来表示。

为了满足上述要求，我们采用**张量环矩阵 (Tensor-Ring Matrix, TRM)** 形式的张量分解方法。

给定一个矩阵 $\boldsymbol{T} \in \mathbb{R}^{I \times J}$，假设 $I = \prod\_{d=1}^D I\_d$，$J = \prod\_{d=1}^D J\_d$，并且它可以被重排（张量化）为许多子数组。TRM 将该矩阵分解为 $D$ 个 4 阶因子 $\mathbf{A}^d \in \mathbb{R}^{I\_d \times J\_d \times R\_d \times R\_{d+1}}, \forall d = 1, \dots, D$ 的收缩（contractions），这些因子被称为核心张量 (core tensors)。

序列 $[R\_1, \dots, R\_{D+1}]$（其中 $R\_{D+1} = R\_1$）被称为 TR 秩。为简单起见，我们在本文中假设 $R = R\_1 = \cdots = R\_{D+1}$ 且 $I = J$。

{% note success no-icon %}
**定义（张量的矩阵化）** 设 $\boldsymbol{T} \in \mathbb{R}^{n\_1 \times n\_2 \times \cdots \times n\_d}$ 为一个 $d$ 阶张量。$\boldsymbol{T}$ 的 $k$-展开 ($k$-unfolding) 是一个矩阵，记作 $\mathbf{T}\_{\langle k \rangle}$，其尺寸为 $\prod\_{i=1}^k n\_i \times \prod_{i=k+1}^d n\_i$，其元素为：

$$
\boldsymbol{T}_{\langle k \rangle}(\overline{i_1 \cdots i_k}, \overline{i_{k+1} \cdots i_d}) = \boldsymbol{T}(i_1, i_2, \dots, i_d), \tag{6}
$$
{% endnote %}

其中，前 $k$ 个索引枚举 $\mathbf{T}_{\langle k \rangle}$ 的行，而后 $d-k$ 个索引枚举其列。

TRM 假设矩阵的每个元素通过以下方式计算：

{% note info no-icon %}
**TRM分解：**
采用了矩阵张量化的技巧，先将矩阵张量化后再作分解。
$$
\boldsymbol{T}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = \text{tr}(\mathbf{A}^1[i_1, j_1, :, :] \cdots \mathbf{A}^D[i_D, j_D, :, :]).
$$
{% endnote %}

为简便起见，我们将 TRM 格式记为 $\boldsymbol{T} = \text{TRM}(\mathbf{A}^{1:D})$，其中 $\mathbf{A}^{1:D}$ 表示 $\\{\mathbf{A}^1, \dots, \mathbf{A}^D\\}$。

TRM 对于表示我们需要的矩阵是一个高度紧性的分解形式。若 $\mathbf{A}^{1:D}$是稠密且满秩的，那么$\boldsymbol{T} = \text{TRM}(\mathbf{A}^{1:D})$也是一样的。

>在参数上，可以计算得到 TRM 的储存开销是$\mathcal{O}(DI^{2/D}R^2)$。 通过调整合适的超参数$D$和$R$，开销将远小于原来的 $\mathcal{O}(I^2)$。

#### 如何初始化
为保证这个模型和原模型的初始化一致，我们需要保证$T$被初始化为单位矩阵，而我们可以按照如下方法构造单位矩阵。


{% note info no-icon %}
**Proposition 1** 
如果我们按照如下方式初始化TRM的每一个因子：
$$
\mathbf{A}^d[:, :, r, r'] = \boldsymbol{I}_{I_d} / R, \quad \forall d = 1, \dots, D, \text{ 且 } r, r' = 1, \dots, R,
$$

则生成的 TRM $\boldsymbol{T}$ 是一个单位矩阵。
{% endnote %}

**证明：**

根据 TRM 的定义，我们有：

$$
\boldsymbol{T}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = \text{tr}(\mathbf{A}^1[i_1, j_1, :, :] \cdots \mathbf{A}^D[i_D, j_D, :, :]).
$$

对于 $d = 1, \dots, D$，如果 $i_d = j_d$，则 $\mathbf{A}^d[i_d, j_d, :, :]$ 的所有元素均为 $1/R$；否则为零。接下来我们可以分别讨论 $\boldsymbol{T}$ 的对角元素和非对角元素。

**1. 对于非对角元素**
即 $\overline{i_1 \cdots i_D} \neq \overline{j_1 \cdots j_D}$，意味着至少存在一个子索引 $i_d \neq j_d$。因此：

$$
\boldsymbol{T}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = 0, \quad \text{因为当 } i_d \neq j_d \text{ 时 } \mathbf{A}^d[i_d, j_d, :, :] = \mathbf{0}.
$$

**2. 对于对角元素**
即 $\overline{i_1 \cdots i_D} = \overline{j_1 \cdots j_D}$，意味着对于所有 $d = 1, \dots, D$，都有 $i_d = j_d$。此时核心张量变为：

$$
\mathbf{A}^d[i_d, j_d, :, :] = \mathbf{1}_{R \times R} / R, \quad \forall d = 1, \dots, D \text{ 且 } i_d, j_d = 1, \dots, I_d,
$$

其中 $\mathbf{1}_{R \times R}$ 是一个形状为 $R \times R$ 且所有元素均为 1 的矩阵。因此，$\boldsymbol{T}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = 1$，因为：

$$
\text{tr}\left( \frac{\mathbf{1}_{R \times R}}{R} \cdots \frac{\mathbf{1}_{R \times R}}{R} \right) = 1.
$$

综上所述，$\boldsymbol{T}$ 是一个单位矩阵。

#### 如何正则化
微调需要正则化，在过去其他模型中，有恒等正则化（identity regularization）和正交正则化（orthogonal regularization）的方法。但是直接计算 $T$ 的正则化是一件比较昂贵的事，通过TRM的性质，我们可以通过高效的计算核心张量 $\mathbf{A}^{1:D}$ 来减少计算开销。

通过Proposition 1，我们可以通过如下方式计算$||\boldsymbol{T} - \boldsymbol{I}||_F$。

{% note success no-icon %}
**Identity Regularization**
$$
\mathcal{R}_I(\mathbf{A}^{1:D}) = \sum_{d=1}^D \sum_{r,r'=1}^R ||\mathbf{A}^d[:,:,r,r'] - \frac{I_{I_d}}{R}||_F
$$
{% endnote %}

对于正交正则化，首先证明一个命题。

{% note info no-icon %}
**Proposition 2**
两个 TRM $\boldsymbol{X} = \text{TRM}(\mathbf{A}^{1:D})$ 和 $\boldsymbol{Y} = \text{TRM}(\mathbf{B}^{1:D})$ 的矩阵乘积仍然是一个 TRM，即 $\boldsymbol{X}\boldsymbol{Y}^\mathsf{T} = \text{TRM}(\mathbf{C}^{1:D})$，其中每个核心张量满足：

$$
\mathbf{C}^d[i_d, j_d, :, :] = \sum_{l_d} \mathbf{A}^d[i_d, l_d, :, :] \otimes \mathbf{B}^d[j_d, l_d, :, :]
$$

对于所有 $d = 1, \dots, D$ 和 $i_d, j_d = 1, \dots, I_d$ 均成立。
{% endnote %}

**证明：**

为简便起见，在此证明中，我们将切片记为 $\mathbf{A}^d[i_d, j_d] = \mathbf{A}^d[i_d, j_d, :, :]$ 以及 $\mathbf{B}^d[i_d, j_d] = \mathbf{B}^d[i_d, j_d, :, :]$。

假设 $\boldsymbol{X}$ 的形状为 $I \times K$，其中 $I = \prod\_{d=1}^D I\_d$ 且 $K = \prod\_{d=1}^D K\_d$；$\boldsymbol{Y}$ 的形状为 $J \times K$，其中 $J = \prod\_{d=1}^D J\_d$。
根据 TRM 的定义，我们可以按如下方式计算乘积 $\boldsymbol{Z} = \boldsymbol{X}\boldsymbol{Y}^\mathsf{T}$：

$$
\begin{aligned}
\boldsymbol{Z}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}]
&= \sum_{k_1, \dots, k_D}^{K_1, \dots, K_D} \boldsymbol{X}[\overline{i_1 \cdots i_D}, \overline{k_1 \cdots k_D}] \boldsymbol{Y}[\overline{j_1 \cdots j_D}, \overline{k_1 \cdots k_D}] \\
&= \sum_{k_1, \dots, k_D}^{K_1, \dots, K_D} \text{tr}(\mathbf{A}^1[i_1, k_1] \cdots \mathbf{A}^D[i_D, k_D]) \\
&\quad \quad \quad \quad \cdot \text{tr}(\mathbf{B}^1[j_1, k_1] \cdots \mathbf{B}^D[j_D, k_D]) \\
&= \sum_{k_1, \dots, k_D}^{K_1, \dots, K_D} \text{tr} \left( \{ \mathbf{A}^1[i_1, k_1] \cdots \mathbf{A}^D[i_D, k_D] \} \right. \\
&\quad \quad \quad \quad \left. \otimes \{ \mathbf{B}^1[j_1, k_1] \cdots \mathbf{B}^D[j_D, k_D] \} \right) \\
&= \sum_{k_1, \dots, k_D}^{K_1, \dots, K_D} \text{tr} \left( (\mathbf{A}^1[i_1, k_1] \otimes \mathbf{B}^1[j_1, k_1]) \right. \\
&\quad \quad \quad \quad \left. \cdots (\mathbf{A}^D[i_D, k_D] \otimes \mathbf{B}^D[j_D, k_D]) \right) \\
&= \text{tr} \left\{ \sum_{k_1, \dots, k_D}^{K_1, \dots, K_D} [ (\mathbf{A}^1[i_1, k_1] \otimes \mathbf{B}^1[j_1, k_1]) \right. \\
&\quad \quad \quad \quad \left. \cdots (\mathbf{A}^D[i_D, k_D] \otimes \mathbf{B}^D[j_D, k_D]) ] \right\} \\
&= \text{tr} \left\{ \sum_{k_1}^{K_1} (\mathbf{A}^1[i_1, k_1] \otimes \mathbf{B}^1[j_1, k_1]) \right. \\
&\quad \quad \quad \quad \left. \cdots \sum_{k_D}^{K_D} (\mathbf{A}^D[i_D, k_D] \otimes \mathbf{B}^D[j_D, k_D]) \right\},
\end{aligned}
$$

上述结果遵循 TR 格式。因此 $\boldsymbol{X}\boldsymbol{Y}^\mathsf{T} = \text{TRM}(\mathbf{C}^{1:D})$，其中每个核心张量 $\mathbf{C}^d[i\_d, j\_d, :, :] = \sum_{l\_d} (\mathbf{A}^d[i\_d, l\_d, :, :] \otimes \mathbf{B}^d[j\_d, l\_d, :, :])$。

为了使 $\boldsymbol{X}$ 正交，即 $\boldsymbol{X}\boldsymbol{X}^\mathsf{T} = \boldsymbol{I}$，我们可以根据命题 1 中的初始化方案对 $\mathbf{C}^{1:D}$ 进行正则化。

通过Proposition 2，我们可以通过如下方式计算$||\boldsymbol{TT^{\text{T}}} - \boldsymbol{I}||_F$。

{% note success no-icon %}
**Orthogonal Regularization**
$$
\mathcal{R}_O(\mathbf{A}^{1:D}) = \sum_{d=1}^D \sum_{i_d, j_d=1}^{I_d, J_d} ||\sum_{l=1}^{I_d} (\mathbf{A}^d[i_d, l, :, :] \otimes \mathbf{B}^d[j_d, l, :, :]) - \frac{I_{R^2}}{R}||_F
$$
{% endnote %}

>实践中发现恒等正则化的效果优于正交正则化。

### 对残差的TR分解

我们希望在transform之后，剩下的参差可以用非常紧致的结构近似。这里的 'Compact' (紧致) 并非指拓扑学意义上的紧致性，而是指可以高度压缩参数的技术。

假设残差 $\boldsymbol{\Delta} \in \mathbb{R}^{I \times J}$，这里 $I = \prod\_{d=1}^D I\_d$，$J = \prod\_{d=1}^D J\_d$，TR分解将这个矩阵分解为$2D$个三阶核心张量，可记为$\mathbf{B}^{d}\in\mathbb{R}^{I_d\times R\times R}$和$\mathbf{C}^{d}\in\mathbb{R}^{J_d\times R\times R}$

{% note info no-icon %}
**TR分解：**

$$
\boldsymbol{\Delta}[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = \text{tr}(\mathbf{B}^1[i_1, :, :] \cdots \mathbf{B}^D[i_D, :, :] \mathbf{C}^{1}[j_1, :, :] \cdots \mathbf{C}^{D}[j_D, :, :]).
$$
{% endnote %}

>可以计算得到 TR 的空间复杂度是$\mathcal{O}(DI^{1/D}R^2)$。 通过调整合适的超参数$D$和$R$，比TRM的空间复杂度还要更小。

#### TR的初始化
由于高阶结构，TR格式可能对初始化更敏感。先前在PEFT中采用TR的工作对所有因子使用随机高斯初始化，因此失去了像LoRA中那样的整体适配的零初始化。这可能会导致优化不稳定，并导致预训练模型信息的丢失。

特别地，我们可以将TR层表示为一系列线性层。具体做法如下：

给定张量$\mathbf{A}$和$\mathbf{X}$，形状分别为$I_d\times R\times R$和$I\_1\times\cdots\times I\_d\times R$，我们定义 $\times_2$：

{% note success no-icon %}
$\times_2$运算：
$$
\mathbf{A} \times_2 \mathbf{X} = \sum_{l=1}^{I_d}\sum_{r=1}^{R}\mathbf{A}[l,;,r]\mathbf{X}[I_1,\cdots,l,r]
$$
{% endnote %}

这种运算的结果形状为$I\_1\times\cdots\times I\_{d-1}\times R$，因此我们可以将TR残差层定义为：

$$
\text{TR}(\mathbf{B},\mathbf{C})\mathbf{x} = \text{tr}(\mathbf{B}^1\times_2\cdots\times_2\mathbf{B}^D\times_2\mathbf{C}^1\times_2\cdots\times_2\mathbf{C}^{D}\times_2\mathbf{X})
$$

为了保证零初始化，我们将$\mathbf{B}^1$初始化为零张量，其余初始化服从高斯分布$\mathcal{N}(0,\sigma^2)$，这里$\sigma$的选取符合$\mu$P 框架，即$\sigma = \Theta(\sqrt{n\_{out}}/n\_{in})$，这里的$n\_{out} = R$，$n\_{in} = I_dR$，这样的目的是通过控制权重的谱范数比例，可以确保无论网络多宽，每一层的激活值变化都能保持在一个最佳的量级，既不消失也不爆炸，从而最大化模型的学习效率。

## 与其他方法的比较

在先前已有的工作中，主要依赖了一些稀疏或固定的变换，而非这里稠密且可学习的TRM。

### DoRA

{% note info no-icon %}
**DoRA：**

目前非常流行的微调方法，它将权重分解为幅度（Magnitude）和方向（Direction），分别微调它们。
$$
\mathbf{W}' = \dfrac{\mathbf{W}_0 + \mathbf{BA}}{\|\mathbf{W}_0 + \mathbf{BA}\|_c}\cdot \text{diag}(\mathbf{m})
$$

这里的$\mathbf{m}\in\mathbb{R}^{1\times J}$, $\mathbf{B}\in\mathbb{R}^{I\times R}$和$\mathbf{A}\in\mathbb{R}^{R\times J}$是可训练的参数，$\|\cdot\|_c$代表列范数。
{% endnote %}

作者指出在实际训练的时候，将范数视为常数不进行更新，因此可以将DoRA重写为
$$
\mathbf{W}'\propto \mathbf{W}_0\text{diag}(\mathbf{m}) + \mathbf{BA}\text{diag}(\mathbf{m}) \approx \mathbf{W}_0\text{diag}(\mathbf{m}) + \mathbf{BA}
$$
这里可以将最后的$\mathbf{m}$融入$\mathbf{A}$中训练

这意味着，DoRA 实际上隐含了一个**对角变换矩阵 ** $\text{diag}(m)$。相比之下，TRM是一个更稠密的变换。


### 采用固定变换的方法

另一类 PEFT 方法（如 **FouRA**, **BoFT** 等）采用固定的正交变换或频域变换。这类方法假设权重在特定的固定变换下具有更好的低秩结构。例如，FouRA 在变换后的权重空间进行微调：

{% note info no-icon %}
**FouRA：**

$$
\mathbf{y} = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{W}_0) + \mathbf{B} \cdot \mathcal{F}(\mathbf{A}))\mathbf{x}
$$
其中 $\mathcal{F}$ 是离散傅里叶变换（DFT）矩阵，它是**固定不变的**。

{% endnote %}

这类方法应用固定的变换，而TRM采用可学习的变换，能够自适应地学习不同模型和任务之间的这种投影。


下表总结了 TLoRA 在现有方法中的定位：

| 方法 | 变换类型 (Transform Type) | 变换性质 (Property) | 潜在问题 |
| :--- | :--- | :--- | :--- |
| **DoRA** | 对角矩阵 (Diagonal) | **可学习** 但 **极度稀疏** | 无法建模复杂的特征相关性 |
| **FouRA/OFT** | 正交/傅里叶矩阵 | **稠密** 但 **固定不可变** | 强依赖先验假设，缺乏灵活性 |
| **TLoRA** | **Tensor-Ring 矩阵** | **稠密 且 可学习** | 兼顾了表达能力与自适应性 |

