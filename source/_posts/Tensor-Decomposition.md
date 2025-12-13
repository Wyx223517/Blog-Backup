---
title: 张量分解
tags:
  - 毕业论文
  - CP 分解
  - Tucker 分解
  - TT 分解
  - TR 分解
  - 张量
categories:
  - 统计机器学习
cover: /img/tensor_cover.jpg
mathjax: true
abbrlink: e05ab48b
date: 2025-12-12 16:40:43
---

张量又称多维数组，是矩阵的高维推广。与传统的矩阵（二维数据）不同，张量能够处理更高维度的数据，因此它是多维数据表示的理想工具。

## 张量的概念

{% note info no-icon %}
**定义 (张量):**
张量是一个多维数组，$d$ 阶张量是一个具有 $d$ 个维度的数组，表示为：
$$
\mathcal{X} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}
$$
{% endnote %}

下图展示了从标量到三阶张量的直观对比：
<div style="width: 60%; margin: 0 auto;">
{% asset_img Example_of_Tensors.png 标量、向量、矩阵和三阶张量 %}
</div>

## 张量分解
### Tucker 分解

张量分解是一种将高维张量分解为若干低维因子组合的技巧。其中一种为 **Tucker 分解**，它可以被视为张量的高阶主成分分析（PCA）。

Tucker 分解的核心思想是将张量分解为**核心张量**与**各模式因子矩阵**的乘积。

具体地，给定一个张量 $\mathcal{X} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$，Tucker 分解表示为：

{% note success no-icon %}
**Tucker Decomposition:**

$$
\mathcal{X} \approx \mathcal{C} \times_1 Q^1 \times_2 Q^2 \cdots \times_d Q^d = \sum_{j_1=1}^{m_1} \sum_{j_2=1}^{m_2} \cdots \sum_{j_d=1}^{m_d} c_{j_1 j_2 \dots j_d} \, \mathbf{q}^{1}_{j_1} \circ \mathbf{q}^{2}_{j_2} \circ \cdots \circ \mathbf{q}^{d}_{j_d}
$$
{% endnote %}

其中：
* $\mathcal{C}\in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_d}$ 是 **核心张量 (Core Tensor)**，反映了不同模式之间的相互作用。
* $Q^{k} \in \mathbb{R}^{n_k \times m_k}$ 是 **因子矩阵 (Factor Matrices)**。

<div style="width: 60%; margin: 0 auto;">
{% asset_img Tucker.png Tucker分解示意图 %}
</div>

### CP 分解

特别地，当 $m\_{1} =m\_{2} = \cdots =m\_{d} = r$，且核心张量限制为**对角型**时，被称为 **CP 分解** (Canonical Polyadic Decomposition)。

它将张量近似为一组**秩为 1 的张量之和**。具体而言，给定一个张量 $\mathcal{X} \in \mathbb{R}^{n\_1 \times n\_2 \times \cdots \times n\_d}$，其分解形式为：

{% note success no-icon %}
**CP Decomposition:**

$$
\mathcal{X} \approx \sum_{j=1}^r \mathbf{p}_j^1 \circ \mathbf{p}_j^2 \circ \cdots \circ \mathbf{p}_j^d
$$
{% endnote %}

其中 $\mathbf{p}\_j^k$ 为 $n\_k$ 维向量（$k = 1, 2, \cdots, d$）。

**优势：** CP 分解将系数从 $p\_{1}\times\cdots\times p\_{d}$ 降低至 $r(p\_{1} + p\_{2} +\cdots +p\_{d})$，提供了极高效的数据降维能力。

**缺点：** CP 分解的最优秩的确定是一个NP-hard 的问题，实际操作时难以实现。

<div style="width: 60%; margin: 0 auto;">
{% asset_img PARAFAC.png PARAFAC分解示意图 %}
</div>

### TT 分解

**TT分解的核心思想**

* 假设模态的排序具有某种语义相关性。
* 将高阶张量 $\mathcal{A}$ 近似为一串 3 阶张量的链式乘积。

{% note success no-icon %}
**TT Decomposition:**

$$
\mathcal{A}(i_1, \dots, i_d) = \sum_{\alpha_0=1}^{r_0} \cdots \sum_{\alpha_d=1}^{r_d} \mathcal{G}_1(\alpha_0, i_1, \alpha_1) \mathcal{G}_2(\alpha_1, i_2, \alpha_2) \cdots \mathcal{G}_d(\alpha_{d-1}, i_d, \alpha_d)
$$
{% endnote %}

其中 $\mathcal{G}\_k \in \mathbb{R}^{r\_{k-1} \times n\_k \times r\_k}$。$r\_k$ 称为 **TT-秩 (TT-rank)**，且满足边界条件 $r\_0 = r\_d = 1$。

<div style="width: 60%; margin: 0 auto;">
  {% asset_img TT.png TT分解示意图 %}
</div>

**优势：** TT分解的好处在于随着张量阶数提升，分解后的形式仅仅是在已有分解的基础上在后面加上几节“车厢”，而相比之下Tucker分解的核心张量参数量仍是指数阶的。

**不足：** 然而，TT分解高度依赖于张量维度的排列，这导致难以找到最佳的TT表示。

### TR 分解
TR 分解将高阶张量 $\mathcal{T}$ 表示为一系列三阶核心张量的**循环**矩阵乘积的迹。它可以看作是 TT 分解的推广（当首尾秩为 1 时退化为 TT），也可以看作是首尾相连的“环状”结构。

{% note success no-icon %}
**TR Decomposition:**

$$
\mathcal{T}(i_1, i_2, \dots, i_d) = \text{Tr} \left( \mathbf{Z}_1(i_1) \mathbf{Z}_2(i_2) \cdots \mathbf{Z}_d(i_d) \right) = \text{Tr} \left( \prod_{k=1}^d \mathbf{Z}_k(i_k) \right)
$$
{% endnote %}

其中 $\mathcal{Z}\_k \in \mathbb{R}^{r\_k \times n\_k \times r\_{k+1}}$ 为第 $k$ 个核心张量，$\mathbf{Z}\_k(i\_k)$ 表示核心张量 $\mathcal{Z}\_k$ 的第 $i\_k$ 个侧切片矩阵（大小为 $r\_k \times r\_{k+1}$）。

这里的 $\mathbf{r} = [r\_1, r\_2, \cdots, r\_d]^{\text{T}}$ 被称为TR秩。

<div style="width: 60%; margin: 0 auto;">
  {% asset_img TR.png TR分解示意图 %}
</div>

如果我们把迹运算展开，就可以得到索引形式的TR分解。

{% note success no-icon %}
**TR Decomposition in Index Form:**

$$
\mathcal{T}(i_1, \dots, i_d) = \sum_{\alpha_1=1}^{r_1} \cdots \sum_{\alpha_d=1}^{r_d} \prod_{k=1}^{d} \mathcal{Z}_k(\alpha_k, i_k, \alpha_{k+1})
$$
{% endnote %}

注意这里隐含了条件 $\alpha_{d+1} = \alpha_1$。这个公式直观地展示了每一个张量元素是如何通过核心张量中的元素相乘并求和得到的。

更进一步，我们可以将 TR 分解理解为一系列由 $d$ 个向量外积产生的 **秩-1 张量 (Rank-1 Tensors)** 之和。

{% note success no-icon %}
**TR Decomposition in the Tensor Form:**

$$
\mathcal{T} = \sum_{\alpha_1=1}^{r_1} \cdots \sum_{\alpha_d=1}^{r_d} \mathbf{z}_1(\alpha_1, \alpha_2) \circ \mathbf{z}_2(\alpha_2, \alpha_3) \circ \cdots \circ \mathbf{z}_d(\alpha_d, \alpha_1)
$$
{% endnote %}

其中：
* 符号 $\circ$ 表示向量的外积。
* $\mathbf{z}\_k(\alpha\_k, \alpha\_{k+1})$ 表示核心张量 $\mathcal{Z}\_k$ 的**第 $(\alpha\_k, \alpha\_{k+1})$ 根 mode-2 纤维 (Fiber)**（即固定了前后两个秩索引后，剩下的那个长度为 $n_k$ 的向量）。

由此可见，通过TR分解可以将参数量从 $\mathcal{O}(n^d)$降低至 $\mathcal{O}(dnr^2)$.

TR 分解最显著的一个性质是其对张量维度的循环移位具有不变性。这一点将 TR 与 TT 分解严格区分开来。

{% note primary no-icon %}
**Theorem 1 (Circular Dimensional Permutation Invariance)**

令 $\mathcal{T} \in \mathbb{R}^{n\_1 \times n\_2 \times \cdots \times n\_d}$ 为一个 $d$ 阶张量，其 TR 分解表示为 $\mathcal{T} = \mathfrak{R}(\mathcal{Z}\_1, \mathcal{Z}\_2, \dots, \mathcal{Z}\_d)$。

如果我们定义 $\overrightarrow{\mathcal{T}}^{k}$ 为将 $\mathcal{T}$ 的维度**循环右移** $k$ 次后得到的新张量，$\overrightarrow{\mathcal{T}}^{k}\in\mathbb{R}^{n\_{k+1} \times \cdots \times n\_d \times n\_1 \times \cdots \times n\_k}$，那么 $\overrightarrow{\mathcal{T}}^{k}$ 的 TR 分解核心张量序列仅需做同样的循环移位：

$$
\overrightarrow{\mathcal{T}}^{k} = \mathfrak{R}(\mathcal{Z}_{k+1}, \dots, \mathcal{Z}_d, \mathcal{Z}_1, \dots, \mathcal{Z}_k)
$$
{% endnote %}

**证明:**

这一性质的证明直接依赖于矩阵迹运算的循环性质 $\text{Tr}(\mathbf{A}\mathbf{B}) = \text{Tr}(\mathbf{B}\mathbf{A})$。

根据 TR 分解的定义公式 (1)，我们可以将其重写为：
$$
\begin{aligned}
\mathcal{T}(i_1, i_2, \dots, i_d) &= \text{Tr}(\mathbf{Z}_1(i_1) \mathbf{Z}_2(i_2) \cdots \mathbf{Z}_d(i_d)) \\
&= \text{Tr}(\mathbf{Z}_2(i_2) \mathbf{Z}_3(i_3) \cdots \mathbf{Z}_d(i_d) \mathbf{Z}_1(i_1)) \\
&\quad \vdots \\
&= \text{Tr}(\mathbf{Z}_{k+1}(i_{k+1}) \cdots \mathbf{Z}_d(i_d) \mathbf{Z}_1(i_1) \cdots \mathbf{Z}_k(i_k))
\end{aligned}
$$

这表明，如果我们改变张量维度的物理顺序（例如将前 $k$ 个维度移到末尾），我们只需要相应地循环移动核心张量的顺序，而不需要重新计算核心张量的值。

