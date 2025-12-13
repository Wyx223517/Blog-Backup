---
title: Laplace Equation Note
tags:
  - Harmonic Analysis
  - Laplace Equation
categories:
  - PDE
cover: /img/laplace_cover.jpg
top_img: /img/default_top_img.jpg
mathjax: true
abbrlink: 1f9ad889
date: 2025-12-10 15:20:00
---

这篇博客主要是为了测试写博客的功能，最初来自于PDE课程上的课堂笔记。教材是Evans的PDE，封面图来自在Berkeley交换时在Evans Hall借阅的纸质书。耿老师教得很好，考核也很仁慈，现在他已经发了四大了可喜可贺！以下是正文。

Among the most important of all partial differential equations are undoubtedly *Laplace's equation*

$$
\Delta u = 0
$$

and *Poisson's equation*

$$
-\Delta u = f
$$

First of all we study the properties of **harmonic** functions.

{% note info no-icon %}
**Definition 1 (Harmonic Function):**
A $C^2$ function $u$ satisfying $\Delta u = 0$ is called a *harmonic function*.
{% endnote %}

## Mean-Value Formulas

### Theorem: Mean-value formulas for Laplace's equation

{% note success no-icon %}
**Theorem 1 (Mean-value formulas):**
If $u \in C^2(U)$ is harmonic, then

$$
u(x) = \frac{1}{|\partial B(x,r)|} \int_{\partial B(x,r)} u \, d\sigma = \frac{1}{|B(x,r)|} \int_{B(x,r)} u \, dy
$$

for each ball $B(x, r) \subset U$.
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**

Set

$$
\varphi(r) = \frac{1}{|\partial B(x,r)|} \int_{\partial B(x,r)} u(y) \, d\sigma(y)
$$

where $|\partial B(x,r)| = n\alpha(n) r^{n-1}$, and $\alpha(n) = \frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2} + 1\right)}$.

By a change of variables $y = x + rz$, we have

$$
\varphi(r) = \frac{r^{n-1}}{n\alpha(n) r^{n-1}} \int_{\partial B(0,1)} u(x + rz) \, d\sigma(z)
$$

Then,

$$
\varphi'(r) = \frac{1}{n\alpha(n) } \int_{\partial B(0,1)} \nabla u(x + rz) \cdot z \, d\sigma(z)
$$

By the divergence theorem:

$$
\int_{\partial B(x,r)} \frac{\partial u}{\partial \nu} \, d\sigma = \int_{B(x,r)} \Delta u \, dy
$$

We have

$$
\begin{aligned}
\varphi^{\prime}(r) &= \frac{1}{n\alpha(n) } \int_{\partial B(0,1)} \nabla u(x + rz) \cdot \dfrac{y-x}{r} \, d\sigma(z) \\
&= \frac{1}{n\alpha(n)r^{n-1} } \int_{\partial B(x,r)} \nabla u(y) \cdot \nu \, d\sigma(y)\\
&= \frac{1}{n\alpha(n)r^{n-1} } \int_{\partial B(x,r)} \dfrac{\partial u (y)}{\partial\nu}  \, d\sigma(y)\\
&= \frac{1}{n\alpha(n)r^{n-1} } \int_{ B(x,r)} \Delta u (y)  \, dy\\
&= \dfrac{r}{n}\cdot\dfrac{1}{|B(x,r)|}\int_{ B(x,r)} \Delta u (y)  \, dy
\end{aligned}
$$

Since $u$ is harmonic ($\Delta u = 0$), we conclude that $\varphi'(r) = 0$, which implies that $\varphi(r)$ is constant.

By the Lebesgue Differentiation Theorem

$$
\varphi(r) =\lim_{t \to 0} \frac{1}{|\partial B(x,t)|} \int_{\partial B(x,t)} u \, d\sigma = u(x)
$$

By employing polar coordinates, we have

$$
\int_{B(x,r)} u \, dy = \int_0^r \int_{\partial B(x,s)} u \, d\sigma \, ds
$$

and

$$
\int_{B(x,r)} u \, dy = n\alpha(n) \int_0^r s^{n-1} u(x) \, ds
$$

Finally,

$$
u(x) = \frac{1}{|B(x,r)|} \int_{B(x,r)} u \, dy
$$
**Q.E.D.**
{% endfolding %}

### Theorem: Converse to Mean Value Property

{% note success no-icon %}
**Theorem 2 (Converse to mean-value property):**
If $u \in C^2(U)$ satisfies

$$
u(x) = \frac{1}{|\partial B(x,r)|} \int_{\partial B(x,r)} u \, dS
$$

for each ball $B(x,r) \subset U$, then $u$ is harmonic.
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
Suppose not. WLOG, there exists a ball $B(x, r) \subset U$, such that $\Delta u > 0$ in $B(x, r)$.

Define

$$
\varphi(r) = \dfrac{1}{|B(x,r)|}\int_{\partial B(x,r)} u \, d\sigma
$$

Then

$$
\varphi'(r) = \dfrac{r}{n}\cdot\dfrac{1}{|B(x,r)|}\int_{ B(x,r)} \Delta u (y)  \, dy > 0
$$

However, since $\varphi(r)$ is a constant, $\varphi'(r) = 0$, which is a contradiction (as $\Delta u > 0$).
Therefore, $u$ must be harmonic. **Q.E.D.**
{% endfolding %}

## Maximum Principle

{% note success no-icon %}
**Theorem 3 (Strong maximum principle):**
Suppose $u \in C^2(U) \cap C(\overline{U})$ is harmonic within $U$.

1.  **Then**
    $$
    \max_{\overline{U}} u = \max_{\partial U} u
    $$
2.  **Furthermore**, if $U$ is connected and there exists a point $x_0 \in U$ such that
    $$
    u(x_0) = \max_{\overline{U}} u
    $$
    then $u$ is constant within $U$.

Assertion (i) is the maximum principle for Laplace's equation and (ii) is the strong maximum principle. Replacing $u$ by $-u$, we recover also similar assertions with "min" replacing "max".
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
Suppose $\exists x_0 \in U$ such that $u(x_0) = \max_{\overline{U}} u := M$.

Then for $0 < r < \delta(x_0) = \operatorname{dist}(x_0, \partial  U)$, the mean value property implies that

$$
u(x_0) = \frac{1}{|B(x_0, r)|} \int_{B(x_0, r)} u(x) \, dx \leq M
$$

Since $u(x_0) = M$, equality holds iff $u(x) = M$ for all $x \in B(x_0, r)$.

Repeating this argument, we see that $u(y) = M$ for all $y \in B(x_0, r)$.

Hence the set $\{ x \in U \mid u(x) = M \}$ is both open and relatively closed in $U$, and thus equals $U$ if $U$ is connected. **Q.E.D.**
{% endfolding %}

Before we learn the weak maximum principle, we first introduce the following definition:

{% note info no-icon %}
**Definition 2 (Subharmonic and Superharmonic Functions):**
Let $u$ be a $C^2$ function in $U$. Then $u$ is a **subharmonic** (superharmonic) function in $U$ if $\Delta u \geq 0$ ($\Delta u \leq 0$).
{% endnote %}

Subharmonic and superharmonic functions both have maximum principle, we only show one of it.

{% note success no-icon %}
**Theorem 4 (Maximum Principle for Subharmonic Functions):**
Let $U$ be a bounded domain in $\mathbb{R}^n$ and $u \in C^2(U) \cap C(\overline{U})$ be subharmonic in $U$. Then $u$ attains its maximum in $\overline{U}$, i.e.,

$$
\max_{\overline{U}} u = \max_{\partial U} u
$$
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
1. First, consider $\Delta u > 0$ in $U$. If $u$ has a local maximum at a point $x_0 \in U$, then the Hessian matrix $\nabla^2 u(x_0)$ is negative semi-definite. Thus,
$$
\Delta u(x_0) = \operatorname{tr}(\nabla^2 u(x_0)) \leq 0
$$
which is a contradiction.

2. Now consider $\Delta u \geq 0$. To handle this, for any $\varepsilon > 0$, define
$$
u_\varepsilon(x) = u(x) + \varepsilon |x|^2
$$
Then,
$$
\Delta u_\varepsilon(x) = \Delta u(x) + \Delta (\varepsilon |x|^2) = \Delta u(x) + 2n\varepsilon > 0
$$
By Step 1, we have
$$
\max_{\overline{U}} u_\varepsilon = \max_{\partial U} u_\varepsilon
$$

Observe that
$$
\max_{\overline{U}} u \leq \max_{\overline{U}} u_\varepsilon = \max_{\partial U} u_\varepsilon \leq \max_{\partial U} u + \max_{\partial U} (\varepsilon |x|^2)
$$
Taking $\varepsilon \to 0$, we conclude
$$
\max_{\overline{U}} u = \max_{\partial U} u
$$
**Q.E.D.**
{% endfolding %}

{% note success no-icon %}
**Theorem 5 (Uniqueness):**
Let $g \in C(\partial U)$, $f \in C(U)$. Then there exists at most one solution $u \in C^2(U) \cap C(\overline{U})$ of the boundary-value problem

$$
\begin{cases}
    -\Delta u = f & \text{in } U, \\
    u = g & \text{on } \partial U.
\end{cases}
$$
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
If $u$ and $\tilde{u}$ are both solutions for the boundary-value problem , apply maximum principle to the harmonic functions $w := \pm (u - \tilde{u})$.
{% endfolding %}

{% note warning no-icon %}
**Remark:** If $U$ is unbounded, the conclusion fails.
{% endnote %}

## Regularity

Our main result in this section is that if $u\in C^{2}$ is harmonic, then necessarily $u\in C^{\infty}$.

Let $\Omega \subset \mathbb{R}^n$ be open. For $\varepsilon > 0$, denote

$$
\Omega_\varepsilon = \{ x \in \Omega \mid \text{dist}(x, \partial \Omega) > \varepsilon \}
$$

{% note info no-icon %}
**Definition 3 (Mollifier):**
Define the mollifier as:

$$
\eta(x) =
\begin{cases}
    C \exp\left(\frac{1}{ |x|^2 - 1}\right), & \text{if } |x| < 1, \\
    0, & \text{if } |x| \geq 1.
\end{cases}
$$

where $\eta \in C^\infty(\mathbb{R}^n)$ and satisfies $\int_{\mathbb{R}^n} \eta(x) \, dx = 1$.

Let
$$
\eta_\varepsilon(x) = \frac{1}{\varepsilon^n} \eta\left(\frac{x}{\varepsilon}\right)
$$
This is called the *standard mollifier*.
{% endnote %}

By definition, we know that $\eta_{\varepsilon}$ has support, and $spt(\eta_{\varepsilon})\subset B(0,\varepsilon)$.

{% note info no-icon %}
**Definition 4 (Mollification):**
For a function $f$, define its mollification as:

$$
f^\varepsilon = \eta_\varepsilon * f = \int_{\Omega} \eta_\varepsilon(x-y) f(y) \, dy = \int_{B(0,\varepsilon)} \eta_\varepsilon(y) f(x-y) \, dy
$$
{% endnote %}

{% note success no-icon %}
**Theorem 6 (Properties of Mollifiers):** $f^{\varepsilon}\in C^{\infty}(\Omega_{\varepsilon})$.
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
We can rewrite:

$$
f^\varepsilon(x) = \int_{\Omega} \eta_\varepsilon(y) f(x - y) \, dy = \frac{1}{\varepsilon^n} \int_{\Omega} \eta\left(\frac{y}{\varepsilon}\right) f(x - y) \, dy
$$

Change variables with $z = \frac{y}{\varepsilon}$, $dy = \varepsilon^n dz$:

$$
f^\varepsilon(x) = \int_{\mathbb{R}^n} \eta(z) f(x - \varepsilon z) \, dz
$$

Since $\eta \in C^\infty$, we conclude $f^\varepsilon \in C^\infty$.

In fact, fix $x \in U_\varepsilon$, $i \in \{ 1, \dots, n \}$, and $h$ so small that $x + h e_i \in U_\varepsilon$. Then:

$$
\frac{f^\varepsilon(x + h e_i) - f^\varepsilon(x)}{h} = \frac{1}{\varepsilon^n} \int_\Omega \frac{1}{h} \left[ \eta\left(\frac{x + h e_i - y}{\varepsilon}\right) - \eta\left(\frac{x - y}{\varepsilon}\right)\right] f(y) \, dy
$$

$$
= \frac{1}{\varepsilon^n} \int_V \frac{1}{h} \left[ \eta\left(\frac{x + h e_i - y}{\varepsilon}\right) - \eta\left(\frac{x - y}{\varepsilon}\right)\right] f(y) \, dy
$$

for some open set $V \subset\subset U$. As

$$
\frac{1}{h} \left[ \eta\left(\frac{x + h e_i - y}{\varepsilon}\right) - \eta\left(\frac{x - y}{\varepsilon}\right)\right] \to \frac{1}{\varepsilon} \eta_{x_i}\left(\frac{x - y}{\varepsilon}\right)
$$

uniformly on $V$, the partial derivative $f^\varepsilon_{x_i}(x)$ exists and equals

$$
f^\varepsilon_{x_i}(x) = \int_\Omega \eta_{\varepsilon, x_i}(x - y) f(y) \, dy
$$

A similar argument shows that $D^\alpha f^\varepsilon(x)$ exists, and

$$
D^\alpha f^\varepsilon(x) = \int_\Omega D^\alpha \eta_\varepsilon(x - y) f(y) \, dy, \quad (x \in U_\varepsilon)
$$

for each multiindex $\alpha$. **Q.E.D.**
{% endfolding %}

{% note success no-icon %}
**Theorem 7 (Smoothness):**
Suppose $u \in C^2(\Omega)$ and $\Delta u = 0$ in $\Omega$. Then $u \in C^\infty(\Omega)$.
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
Let $\eta$ be the standard mollifier. Define

$$
u^\varepsilon(x) = \eta_\varepsilon * u = \int_{\Omega} \eta_\varepsilon(x-y) u(y) \, dy = \frac{1}{\varepsilon^n} \int_{B(x, \varepsilon)} \eta\left(\frac{x-y}{\varepsilon}\right) u(y) \, dy
$$

Using polar coordinates and the mean value property:

$$
\begin{aligned}
    u^\varepsilon(x) &= \frac{1}{\varepsilon^n} \int_{0}^{\varepsilon}\int_{\partial B(x, r)} \eta\left(\frac{r}{\varepsilon}\right) u(y) \, d\sigma dr \\
    &= \frac{1}{\varepsilon^n} \int_{0}^{\varepsilon} \eta\left(\frac{r}{\varepsilon}\right) \frac{n\alpha(n) r^{n-1}}{n\alpha(n) r^{n-1}} \int_{\partial B(x, r)} u(y) \, d\sigma dr\\
    &=\frac{1}{\varepsilon^n} \int_{0}^{\varepsilon} \eta\left(\frac{r}{\varepsilon}\right) n\alpha(n) r^{n-1}u(x) \, dr\\
    &= u(x)\frac{1}{\varepsilon^n} \int_{0}^{\varepsilon}\int_{\partial B(x,r)} \eta\left(\frac{r}{\varepsilon}\right) \, d\sigma dr\\
    &= u(x)\int_{B(x,\varepsilon)}\dfrac{1}{\varepsilon^{n}}\eta\left(\frac{r}{\varepsilon}\right) \, dr\\
    &= u(x)\in C^{\infty}(\Omega_{\varepsilon})
\end{aligned}
$$
**Q.E.D.**
{% endfolding %}

The stronger conclusion is that:

{% note success no-icon %}
**Theorem 8 (Analyticity):**
Assume $u$ is harmonic in $\Omega$, then $u$ is analytic in $\Omega$.
{% endnote %}

## Interior Estimate

{% note success no-icon %}
**Theorem 9 (Estimates on Derivatives):**
Assume $u$ is harmonic in $U$. Then:

$$
|D^\alpha u(x_0)| \leq \frac{C_k}{r^{n+k}} \| u \|_{L^1(B(x_0, r))}
$$

for each ball $B(x_0, r) \subset U$ and each multiindex $\alpha$ of order $|\alpha| = k$. 

Here:
$$
C_0 = \frac{1}{\alpha(n)}, \quad C_k = \frac{(2^{n+1}nk)^k}{\alpha(n)}, \quad (k = 1, 2, \dots)
$$
where $\alpha(n)$ is the volume of the unit ball in $\mathbb{R}^n$.
{% endnote %}

{% folding cyan, Proof (Click to expand) %}
**Proof:**
We argue by induction on $k$.

1. For $k = 0$:
$$
\text{LHS} = u(x_0), \quad \text{RHS} = \frac{1}{\alpha(n) r^n} \int_{B(x_0, r)} u
$$
Since $u$ is harmonic, by the mean value property:
$$
|u(x_0)| = \frac{1}{\alpha(n) r^n} \int_{B(x_0, r)} |u| \leq \frac{1}{\alpha(n) r^n} \int_{B(x_0, r)} |u|
$$

2. For $k = 1$:
$$
C_1 = \frac{2^n n}{\alpha(n)}
$$
Observe that $\frac{\partial u}{\partial x_i}$ is still harmonic, since $\Delta \frac{\partial u}{\partial x_i} = \frac{\partial}{\partial x_i} (\Delta u) = 0$.

By the mean value property:
$$
\frac{\partial u}{\partial x_i}(x_0) = \frac{1}{\alpha(n) r^n} \int_{B(x_0, r)} \frac{\partial u}{\partial x_i}
$$

Using the divergence theorem:
$$
\frac{\partial u}{\partial x_i}(x_0) = \frac{1}{\alpha(n) r^n} \int_{\partial B(x_0, r)} u \, \nu_i
$$
where $\nu_i$ is the $i$-th component of the outward unit normal. Then:
$$
\left| \frac{\partial u}{\partial x_i}(x_0) \right| \leq \frac{2^n}{\alpha(n) r^{n+1}} \| u \|_{L^1(B(x_0, r))}
$$

Combining these inequalities, we deduce:
$$
| \nabla u(x_0) | \leq \frac{2^n n}{\alpha(n) r^{n+1}} \| u \|_{L^1(B(x_0, r))}
$$

3. Inductive Step:
Assume the estimate holds for $k-1$. Then for $|\alpha| = k$, apply similar arguments using the derivatives of $u$, the divergence theorem, and scaling properties to obtain:
$$
|D^\alpha u(x_0)| \leq \frac{C_k}{r^{n+k}} \| u \|_{L^1(B(x_0, r))}
$$

Thus, the result follows by induction. **Q.E.D.**
{% endfolding %}