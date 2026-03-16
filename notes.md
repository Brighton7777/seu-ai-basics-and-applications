
## Lec 2 : Regression

> 所有的模型都是错误的，但有用。 

### 回归：Linear Regression 线性回归 

**评估函数的拟合程度时，为什么不使用预测值和实际值之差的绝对值，而是差的平方？**  
1. 使用平方能够有效惩罚偏差大的样本 
2. 二次函数连续可导的，而绝对值则在x=0不可导 

**使用Sum-squared error (SSE) 作为损失函数，以 $y = w x + b$ 为例，实际上就是二元函数的最小值。**

<p align="center"><img src="./images/lec2/linreg-sse-1.png" width=200><img src="./images/lec2/linreg-sse-2.png" width=200></p>
使用SSE的线性回归一定有最小值，但不一定唯一。


### Gradient Descent 梯度下降

**Modeling Recipe:**
- pick model
- pick loss
- fit model by running gradient descent

### 分类：Logistic Regression 逻辑回归

例子：**大众点评**评分系统
- Input: 一段评论
- Output: 好评/差评
1. 使用train data为每个词语训练出一个权重
2. 对句子进行评分
3. 设计 **Decision Boundaries** 决策边界，如score > 0 为好评

以上完成了“硬分类”问题，但"软分类"即对分类结果有多大把握呢？
将score映射到0到1区间，作为可信度。

**如何评估预测的准确程度？**
定义Loss为预测的概率和实际结果的差别。
例如在二分类问题中，$L(p,y) = -y\log p-(1-y)\log (1-p)$

**Cross Entropy 作为 Loss 时的梯度下降：**
<p align='center'><img src='./images/lec2/cross-entropy.png' width=90%></p>
因此，以Cross Entropy作为Loss函数的Model在训练时，会向着缩小预测概率与实际结果误差的方向修改参数。

### 多分类问题中的Cross-Entropy Loss

在二分类问题中，由于只需要输出预测其中一个类别的概率，交叉熵为
$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)
\right]
$$
而在多分类问题中，需要输出每个类别的概率，此时交叉熵为
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik}\log(p_{ik})
$$
通常将标签看作 **One-Hot Vector** 独热向量，因此在上面的式子中每个样本只有一个非零项。

## Lec 3 : Classification
### 分类问题的形式化表示
将距离分界面最近的数据点，称为**Support Vector**。
可以通过分界线到分界线两侧的Support Vector的"宽度"，判断一个分类是否"清晰"。
形式化地表示为：
$$
y_i(w^T x_i + b) \ge \frac{\rho}{2}
$$
其中 $\rho$ 表示分类界限。
几何间隙可表示为：
$$
\gamma_i = \frac{y_i (w^T x_i + b)}{\|w\|}
$$
优化目标为最小化“宽度”：
$$
max \frac{\rho}{\|w\|}
$$
进一步转化为最优化问题：

$$
\begin{aligned}
\min_{w,b} \quad & \frac{1}{2}\|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \ge 1, \quad i = 1,2,\dots,n
\end{aligned}
$$

进一步使用最优化方法解决。构造拉格朗日函数：
$$
\begin{aligned}
L(w,b,\lambda)
= \frac{1}{2}\|w\|^2 - \sum_{i=1}^{n} \lambda_i \left[
y_i(w^T x_i + b) - 1
\right]
\end{aligned}
$$
进而得到对偶问题：
$$
\begin{aligned}
\max_{\lambda} \quad
&
\sum_{i=1}^{n}\lambda_i - \frac{1}{2}
\sum_{i=1}^{n}
\sum_{j=1}^{n}
\lambda_i \lambda_j y_i y_j x_i^T x_j
\\
\text{s.t.}\quad
&
\lambda_i \ge 0
\\
&
\sum_{i=1}^{n}\lambda_i y_i = 0
\end{aligned}
$$

### 线性不可分问题
对于线性不可分的情况，可以采用映射到高维空间中，转化为线性可分问题。
在此给出引入 **松弛变量** $\xi$ 的方法——软间隔分类。
$$
\begin{aligned}
\min_{w,b,\xi} \quad 
& \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i \\
\text{s.t.} \quad 
& y_i (w^T x_i + b) \ge 1 - \xi_i, \quad i=1,\dots,n \\
& \xi_i \ge 0
\end{aligned}
$$
$\xi_i$也称作正则项，可以用来对抗过拟合，$C$是惩罚参数。

### 高维映射与kernel trick
当真正将数据映射到高维时，时间和空间复杂度是成倍增长的，而是否能在不增长复杂度的前提下，实现高维映射并分类？
观察最终的最优化问题，其中$x_i^T x_j$表示两个数据点的直接相乘。
例如，将$x\in\mathbb{R}$映射到$\hat{x}\in \mathbb{R}^d$后，$x_i^T x_j$转化为两个高维向量的内积$\langle \hat{x_i}, \hat{x_j} \rangle$，将内积展开有 
$$\langle \hat{x_i}, \hat{x_j} \rangle = K(x_i, x_j)$$
其中$K(x, y)$是一个标量函数，因此只需要$K(x, y)$能够计算，就能进行训练，而不需要真正地映射到高维。
将$K(x, y)$称为**kernel function**，这种方法成为**kernel trick**。

### Decision Tree
对于人类来说，分类往往会有顺序地考虑不同feature，进行排除和选择，这种分类方式实际上是在一颗决策树上进行遍历。
对于机器来说，这种方法是很有参考价值的，但机器并不会像人类一样学到一颗决策树。
一种比较“粗暴”的方式是对特征进行枚举，考虑按每种特征分类的`error`，选择`error`最小的特征作为分类方法，进入下一层继续分类。
在实际的使用中，并不会直接使用`error`，而是impurity。可以使用**entropy**度量impurity。
$$
H(x) = -\sum_{i=1}^{n} P(x=i)\log_2^{P(x=i)}
$$
进而定义**information gain**为父节点的entropy - 子结点的平均entropy。
$$
Gain(parent, sons) = H(parent) - \frac{1}{|sons|}\sum_{son\in sons}H(parent | son)
$$
取有最大information gain的feature进行split。