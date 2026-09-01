## System Model

- 基础知识
  - 传输公式:$\mathbf y = \mathbf H \mathbf W\mathbf s + \mathbf n$
    - 接收信号 $\mathbf y\in\mathbb C^{N_r\times 1}$
    - 信道矩阵 $\mathbf H\in\mathbb C^{N_r\times N_t}$
    - 预编码矩阵 $\mathbf W\in\mathbb C^{N_t\times N_l}$
    - 发射信号 $\mathbf s\in\mathbb C^{N_l\times 1}$
    - AWGN $\mathbf n\in\mathcal {CN}(\mathbf O_{N_r\times 1},\sigma_n^2\mathbf I_{N_r})$
  - 接收天线$N_r$，发射天线$N_t=N_1N_2$，传输流数$N_l$
  - $N_1,N_2$: 水平和垂直维度的极化振子数
  - $O_1,O_2$: 水平和垂直维度的过采样因子
- 空间波束矢量
  - $\mathbf v_l$: 水平方向相位偏转矢量，由水平索引$l$决定，长度为$N_1$
  - $\mathbf v_m$: 垂直方向相位偏转矢量，由垂直索引$m$决定，长度为$N_2$
  - $\mathbf v_{l,m} = \vec v_l \otimes \vec v_m$: 2D空间波束矢量，水平与垂直克罗内克积，长度为$N_1N_2$
- 极化共相因子
  - $\varphi_n=e^{j\pi n/2}$: 由索引$n$决定相位调节值
- 最终预编码矩阵
  - 单流: $\mathbf W = \frac{1}{\sqrt{2N_1N_2}}\begin{bmatrix} v_{l,m}\\ \varphi_n v_{l,m} \end{bmatrix}$
  - 多流(仅示意): $\mathbf W = \frac{1}{\sqrt{2N_1N_2}}\begin{bmatrix} v_{l,m}& \cdots& v_{l',m'}   & \cdots \ \\ \varphi_n v_{l,m} & \cdots & \varphi_n' v_{l',m'}& \cdots  \ \end{bmatrix}$
  
- 信道容量
  - 假设选择了索引$l,m,n$，得到了预编码矩阵$\mathbf W$
  - $C = \log_2\det \big(\mathbf I_{N_l} + \frac{1}{\sigma_n^2} \mathbf W^H\mathbf H^H\mathbf H\mathbf W \big)$

- 任务描述
  - 已知：$\mathbf H, \sigma_n^2$，以及$\mathbf W$的码本
  - 求：$l,m,n(,l',m',\cdots)$，或表示为对应的$\mathbf v, \varphi $


## Approach1 参数估计
### 1. $\mathbf v$索引的连续表示
- $\mathbf v_l = \begin{bmatrix}1& e^{j\frac{2\pi l}{O_1N_1}}&\cdots & e^{j\frac{2\pi l (N_1-1)}{O_1N_1}} \end{bmatrix}^T$，长度为$N_1$，$l\in\{0,1,\cdots,O_1N_1-1\}$ 
- $\mathbf v_m = \begin{bmatrix}1& e^{j\frac{2\pi m}{O_2N_2}}&\cdots & e^{j\frac{2\pi m (N_2-1)}{O_2N_2}} \end{bmatrix}^T$，长度为$N_2$，$m\in\{0,1,\cdots,O_2N_2-1\}$ 

换种写法，设$\theta_h = l/O_1N_1, \theta_v = m/O_2N_2$，有 $\theta_h,\theta_v\in(0,1)$：
- $\mathbf v_l = \begin{bmatrix}1& e^{j 2\pi \theta_h}&\cdots & e^{j2\pi \theta_h (N_1-1)} \end{bmatrix}^T, \mathbf v_m = \begin{bmatrix}1& e^{j2\pi \theta_v}&\cdots & e^{j2\pi \theta_v(N_2-1)} \end{bmatrix}^T$
- 将波束选择任务转化为寻求对连续值$\theta_h,\theta_v\in(0,1)$的寻优，然后再进行量化得到索引$l,m$
- 多流情况可以将其他波束建模为$\theta_h+\Delta\theta_{h1}+\cdots, \ \theta_v+\Delta\theta_{v1}+\cdots$

### 2. $\varphi$索引的连续表示(暂定)
$\varphi_n=e^{j\pi n/2}$，对$\varphi_n$寻优+量化，即可得到索引$n$

### 3. Loss与约束
- 网络输出：$\Theta = \{\theta_h,\theta_v,\Delta\theta_{hi},\Delta\theta_{vi},\varphi_{nj}\}$
- 主要约束：追求信道容量的最大化，将Loss设置为信道容量的负值，设最终预编码矩阵$\hat{\mathbf W} = \mathbf W(\Theta)$
    $$
    L_{cap} = -C = - \log_2\det \big(\mathbf I_{N_l} + \frac{1}{\sigma_n^2} \hat{\mathbf W}^H\mathbf H^H\mathbf H\hat{\mathbf W} \big)
    $$
- 次要约束，正交性正则化，追求$\Delta \theta_{hi} N_1, \Delta\theta_{vi} N_2$尽量为整数，引入周期性的正弦平方惩罚函数
    $$
    \mathcal{L}_{reg} = \sum_{i} \left[ \sin^2(\pi \Delta\theta_{hi} N_1) + \sin^2(\pi \Delta\theta_{vi} N_2) \right]
    $$
- 总Loss为两个约束的加权相加。


### 4. 训练
无监督，训练集为已知SNR，已知RI情况下的信道矩阵$\mathbf H$


### 5. 推理
在实际部署推理时，对输出值进行舍入，即为码本索引

