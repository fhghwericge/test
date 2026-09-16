# Search-Free PMI Selection — 项目设计文档

---

## 一、项目概述

### 1.1 目标

给定信道矩阵 **H** 与信噪比 SNR，以**无监督**方式直接预测 3GPP R19 Type I 码本预编码参数集 **Θ**，最大化总信道容量 *C*。训练阶段使用连续参数保证梯度流通；推理阶段量化到离散码本索引。

### 1.2 最小可行方案约束

| 约束项 | 值 | 说明 |
|--------|-----|------|
| 流数 υ (RI) | 2 | 固定双流传输 |
| 独立波束组数 K | 2 | 每流独立选择波束 |
| 天线配置 | N₁=8, N₂=2, 双极化 → N_t=32 | 3GPP R19 UPA |
| 接收天线 N_r | 4 | 2×双极化 |
| 子载波数 N_RB | 32 | 2.88 MHz 子带 |

### 1.3 核心思路

```
H + SNR → PMIModel → Θ(连续) → Ŵ(可微) → CapacityLoss → 优化
                        ↓
                  Quantizer → Θ_q, W_q → Capacity (推理评估)
```

- **训练**：网络直接输出连续参数 Θ，经 Codebook 构建可微预编码矩阵 Ŵ，以负容量为主要损失端到端优化，正则化项引导参数向离散网格收敛。
- **推理**：将连续 Θ 量化到离散码本网格，重建 W_q 评估真实码本容量。

---

## 二、系统架构

### 2.1 端到端数据流
**训练：**
```
  H_real (B,32,32,4,2)    SNR (B,1) 
          ↓                    ↓
  [InputPreprocessor]    [FiLMGenerator]  
  tokens (B,32,128)    γ(B,128), β(B,128)
          ↓                    ↓
┌─────────────────────────────────────────────┐
│ [TransformerBackbone] → encoded (B,32,128) │
│                      ↓                      │
│ [OutputHead_ri2] → Θ (B,6)                 │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│            Codebook (码本)                   │
│ Θ (B,6) → Ŵ (B,32,2) complex64             │
└─────────────────────┬───────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         CapacityLoss (损失函数)              │
│ Ŵ + H_real + Θ + SNR → L_total (标量)      │
└─────────────────────────────────────────────┘
```
**推理：**

```
Θ (B,6) → [Quantizer] → Θ_q (B,6)  → [Codebook] → W_q (B,32,2) → 评估容量
```

### 2.2 模块依赖关系

```
train.py (训练入口)
    ├─→ utils.config       (加载 YAML → ProjectConfig)
    ├─→ utils.dataset      (构建 train/val/test DataLoader)
    ├─→ utils.quantizer    (推理量化)
    ├─→ utils.utils        (日志、打印、参数量统计、理想上界)
    ├─→ models.pmi_model   (PMIModel = Preprocessor + FiLM + Backbone + Head)
    ├─→ codebook.codebook  (Θ → Ŵ)
    ├─→ losses.capacity    (Ŵ + H + Θ → L)
    ├─→ train.trainer      (训练循环、早停、checkpoint，用 val 验证)
    ├─→ eval.visualize     (可视化)
    └─→ eval.evaluate_fn   (推理评估)

evaluate.py (测试集评估入口)   → 复用 eval.evaluate_fn
diag_variance.py (方差诊断)     → 复用 models.pmi_model
```

### 2.3 模型参数量

| 子模块 | 参数量 | 说明 |
|--------|--------|------|
| preprocessor | 37,120 | Linear(256→128, bias=False) + LayerNorm(128) + pos_embed(32×128) |
| film_generator | 16,896 | Linear(1→64) + LayerNorm(64) + ReLU + Linear(64→256) |
| backbone | 396,544 | 2 层 TransformerEncoderLayer (D=128, H=4, FFN=512) |
| head | 2,164,102 | MLP (4096→512→128→6) |
| **PMIModel 总计** | **2,614,662** | 全部可训练 |

---

## 三、模块详述

---

### 3.1 信道数据 (Channel Data)

> **源文件**：`utils/dataset.py`、`data/data_gen.ipynb`、`data/data_gen.py`、`data/data_process.ipynb`
>
> **配置**：`AntennaConfig`

#### 物理参数符号表

| 数学符号 | 含义 | 代码变量 | Config 字段 | 默认值 |
|----------|------|----------|-------------|--------|
| N₁ | 水平天线数 (单极化) | `cfg.N1` | `AntennaConfig.N1` | 8 |
| N₂ | 垂直天线数 (单极化) | `cfg.N2` | `AntennaConfig.N2` | 2 |
| N_t | 总发射天线数 (双极化) | `cfg.Nt` | `AntennaConfig.Nt` | 32 (= 2×N₁×N₂) |
| N_r | 接收天线数 | `cfg.Nr` | `AntennaConfig.Nr` | 4 |
| N_RB | 频域子载波数 | `cfg.N_RB` | `AntennaConfig.N_RB` | 32 |
| SNR_dB | 默认评估信噪比 (dB) | `cfg.SNR_dB_Low` | `AntennaConfig.SNR_dB_Low` | 20.0 |
| SNR_dB | 训练信噪比上限 (dB) | `cfg.SNR_dB_High` | `AntennaConfig.SNR_dB_High` | 20.0 |
| O₁ | 水平过采样因子 (仅量化用) | `cfg.O1` | `AntennaConfig.O1` | 4 |
| O₂ | 垂直过采样因子 (仅量化用) | `cfg.O2` | `AntennaConfig.O2` | 4 |
| υ | 传输流数 (RI) | `cfg.upsilon` | `AntennaConfig.upsilon` | 2 |


#### 数据生成

- **信道模型**：Sionna 3GPP TR 38.901 **TDL-A**，delay_spread=300 ns，f_c=3.5 GHz，速度 5 km/h
- **归一化**：`normalize=True`，确保 E[|h|²]≈1
- **原始格式**：complex64，约 200,000 样本
- **预处理**：拆分为实/虚部 → float32 npy

#### 数据格式

按 block 轴 80/10/10 划分为训练/验证/测试三份，维度顺序 `[样本, Nt, N_RB, Nr, 实虚]`，末轴 `[实部, 虚部]`。

| 文件 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `train_data.npy` | (160000, 32, 32, 4, 2) | float32 | 训练集 |
| `val_data.npy` | (20000, 32, 32, 4, 2) | float32 | 验证集 |
| `test_data.npy` | (20000, 32, 32, 4, 2) | float32 | 测试集 |
| `train_idx.npy` / `val_idx.npy` / `test_idx.npy` | (N,) | int64 | 对应样本下标 |


#### 波束域预处理 (Beam Transform)

> **开关**：`DebugConfig.beam_transform`（`configs/default.yaml` → `debug.beam_transform`）
>
> **功能**：在 DataLoader 加载阶段，对信道矩阵的天线维度执行正交 2D DFT 变换，输出维度保持不变。

**变换矩阵构造**：

1. 归一化 1D DFT 矩阵：

$$[\mathbf{D}_h]_{k, n} = \frac{1}{\sqrt{N_1}} e^{j \frac{2\pi k n}{N_1}}, \quad [\mathbf{D}_v]_{m, l} = \frac{1}{\sqrt{N_2}} e^{j \frac{2\pi m l}{N_2}}$$

2. 单极化 2D 波束空间基底（Kronecker 积）：

$$\mathbf{D}_{2D} = \mathbf{D}_v \otimes \mathbf{D}_h \in \mathbb{C}^{N_1 N_2 \times N_1 N_2} = \mathbb{C}^{16 \times 16}$$

3. 双极化分块对角全局变换矩阵：

$$\mathbf{W} = \begin{bmatrix} \mathbf{D}_{2D} & \mathbf{0} \\ \mathbf{0} & \mathbf{D}_{2D} \end{bmatrix} \in \mathbb{C}^{N_t \times N_t} = \mathbb{C}^{32 \times 32}$$

4. 天线域→波束域变换，使用 $\mathbf{W}^H$：

$$\tilde{\mathbf{H}}_c = \mathbf{W}^H \mathbf{H}_c$$

**实现要点**：

- 变换矩阵在 `PreprocessedChannelDataset.__init__` 中预计算并缓存，`__getitem__` 中对逐样本执行矩阵乘法
- $\mathbf{W}^H$ 构造方式：先按正指数构造 $\mathbf{W}$，再取共轭转置 `W.conj().T`
- Kronecker 积通过广播外积实现：`h[:, None] * v[None, :]` → reshape
- 每个 `__getitem__` 调用的变换步骤：
  1. 实虚合并：`H_c = torch.complex(h_real[...,0], h_real[...,1])` → `(Nt, N_RB, Nr)`
  2. 展平后两维：`H_flat = H_c.reshape(Nt, -1)` → `(32, 128)`
  3. 左乘变换：`H_beam_flat = W_H @ H_flat` → `(32, 128)`
  4. 恢复形状：`H_beam = H_beam_flat.reshape(Nt, N_RB, Nr)`
  5. 实虚拆回：`h_real = torch.stack([H_beam.real, H_beam.imag], dim=-1)` → `(32, 32, 4, 2)`

**输出维度**：与输入完全一致 `(Nt, N_RB, Nr, 2) = (32, 32, 4, 2)`，对下游模块透明。

**配置控制**：

| 字段 | 含义 | 默认值 | 说明 |
|------|------|--------|------|
| `debug.beam_transform` | 是否启用波束域预处理 | `false` | `true` 时 Dataset 在 `__getitem__` 中自动执行变换 |

启用/禁用仅影响 DataLoader 输出的数据内容，不改变维度、不改变训练/推理流程中的任何其他模块。

**符号-变量映射表**：

| 数学符号 | 含义 | 代码变量 | 来源 |
|----------|------|----------|------|
| $\mathbf{D}_h$ | 水平归一化 1D DFT | `D_h` | `_build_beam_transform_matrix()` 内局部变量 |
| $\mathbf{D}_v$ | 垂直归一化 1D DFT | `D_v` | `_build_beam_transform_matrix()` 内局部变量 |
| $\mathbf{D}_{2D}$ | 单极化 2D 波束基底 | `D_2d` | `_build_beam_transform_matrix()` 内局部变量 |
| $\mathbf{W}$ | 双极化全局变换矩阵 | `W_full` | `_build_beam_transform_matrix()` 内局部变量 |
| $\mathbf{W}^H$ | 分析变换矩阵 (天线域→波束域) | `self.W_H` | `PreprocessedChannelDataset` 实例属性 |
| $\tilde{\mathbf{H}}_c$ | 波束域复数信道 | `H_beam` | `__getitem__` 内局部变量 |


---

### 3.2 码本与预编码 (Codebook & Precoding)

> **源文件**：`codebook/codebook.py`（训练用）、`utils/quantizer.py`（推理用）
>
> **配置**：`AntennaConfig`

#### 符号-变量映射表

| 数学符号 | 含义 | 代码变量 / 函数 | Config 字段 |
|----------|------|-----------------|-------------|
| θ_h | 水平角度参数 ∈ (0,1) | `theta[:, 0]` (流1), `theta[:, 3]` (流2) | — |
| θ_v | 垂直角度参数 ∈ (0,1) | `theta[:, 1]` (流1), `theta[:, 4]` (流2) | — |
| θ_φ | 极化相位参数 ∈ (0,4) | `theta[:, 2]` (流1), `theta[:, 5]` (流2) | — |
| h(θ_h) | 水平 1D 波束 ∈ ℂ^{N₁} | `beam_vector()` 内局部变量 | — |
| v(θ_v) | 垂直 1D 波束 ∈ ℂ^{N₂} | `beam_vector()` 内局部变量 | — |
| v_{l,m} | 2D 空间波束 = h⊗v ∈ ℂ^{N₁N₂} | `beam_vector()` 返回值 | — |
| φ_n | 极化共相因子 = e^{jπn/2} | `polarization_phase()` 返回值 | — |
| w_k | 单流预编码向量 = [v; φ·v]/√P | `single_stream_precoder()` 返回值 | — |
| Ŵ | 预编码矩阵 ∈ ℂ^{B×Nt×υ} | `compose_W()` / `forward()` 返回值 | — |
| K | 归一化系数 (波束组数) | `self.K` | `AntennaConfig.K` |
| P_CSI-RS | 功率归一化因子 | `self.P_CSI_RS` | `AntennaConfig.P_CSI_RS` |

#### 实现要点
对于输出参数集$\Theta = \big \{\{\theta_{hk}, \theta_{vk}, \theta_{\varphi k}\}\big\}_{k=1}^{K}, $取一组参数$\theta_h,\theta_v,\theta_{\varphi}$

- **空间波束矢量**：

  $$\mathbf h(\theta_h) = [1,\ e^{j2\pi\theta_h},\ \ldots,\ e^{j2\pi(N_1-1)\theta_h}]^T$$

  $$\mathbf v(\theta_v) = [1,\ e^{j2\pi\theta_v},\ \ldots,\ e^{j2\pi(N_2-1)\theta_v}]^T$$

  $$\mathbf v_{2D} = \mathbf h \otimes \mathbf v$$

  Kronecker 积通过广播外积实现：`h(B,N₁,1) * v(B,1,N₂) → reshape → (B, N₁N₂)`。

- **极化共相因子** ：

  $$\varphi(\theta_\varphi) = e^{j\pi\theta_\varphi / 2}$$

**多流组装**：

1. 从 Θ 交错提取两流参数：`[θ_h1,θ_v1,θ_φ1, θ_h2,θ_v2,θ_φ2]`
2. 每流独立：`beam_vector → polarization_phase → single_stream_precoder` → (B, Nt, 1)
3. 拼接 → (B, Nt, 2)，除以 √K → 总归一化系数 1/√(K·P_CSI-RS)
$$\hat{\mathbf W}(\Theta) = \frac{1}{\sqrt{K \cdot P_{\text{CSI-RS}}}} \begin{bmatrix} \mathbf v_{l_1,m_1} & \mathbf v_{l_2,m_2} \\ \varphi_{n_1} \mathbf v_{l_1,m_1} & \varphi_{n_2} \mathbf v_{l_2,m_2} \end{bmatrix}$$


#### 量化 (Quantizer)

> 对于输出参数集$\Theta = \big \{\{\theta_{hk}, \theta_{vk}, \theta_{\varphi k}\}\big\}_{k=1}^{K}, $ 映射到离散码本网格，仅推理时使用，脱离计算图。

**量化公式**：
- 单流约束：
  $$\theta_h' = \frac{\mathrm{round}(\theta_h \cdot O_1 N_1) \bmod (O_1 N_1)}{O_1 N_1}
  \\ \theta_v' = \frac{\mathrm{round}(\theta_v \cdot O_2 N_2) \bmod (O_2 N_2)}{O_2 N_2}\\ \theta_\varphi' = \mathrm{round}(\theta_\varphi) \bmod 4$$
- 多流情况下的约束 - 需要保证细索引相同以保证正交：
  对于水平或垂直的波束参数，有：
  $$ f_i = \mathrm{round}\left( \frac{O_i}{2\pi} \angle \sum_{k=1}^K e^{j2\pi \theta_{ik} N_i} \right) \bmod O_i, \quad i \in \{1, 2\}
  \\ \theta_{ik}' = \frac{\left( \mathrm{round}(\theta_{ik} N_i - f_i/O_i) \bmod N_i \right) O_i + f_i}{O_i N_i} $$
  相位参数不变


<!-- 量化后参数严格落在合法区间：

- $\theta_{hk}' \in \{0, \frac{1}{O_1N_1}, \dots, \frac{O_1N_1-1}{O_1N_1}\} \subset [0, 1)$
- $\theta_{vk}' \in \{0, \frac{1}{O_2N_2}, \dots, \frac{O_2N_2-1}{O_2N_2}\} \subset [0, 1)$
- $\theta_{\varphi k}' \in \{0, 1, 2, 3\}$ -->

**输出**：

| 输出 | 形状 | 说明 |
|------|------|------|
| Θ_q | (B, 6) | 量化后连续参数，可重新喂入 Codebook |
| W_q | (B, Nt, υ) | 量化预编码矩阵 (经 Codebook 重建) |

**量化容量评估**：对同一样本分别计算连续容量 C_cont = -L_cap(Ŵ) 与量化容量 C_quant = -L_cap(W_q)，散点图分析量化损失。平均 gap = mean(C_cont - C_quant) 反映码本离散化代价。

---

### 3.3 损失函数 (Loss)

> **源文件**：`losses/capacity.py`
>
> **配置**：`LossConfig`

#### 组合损失公式

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{cap}}}_{\text{容量损失}} + \lambda_1 \underbrace{\mathcal{L}_{v\_\text{reg}}}_{\text{波束正交性}} + \lambda_2 \underbrace{\mathcal{L}_{\varphi\_\text{reg}}}_{\text{相位对齐}}$$

<!-- $$\mathcal{L} = \underbrace{\mathcal{L}_{\text{cap}}}_{\text{容量损失}} + \lambda_1 \underbrace{\mathcal{L}_{v\_\text{reg}}}_{\text{波束正交性}} + \lambda_2 \underbrace{\mathcal{L}_{\varphi\_\text{reg}}}_{\text{相位对齐}} + \lambda_3 \underbrace{\mathcal{L}_{\text{repel}}}_{\text{波束排斥力}}$$ -->

#### MSE 协方差矩阵与容量

$$\mathbf E_b = \left( \mathbf I_\upsilon + \text{SNR} \cdot \hat{\mathbf W}^H \mathbf H_b^H \mathbf H_b \hat{\mathbf W} \right)^{-1},\quad \text{SINR}_{b,i}  = \frac{1}{[\mathbf E_b]_{i,i}}-1$$
$$C = \sum_{b=1}^{N_{RB}} \sum_{i=1}^{\upsilon} \log_2 \left( 1 + \text{SINR}_{b,i} \right) = -\sum_{b=1}^{N_{RB}} \sum_{i=1}^{\upsilon} \log_2 \left( [\mathbf E_b]_{i,i} \right)$$
$$\mathcal{L}_{\text{cap}} =-C = \sum_{b=1}^{N_{RB}} \sum_{i=1}^{\upsilon} \log_2 \left( [\mathbf E_b]_{i,i} \right)$$


> 注：$[\cdot]_{i,i}$ 表示矩阵主对角线上的第 $i$ 个元素。

#### 正则化项物理含义

**L_v_reg — 波束正交性**：

$$\mathcal{L}_{v\_\text{reg}} = \sin^2\big(\pi(\theta_{h1}-\theta_{h2})N_1\big) + \sin^2\big(\pi(\theta_{v1}-\theta_{v2})N_2\big)$$

- (θ_h1-θ_h2)·N₁ 为整数时 = 0 → 两波束对应不同离散网格点（正交）
- 为半整数时 = 1 → 两波束指向相邻网格点（最不正交）
- 最小化此项促使不同流选择空间间距最大的波束

**L_φ_reg — 相位对齐**：

$$\mathcal{L}_{\varphi\_\text{reg}} = \sin^2(\pi\theta_{\varphi1}) + \sin^2(\pi\theta_{\varphi2})$$

- θ_φ 为整数时 = 0 → 完美离散化
- 为半整数时 = 1 → 离离散点最远
- 最小化此项促使连续参数自然收敛到 {0, 1, 2, 3}

<!-- **L_repel — 波束排斥力**（抑制 Δθ=0 的波束重合退化）：

$$\mathcal{L}_{\text{repel}} = \mathrm{ReLU}(1 - |\Delta\theta_h| \cdot N_1)^2 + \mathrm{ReLU}(1 - |\Delta\theta_v| \cdot N_2)^2$$

- 当两波束间距 < 1 个 DFT 网格时产生二次排斥力，间距 ≥ 1 网格后惩罚归零
- Δθ 取圆周距离 `min(|Δθ|, 1-|Δθ|)`，处理 θ 在 0/1 边界环绕
- 与 L_v_reg 互补：L_v_reg 在 Δθ=0（退化零点）取 0，L_repel 在 Δθ=0 取最大值 1 -->



#### 符号-变量映射表

| 数学符号 | 含义 | 代码变量 | Config 字段 |
|----------|------|----------|-------------|
| E_b | MSE 协方差矩阵 | `mse_covariance()` 返回值 | — |
| Ŵ | 预编码矩阵 | `W` (函数参数) | — |
| H_b | 第 b 个 RB 信道 | `H_c` permute 后 (B, N_RB, Nt, Nr) | — |
| SNR | 线性信噪比 | `snr_linear` |  `AntennaConfig.SNR_dB_Low/High` |
| C | 信道容量 | `-detail["L_cap"]` | — |
| L_cap | 负容量 = -C | `capacity_loss()` 返回值 | — |
| L_v_reg | 波束正交性正则化 | `v_reg()` 返回值 | — |
| L_φ_reg | 相位对齐正则化 | `phi_reg()` 返回值 | — |
| L_repel | 波束排斥力正则化 | `repel_reg()` 返回值 | — |
| λ₁ | L_v_reg 权重 | `self.lambda1` | `LossConfig.lambda1` |
| λ₂ | L_φ_reg 权重 | `self.lambda2` | `LossConfig.lambda2` |
| ε | 数值稳定抖动 | `self.eps` | `LossConfig.eps` |
<!-- | λ₃ | L_repel 权重 | `self.lambda3` | `LossConfig.lambda3` | -->



#### 数值稳定性设计

1. **矩阵求逆前**：加 ε·I_υ 防止病态矩阵导致梯度 NaN（`LossConfig.eps = 1e-6`）
2. **log2 运算前**：加 ε 抖动防止 log(0)
3. **SNR 灵活传入**：训练时每个样本独立 SNR；默认时由 `AntennaConfig.SNR_dB_High` 生成

#### 接口

```python
cap_loss = CapacityLoss(cfg.loss, cfg.antenna)
L_total, detail = cap_loss(W, H_real, theta, snr=snr_linear, return_components=True)
# detail = {"L_cap": ..., "L_v": ..., "L_phi": ..., "L_repel": ..., "L_total": ...}
```

---

### 3.4 AI 模型 (Neural Network Model)
![alt text](image-1.png)

> **源文件**：`models/pmi_model.py`、`models/preprocessor.py`、`models/backbone.py`、`models/film.py`、`models/heads.py`
>
> **配置**：`ModelConfig`

#### 整体组装 (PMIModel)

```python
class PMIModel(nn.Module):
    def __init__(self, model_cfg, antenna_cfg):
        self.preprocessor    = InputPreprocessor(model_cfg, antenna_cfg)
        self.film_generator  = FiLMGenerator(model_cfg)
        self.backbone        = TransformerBackbone(model_cfg)
        self.head            = build_head(model_cfg, antenna_cfg.Nt)

    def forward(self, H_real, snr=None) -> Theta:  # (B,6)
```

SNR 为 None 时 fallback 为 30dB 全1张量（推理默认值）。

#### 符号-变量映射表

| 数学符号 | 含义 | 代码变量 | Config 字段 | 默认值 |
|----------|------|----------|-------------|--------|
| H_real | 原始信道矩阵 | `H_real` (输入) | — | (B,32,32,4,2) |
| D | Token 特征维度 (d_model) | `model_cfg.D` | `ModelConfig.D` | 128 |
| tokens | Token 序列 | `preprocessor()` 返回值 | — | (B, 32, 128) |
| γ | FiLM 缩放因子 (残差式) | `gamma` | — | (B, 128) |
| β | FiLM 偏移因子 | `beta` | — | (B, 128) |
| L | Transformer 层数 | `cfg.n_layers` | `ModelConfig.n_layers` | 2 |
| H_attn | 注意力头数 | `cfg.n_heads` | `ModelConfig.n_heads` | 4 |
| Θ | 连续预编码参数 | `head()` 返回值 | — | (B, 6) |

#### InputPreprocessor — 输入预处理与 Tokenization

**功能**：将原始信道矩阵转为 Transformer 可接受的 Token 序列。

**处理流水线**：

```
H_real (B, Nt=32, N_RB=32, Nr=4, 2)
  → reshape → (B, Nt, N_RB×Nr×2 = 256)
  → Linear(256, D=128, bias=False) → (B, Nt, D)
  → LayerNorm(D) → (B, Nt, D)
  → + pos_embed(1, Nt, D) → (B, Nt, D)
```

- 每个 TX 天线的 (子载波×接收天线×实虚) 全部特征视为一个"词"
- 位置编码长度 Nt，编码天线的空间索引信息，初始化尺度 0.02
- Linear 后接 LayerNorm 归一化特征

#### FiLMGenerator — SNR 条件调制

**功能**：将 SNR 标量嵌入为逐特征调制参数。

**SNR 预处理**：输入先取对数并平移 `snr_norm = log10(snr + 1e-8) - 1.0`，将线性 SNR 压缩到近似 [-1,1] 区间。

**网络**：

```
snr (B, 1) 
  → log10 
  → Linear(1, hidden=64) 
  → LayerNorm(64) 
  → ReLU 
  → Linear(64, 2D=256) 
  → split 
  → γ(B,D), β(B,D)
```

**FiLM 调制方式（残差设计）**：

$$x' = (1 + \gamma) \cdot x + \beta$$

γ ≈ 0 时 x' ≈ x，不阻断信号流与梯度。

**初始化策略**：末端 Linear 权重 `uniform_(-1e-4, 1e-4)` + bias `zeros_`，确保训练初期 FiLM ≈ 恒等映射。

#### TransformerBackbone — 自注意力主干

**结构**：2 层 Pre-Norm TransformerEncoderLayer，手动循环（非 `nn.TransformerEncoder` 封装），以便层间插入 FiLM。

**每层**：

```
x → LayerNorm → MultiHeadSelfAttn →残差→ LayerNorm → FFN →残差→ x'
x' → FiLM: x'' = (1+γ.unsqueeze(1)) * x' + β.unsqueeze(1) → x''
```

- `batch_first=True`, `norm_first=True` (Pre-Norm)
- `activation="gelu"`, FFN 隐层 = D×ffn_ratio = 512
- γ, β 的 unsqueeze(1) 使 (B,D) 广播到 (B,Nt,D)

#### OutputHead_ri2 — 输出约束头

**功能**：将全特征矩阵解码为物理约束参数 Θ。

**MLP 结构**：

```
encoded (B, Nt, D) → flatten → (B, Nt×D = 4096)
  → Linear(4096, mid=512) → GELU
  → Linear(512, D=128) → GELU
  → Linear(128, 6)
```

**参数约束（可导）**：

| 参数 | 约束函数 | 值域 | 代码 |
|------|----------|------|------|
| θ_h | sigmoid | (0, 1) | `torch.sigmoid(raw[:, 0:1])` |
| θ_v | sigmoid | (0, 1) | `torch.sigmoid(raw[:, 1:2])` |
| θ_φ | 4·sigmoid | (0, 4) | `4.0 * torch.sigmoid(raw[:, 2:3])` |

参数按流交错排列：`[θ_h1, θ_v1, θ_φ1, θ_h2, θ_v2, θ_φ2]`。

输出层 `Linear(D, 6)` 的 bias 初始化时将两流偏置拉开（θ_h: -1.0/+1.0，θ_v: -0.5/+0.5，θ_φ: 0.0/+0.5），使初始两流波束分离。

**HEAD_REGISTRY**：注册机制支持扩展。当前仅 `"ri2"`，后续添加 RI=1/3/4 头无需改动主干。

#### ModelConfig 字段速查

| 字段 | 含义 | 默认值 |
|------|------|--------|
| D | 特征维度 (d_model) | 128 |
| n_heads | 注意力头数 | 4 |
| n_layers | Transformer 层数 | 2 |
| ffn_ratio | FFN 扩展比 | 4 |
| dropout | 残差/FFN Dropout | 0.1 |
| attn_dropout | 注意力权重 Dropout | 0.1 |
| head_name | 输出头注册名 | "ri2" |
| film_hidden_dim | FiLM 隐层维度 | 64 |
| head_intermediate_dim | 输出头 MLP 中间层 | 512 |

---

### 3.5 配置管理 (Configuration)

> **源文件**：`utils/config.py`、`configs/default.yaml`

#### Dataclass 层级

```
ProjectConfig (顶层)
  ├── AntennaConfig   —— 物理天线参数
  ├── ModelConfig     —— 网络架构超参
  ├── LossConfig      —— 损失函数超参 (λ1, λ2, λ3, eps)
  ├── TrainConfig     —— 训练流程超参
  └── TestConfig      —— 推理评估超参
```


#### TestConfig 字段速查

| 字段 | 含义 | 默认值 | 备注 |
|------|------|--------|------|
| device | 评估设备 | None | None=自动检测 |
| num_test_samples | 测试样本数上限 | None | None=全部 |
| batch_size | 批大小 | 64 | — |


#### TrainConfig 字段速查

| 字段 | 含义 | 默认值 | 备注 |
|------|------|--------|------|
| device | 训练设备 | None | None=自动检测, "cuda"/"cpu" |
| num_train_samples | 训练样本数上限 | None | None=全部 |
| num_test_samples | 测试样本数上限 | None | None=全部 |
| lr | 初始学习率 | 1e-3 | AdamW |
| weight_decay | 权重衰减 | 1e-4 | AdamW |
| batch_size | 批大小 | 64 | — |
| epochs | 最大训练轮数 | 100 | 受早停约束 |
| warmup_ratio | LR 预热比例 | 0.1 | 前 10% 步线性升 |
| max_grad_norm | 梯度裁剪范数 | 1.0 | 防梯度爆炸 |
| patience | 早停耐心值 | 20 | 连续 N epoch 无改善则停 |

#### 设备与样本数配置化

设备 (`device`) 和样本量 (`num_train_samples` / `num_test_samples`) 均从 `TrainConfig` 读取，不再通过命令行 `--device` 传参。设置上限机制：`min(num_samples, actual_total)`，防止请求数超过实际数据量。

---

## 四、目录结构

```
Workspace/
│
├── train.py                     # 训练入口：组装模块、训练、验证集评估、可视化
├── evaluate.py                  # 测试集推理评估入口
├── evaluate.ipynb               # 交互式推理评估 + 链路验证 (含原 playground 验证步骤)
├── diag_variance.py             # 特征方差诊断 (forward hook)
├── test_dataset.py              # DataLoader 健康检查
├── requirements.txt             # 依赖 (待填充)
│
├── configs/
│   └── default.yaml             # 生产配置 (antenna/model/loss/train/test)
│
├── data/
│   ├── data_gen.ipynb           # Sionna 信道生成
│   ├── data_gen.py              # 信道生成脚本 (TF-based)
│   ├── data_process.ipynb       # 预处理: raw → train/val/test npy
│   ├── fulldata.npy             # 原始信道数据
│   ├── train/val/test_data.npy  # 训练/验证/测试集 (160k/20k/20k)
│   └── train/val/test_idx.npy   # 对应样本下标
│
├── models/
│   ├── pmi_model.py             # PMIModel: 端到端组装
│   ├── preprocessor.py          # InputPreprocessor: H → tokens (含 LayerNorm)
│   ├── film.py                  # FiLMGenerator: SNR → γ, β (含 log 归一化)
│   ├── backbone.py              # TransformerBackbone: 自注意力 + FiLM
│   └── heads.py                 # OutputHead_ri2 + HEAD_REGISTRY
│
├── codebook/
│   └── codebook.py              # Codebook: Θ → Ŵ (可微预编码矩阵)
│
├── losses/
│   └── capacity.py              # CapacityLoss: 容量 + 3 项正则化组合
│
├── utils/
│   ├── config.py                # dataclass 配置 + YAML 加载
│   ├── dataset.py               # Dataset (mmap) + DataLoader 工厂
│   ├── quantizer.py             # Quantizer: Θ → Θ_q, S, W_q (推理用)
│   └── utils.py                 # 日志/打印/参数量统计/理想上界
│
├── train/
│   └── trainer.py               # Trainer: 训练循环、早停、checkpoint
│
├── eval/
│   ├── visualize.py             # 可视化
│   └── evaluate_fn.py           # 推理评估公共函数 (train/evaluate 共用)
│
├── checkpoints/                 # 模型 checkpoint 输出目录
│   └── best_model.pth
│
├── figures/                     # 可视化图片输出目录
│   ├── train_loss.png / val_capacity.png / diagnostics.png
│   ├── theta_distribution.png / quantization_analysis.png
│
└── Docs/
    ├── 项目设计.md              # 本文档
    ├── 实验日志.md              # 实验记录
    └── Logs/                    # 每次运行的日志文件
```

