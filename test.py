# # 信道数据处理：`fulldata.npy`

# 数据由 `data_gen.ipynb` 生成（TDL-A，delay_spread=300ns，fc=3.5GHz，速度 5km/h，空间不相关 corr_list=[[0,0,0,0]]）。

# 原始数组 `[1, 1000, 2, 4, 32, 100, 32]`，dtype complex64，约 6.1 GiB，**必须用 mmap 加载**。已核对 E[|h|²]≈1（normalize=True），无 NaN/Inf。

# | 轴 | 大小 | 含义 |
# |---|---|---|
# | 0 | 1 | 配置组合（corr × tdl × delay_spread） |
# | 1 | 1000 | seed(10) × batch(100) |
# | 2 | 2 | batch_size |
# | 3 | 4 | RX 天线（2RX × 双极化） |
# | 4 | 32 | TX 天线（N1×N2=16 × 双极化） |
# | 5 | 100 | 时间步（T_sample=0.02s，共 2s） |
# | 6 | 32 | 子载波（fft_size，子载波间隔 2.88MHz） |

# 处理流程：合并轴 0/1/2/5 得到 N=200000 个样本 → 可视化 → 按 block 80/10/10 划分 → 转为 `[N, TX, 子载波, RX, 2]` 分块写出。


import os
import numpy as np
import matplotlib.pyplot as plt

fpath = None
# name = 'TDLA_32T4R_16_1_16RB_30k_300ns_XPL_5km_sionna_L_pred1.npy'
name = 'fulldata.npy'
for p in [f"./{name}", f"./Workspace/data/{name}", f"../data/{name}"]:
    if os.path.exists(p):
        fpath = p
        break
if fpath is None:
    raise FileNotFoundError("fulldata.npy not found")

h = np.load(fpath, mmap_mode="r")
print("file  :", os.path.abspath(fpath))
print("shape :", h.shape)
print("dtype :", h.dtype)
print("size  : %.2f GiB" % (h.nbytes / 1024**3))



# ## 1. 样本索引

# 轴 0/1/2/5 合并为样本量 **N = 1×1000×2×100 = 200000**，逻辑形状 `[N, 子载波, RX, TX] = [200000, 32, 4, 32]`。

# 磁盘数组是 C 序，直接 `transpose + reshape` 会把整份 6.1 GiB 复制进内存，因此不物化：`get_sample(i)` 惰性索引（单样本 16 KB）。

# 样本序：`i = ((blk×2)+b)×100 + t`，blk∈[0,1000)、b∈[0,2)、t∈[0,100)。


N_BLK, N_B, N_T = h.shape[1], h.shape[2], h.shape[5] # 1000,2,100
N_SAMPLE = h.shape[0] * N_BLK * N_B * N_T
N_SUBC, N_RX, N_TX = h.shape[6], h.shape[3], h.shape[4]
print("N =", N_SAMPLE, "  per-sample [subc, RX, TX] =", (N_SUBC, N_RX, N_TX))

def get_sample(i):
    assert 0 <= i < N_SAMPLE
    blk, b, t = np.unravel_index(i, (N_BLK, N_B, N_T))
    s = np.asarray(h[0, blk, b, :, :, t, :])
    return s.transpose(2, 0, 1)

# ## 2. 训练 / 验证 / 测试划分与保存

# 不在 200000 个样本上随机打乱：同一 block 内 `t` 相邻、同一 seed 下 100 个 batch 连续生成，随机切分会泄漏。在 **block 轴**（1000 = 10 seed × 100 batch）上 80/10/10 划分，再展开成样本下标。

# 写出格式：复数 `[子载波, RX, TX]` → 实数 `[TX, 子载波, RX, 2]`（末轴为实/虚）。按 sorted block 分块写入，避免一次性读入 6.1 GiB。

# 输出：
# - `train_data.npy` `[160000, 32, 32, 4, 2]`
# - `val_data.npy` `[20000, 32, 32, 4, 2]`
# - `test_data.npy` `[20000, 32, 32, 4, 2]`
# - `train_idx.npy` / `val_idx.npy` / `test_idx.npy`

SPLIT_RATIOS = (0.8, 0.1, 0.1)
RNG_SEED = 0

rng = np.random.default_rng(RNG_SEED)
perm = rng.permutation(N_BLK)
n_train_blk = int(round(N_BLK * SPLIT_RATIOS[0]))
n_val_blk = int(round(N_BLK * SPLIT_RATIOS[1]))
train_blk = perm[:n_train_blk]
val_blk = perm[n_train_blk:n_train_blk + n_val_blk]
test_blk = perm[n_train_blk + n_val_blk:]

def blocks_to_sample_idx(blks):
    base = np.asarray(blks)[:, None] * (N_B * N_T)
    return np.sort((base + np.arange(N_B * N_T)[None, :]).ravel())

train_idx = blocks_to_sample_idx(train_blk)
val_idx = blocks_to_sample_idx(val_blk)
test_idx = blocks_to_sample_idx(test_blk)
print("train:", train_blk.size, "blocks /", train_idx.size, "samples")
print("val  :", val_blk.size, "blocks /", val_idx.size, "samples")
print("test :", test_blk.size, "blocks /", test_idx.size, "samples")

outdir = os.path.dirname(os.path.abspath(fpath))
train_path = os.path.join(outdir, "train_data.npy")
val_path = os.path.join(outdir, "val_data.npy")
test_path = os.path.join(outdir, "test_data.npy")

def write_split(path, blks):
    blks = np.sort(np.asarray(blks))
    n = blks.size * N_B * N_T
    shape = (n, N_TX, N_SUBC, N_RX, 2)
    fp = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    span = N_B * N_T
    cursor = 0
    for blk in blks:
        arr = np.asarray(h[0, int(blk)])
        arr = np.transpose(arr, (0, 3, 2, 4, 1))  # [N_T, N_SUBC, N_RX, N_TX] complex
        # ---- 逐样本归一化: 每个样本的 E[|h|^2] = 1 ----
        # arr 的轴 0 为时间步(样本), 对其余维度求 |h|^2 均值, 再缩放使逐样本功率归一化
        mean_power = np.mean(np.abs(arr) ** 2, axis=(1, 2, 3))  # (N_T,) 逐样本功率
        scale = 1.0 / np.sqrt(mean_power)                       # (N_T,) 缩放系数
        arr = arr * scale[:, None, None, None]
        # ------------------------------------------------
        out = np.stack((arr.real, arr.imag), axis=-1).astype(np.float32)
        fp[cursor:cursor + span] = out.reshape(span, N_TX, N_SUBC, N_RX, 2)
        cursor += span
    fp.flush()
    del fp
    print("wrote", path, "shape", shape)

write_split(train_path, train_blk)
write_split(val_path, val_blk)
write_split(test_path, test_blk)
np.save(os.path.join(outdir, "train_idx.npy"), train_idx)
np.save(os.path.join(outdir, "val_idx.npy"), val_idx)
np.save(os.path.join(outdir, "test_idx.npy"), test_idx)


