## 代码段1

import torch
import torch.nn.functional as F


# ============================================================
# Gradient Conflict Diagnostic
# ============================================================

def gradient_conflict_stats(G, name="grad"):
    """
    G: (B, P)
       B 个样本对应的梯度向量
    """

    G = G.float()

    grad_norms = G.norm(dim=1)

    # 排除几乎为 0 的梯度：
    # 对接近局部驻点的训练样本，cosine 本身没有太大意义
    valid = grad_norms > 1e-10
    G = G[valid]
    grad_norms = grad_norms[valid]

    B_valid = G.shape[0]

    if B_valid < 2:
        print(f"[{name}] 有效梯度不足，无法计算冲突")
        return

    # --------------------------------------------------------
    # 1. Cancellation Ratio
    # --------------------------------------------------------

    mean_grad = G.mean(dim=0)

    norm_mean = mean_grad.norm()
    mean_norm = grad_norms.mean()

    R = norm_mean / (mean_norm + 1e-12)

    # --------------------------------------------------------
    # 2. Pairwise cosine similarity
    # --------------------------------------------------------

    G_norm = F.normalize(G, p=2, dim=1)

    cosine_matrix = G_norm @ G_norm.T

    # 只取上三角，不重复计算，也不要对角线
    mask = torch.triu(
        torch.ones(
            B_valid,
            B_valid,
            dtype=torch.bool
        ),
        diagonal=1
    )

    pair_cos = cosine_matrix[mask]

    mean_cos = pair_cos.mean()
    median_cos = pair_cos.median()

    negative_ratio = (
        pair_cos < 0
    ).float().mean()

    strong_negative_ratio = (
        pair_cos < -0.5
    ).float().mean()

    positive_ratio = (
        pair_cos > 0.5
    ).float().mean()

    print(f"\n[{name}]")
    print(f"  valid samples       : {B_valid}")
    print(
        f"  mean grad norm      : "
        f"{mean_norm.item():.6f}"
    )
    print(
        f"  ||mean grad||       : "
        f"{norm_mean.item():.6f}"
    )
    print(
        f"  cancellation R      : "
        f"{R.item():.4f}"
    )

    print(
        f"  mean cosine         : "
        f"{mean_cos.item():.4f}"
    )
    print(
        f"  median cosine       : "
        f"{median_cos.item():.4f}"
    )

    print(
        f"  cosine < 0 ratio    : "
        f"{negative_ratio.item():.2%}"
    )
    print(
        f"  cosine < -0.5 ratio : "
        f"{strong_negative_ratio.item():.2%}"
    )
    print(
        f"  cosine > 0.5 ratio  : "
        f"{positive_ratio.item():.2%}"
    )

    return {
        "R": R.item(),
        "mean_cos": mean_cos.item(),
        "median_cos": median_cos.item(),
        "negative_ratio": negative_ratio.item(),
        "strong_negative_ratio":
            strong_negative_ratio.item(),
        "positive_ratio": positive_ratio.item(),
        "mean_grad_norm": mean_norm.item(),
    }


def get_grad_vector(parameters):
    """将一组参数当前的 gradient 拼接成一维向量"""

    grads = []

    for p in parameters:
        if p.grad is not None:
            grads.append(
                p.grad.detach().reshape(-1).cpu()
            )

    if len(grads) == 0:
        return None

    return torch.cat(grads)


# ============================================================
# 实验设置
# ============================================================

B = 64

indices = list(range(B))

H_real = torch.stack(
    [test_ds[i][0] for i in indices]
).to(device)

snr_test = torch.stack(
    [test_ds[i][1] for i in indices]
).to(device)


# IMPORTANT:
# 使用 eval() 消除 Dropout 等随机性
# eval() 不会关闭 autograd
model.eval()


theta_grad_list = []
head_grad_list = []

# 可选：
# whole_model_grad_list = []


target_layer = model.head.mlp[2]


# ============================================================
# Per-sample backward
# ============================================================

for i in range(B):

    model.zero_grad(set_to_none=True)

    H_i = H_real[i:i+1]
    snr_i = snr_test[i:i+1]

    theta_i = model(
        H_i,
        snr=snr_i
    )

    # 保留非叶节点 Theta 的梯度
    theta_i.retain_grad()

    W_i = codebook(theta_i)

    loss_i = cap_loss(
        W_i,
        H_i,
        theta_i,
        snr_i
    )

    loss_i.backward()

    # ========================================================
    # 1. Theta-space gradient
    # ========================================================

    g_theta = (
        theta_i.grad
        .detach()
        .reshape(-1)
        .cpu()
    )

    # --------------------------------------------------------
    # 由于 phi 范围是 [0,4]，
    # 而 h/v 是 [0,1]，
    # 转换成实际相位角坐标后再比较 cosine 更合理：
    #
    # alpha_h/v = 2*pi*theta
    # alpha_phi = pi/2 * theta_phi
    #
    # dL/dalpha =
    # dL/dtheta * dtheta/dalpha
    # --------------------------------------------------------

    scale = torch.tensor(
        [
            1.0 / (2 * torch.pi),
            1.0 / (2 * torch.pi),
            2.0 / torch.pi,

            1.0 / (2 * torch.pi),
            1.0 / (2 * torch.pi),
            2.0 / torch.pi,
        ]
    )

    g_theta_physical = g_theta * scale

    theta_grad_list.append(
        g_theta_physical
    )

    # ========================================================
    # 2. Final-head gradient
    #
    # 同时包含 weight + bias
    # ========================================================

    g_head = torch.cat([
        target_layer.weight.grad
            .detach()
            .reshape(-1)
            .cpu(),

        target_layer.bias.grad
            .detach()
            .reshape(-1)
            .cpu()
    ])

    head_grad_list.append(g_head)

    # ========================================================
    # 3. Optional: whole-model gradient
    # ========================================================

    # g_model = get_grad_vector(
    #     model.parameters()
    # )
    #
    # whole_model_grad_list.append(
    #     g_model
    # )


# ============================================================
# Stack
# ============================================================

G_theta = torch.stack(
    theta_grad_list
)

G_head = torch.stack(
    head_grad_list
)


# ============================================================
# Statistics
# ============================================================

print("\n" + "=" * 70)
print("Gradient Conflict Diagnostic")
print("=" * 70)

stats_theta = gradient_conflict_stats(
    G_theta,
    name="Theta-space"
)

stats_head = gradient_conflict_stats(
    G_head,
    name="Final Head"
)


# Optional whole model
#
# G_model = torch.stack(
#     whole_model_grad_list
# )
#
# stats_model = gradient_conflict_stats(
#     G_model,
#     name="Whole Model"
# )


## 代码段2
import torch
import torch.nn.functional as F

# ============================================================
# 步骤 6: Batch-Gradient Alignment Diagnostic
#
# 目的：
#   判断真实 Batch 平均梯度是否能够同时优化不同样本，
#   并使用 Leave-One-Out 梯度排除样本自身梯度的正向贡献。
#
# 分析层级：
#   1. Theta-space       : 任务输出空间，仅作为参考
#   2. Preprocessor      : 输入特征提取
#   3. FiLM              : SNR 条件调制
#   4. Backbone          : Transformer 主干
#   5. Head (All)        : 完整输出头
#   6. Head Final Linear : 最后一层输出参数
# ============================================================


def flatten_parameter_grads(params):
    """
    将一个模块中所有参数的梯度展平成一个固定长度的一维向量。

    若某个参数 grad=None，则用同长度 0 向量代替，
    确保不同样本得到的梯度向量维度完全一致。
    """
    grads = []

    for p in params:
        if not p.requires_grad:
            continue

        if p.grad is None:
            g = torch.zeros(
                p.numel(),
                dtype=torch.float32,
                device="cpu"
            )
        else:
            g = (
                p.grad
                .detach()
                .reshape(-1)
                .float()
                .cpu()
            )

        grads.append(g)

    if len(grads) == 0:
        return None

    return torch.cat(grads, dim=0)


def alignment_statistics(
    G,
    name="Gradient",
    eps=1e-10
):
    """
    G: (B, P)
       B 个样本对应的梯度向量。

    输出两套指标：

    1. Full-Batch Alignment
       g_i 与真实 batch 平均梯度 g_bar 的关系。

    2. Leave-One-Out Alignment
       g_i 与除自身外其他样本平均梯度 g_bar_{-i} 的关系。
       该指标更直接反映样本间的梯度干扰。
    """

    G = G.float()

    B = G.shape[0]

    # --------------------------------------------------------
    # 单样本梯度范数
    # --------------------------------------------------------
    grad_norms = G.norm(dim=1)

    valid = grad_norms > eps

    if valid.sum() < 2:
        print(f"\n[{name}] 有效梯度样本不足。")
        return None

    # --------------------------------------------------------
    # 真实 Batch 平均梯度
    #
    # g_bar = (1/B) sum_i g_i
    # --------------------------------------------------------
    g_bar = G.mean(dim=0)

    g_bar_norm = g_bar.norm()

    mean_grad_norm = grad_norms[valid].mean()


    cancellation_R = (
        g_bar_norm /
        (mean_grad_norm + eps)
    )

    # --------------------------------------------------------
    # A. Full-Batch Alignment
    #
    # 包含样本自身梯度：
    #
    # g_i^T g_bar
    # =
    # ||g_i||^2 / B
    # +
    # sum_{j != i} g_i^T g_j / B
    # --------------------------------------------------------

    dot_full = G @ g_bar

    if g_bar_norm > eps:
        cos_full = (
            dot_full /
            (
                grad_norms *
                g_bar_norm +
                eps
            )
        )
    else:
        cos_full = torch.zeros_like(dot_full)

    full_valid = valid

    harmed_full = (
        dot_full[full_valid] < 0
    ).float().mean()

    benefited_full = (
        dot_full[full_valid] > 0
    ).float().mean()

    mean_cos_full = (
        cos_full[full_valid].mean()
    )

    median_cos_full = (
        cos_full[full_valid].median()
    )

    # --------------------------------------------------------
    # B. Leave-One-Out Alignment
    #
    # g_bar_-i =
    # (sum_j g_j - g_i) / (B - 1)
    #
    # 这里彻底去掉了样本自己的正向贡献。
    # --------------------------------------------------------

    sum_g = G.sum(dim=0)

    G_loo = (
        sum_g.unsqueeze(0) - G
    ) / (B - 1)

    loo_norms = G_loo.norm(dim=1)

    dot_loo = (
        G * G_loo
    ).sum(dim=1)

    cos_loo = (
        dot_loo /
        (
            grad_norms *
            loo_norms +
            eps
        )
    )

    loo_valid = (
        valid &
        (loo_norms > eps)
    )

    harmed_loo = (
        dot_loo[loo_valid] < 0
    ).float().mean()

    benefited_loo = (
        dot_loo[loo_valid] > 0
    ).float().mean()

    mean_cos_loo = (
        cos_loo[loo_valid].mean()
    )

    median_cos_loo = (
        cos_loo[loo_valid].median()
    )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print(f"[{name}]")
    print("-" * 65)

    print(
        f"  Valid Samples                  : "
        f"{valid.sum().item()}/{B}"
    )

    print(
        f"  Mean ||g_i||                   : "
        f"{mean_grad_norm.item():.6f}"
    )

    print(
        f"  ||g_bar||                      : "
        f"{g_bar_norm.item():.6f}"
    )

    print(
        f"  Cancellation Ratio R           : "
        f"{cancellation_R.item():.4f}"
    )

    print("\n  [A] Full-Batch Alignment")

    print(
        f"      Harmed Ratio               : "
        f"{harmed_full.item():.2%}"
    )

    print(
        f"      Benefited Ratio            : "
        f"{benefited_full.item():.2%}"
    )

    print(
        f"      Mean Cos(g_i, g_bar)       : "
        f"{mean_cos_full.item():.4f}"
    )

    print(
        f"      Median Cos(g_i, g_bar)     : "
        f"{median_cos_full.item():.4f}"
    )

    print("\n  [B] Leave-One-Out Alignment")

    print(
        f"      Cross-Harmed Ratio         : "
        f"{harmed_loo.item():.2%}"
    )

    print(
        f"      Cross-Benefited Ratio      : "
        f"{benefited_loo.item():.2%}"
    )

    print(
        f"      Mean Cos(g_i, g_bar_-i)    : "
        f"{mean_cos_loo.item():.4f}"
    )

    print(
        f"      Median Cos(g_i, g_bar_-i)  : "
        f"{median_cos_loo.item():.4f}"
    )

    return {
        "mean_grad_norm":
            mean_grad_norm.item(),

        "batch_grad_norm":
            g_bar_norm.item(),

        "R":
            cancellation_R.item(),

        "full_harmed_ratio":
            harmed_full.item(),

        "full_mean_cos":
            mean_cos_full.item(),

        "loo_harmed_ratio":
            harmed_loo.item(),

        "loo_mean_cos":
            mean_cos_loo.item(),
    }


def check_batch_alignment(
    model,
    H_batch,
    snr_batch,
    cap_loss_fn,
    codebook,
    name="Test"
):
    """
    对给定 Batch 进行逐样本反向传播，
    分析各网络模块的 Batch Gradient Alignment。
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # 使用 eval() 去除 Dropout 等随机因素。
    #
    # eval() 不会关闭 autograd，因此仍然可以 backward。
    # --------------------------------------------------------
    model.eval()

    B = H_batch.shape[0]

    # --------------------------------------------------------
    # 网络模块划分
    # --------------------------------------------------------

    final_linear = model.head.mlp[-1]

    modules = {
        "Preprocessor":
            list(model.preprocessor.parameters()),

        "FiLM":
            list(model.film_generator.parameters()),

        "Backbone":
            list(model.backbone.parameters()),

        "Head (All)":
            list(model.head.parameters()),

        "Head Final Linear":
            list(final_linear.parameters()),
    }

    # 每个模块存储 B 个样本梯度
    module_grads = {
        k: []
        for k in modules.keys()
    }

    # Theta-space 也保存，作为任务层面的参考
    theta_grads = []

    print("\n" + "=" * 70)
    print(
        f"Batch-Gradient Alignment Diagnostic "
        f"[{name} Set]"
    )
    print("=" * 70)

    # ========================================================
    # Step 1:
    # 逐样本独立 backward
    # ========================================================

    for i in range(B):

        model.zero_grad(set_to_none=True)

        H_i = H_batch[i:i+1]
        snr_i = snr_batch[i:i+1]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        theta_i = model(
            H_i,
            snr=snr_i
        )

        # Theta 不是 leaf tensor，
        # retain_grad() 后才能读取 theta_i.grad
        theta_i.retain_grad()

        W_i = codebook(theta_i)

        loss_i = cap_loss_fn(
            W_i,
            H_i,
            theta_i,
            snr_i
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss_i.backward()

        # ----------------------------------------------------
        # 1. Theta-space gradient
        #
        # 当前 Theta:
        #
        # [h1, v1, phi1, h2, v2, phi2]
        #
        # h/v 周期为 1
        # phi 周期为 4
        #
        # 为使不同参数尺度更具有物理可比性，
        # 转换为实际相位坐标下的梯度。
        # ----------------------------------------------------

        g_theta = (
            theta_i.grad
            .detach()
            .reshape(-1)
            .float()
            .cpu()
        )

        physical_scale = torch.tensor([
            1.0 / (2.0 * torch.pi),
            1.0 / (2.0 * torch.pi),
            2.0 / torch.pi,

            1.0 / (2.0 * torch.pi),
            1.0 / (2.0 * torch.pi),
            2.0 / torch.pi,
        ])

        g_theta_physical = (
            g_theta *
            physical_scale
        )

        theta_grads.append(
            g_theta_physical
        )

        # ----------------------------------------------------
        # 2. 各网络模块梯度
        # ----------------------------------------------------

        for mod_name, params in modules.items():

            g_vec = flatten_parameter_grads(
                params
            )

            module_grads[
                mod_name
            ].append(g_vec)

    # ========================================================
    # Step 2:
    # Stack
    # ========================================================

    G_theta = torch.stack(
        theta_grads,
        dim=0
    )

    module_G = {}

    for mod_name, grads_list in module_grads.items():

        module_G[mod_name] = torch.stack(
            grads_list,
            dim=0
        )

    # ========================================================
    # Step 3:
    # Alignment statistics
    # ========================================================

    results = {}

    print("\n" + "=" * 70)
    print("Theta-space Reference")
    print("=" * 70)

    results["Theta-space"] = (
        alignment_statistics(
            G_theta,
            name="Theta-space"
        )
    )

    print("\n" + "=" * 70)
    print("Network Parameter Space")
    print("=" * 70)

    for mod_name, G in module_G.items():

        results[mod_name] = (
            alignment_statistics(
                G,
                name=mod_name
            )
        )

    model.zero_grad(set_to_none=True)

    return results


# ============================================================
# 主实验：
# Trained Model + Test Set
# ============================================================

results_test = check_batch_alignment(
    model=model,
    H_batch=H_real,
    snr_batch=snr_test,
    cap_loss_fn=cap_loss,
    codebook=codebook,
    name="Test"
)

