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
