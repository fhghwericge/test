import torch
import torch.nn.functional as F

# ============================================================
# Gradient Structure Diagnostic
# 目的：
# 1. Theta-space：验证不同样本的输出参数梯度存在明显方向分化/冲突
# 2. Final Head：验证上述冲突传入共享参数空间后转化为近似正交
# 3. 各网络模块：验证不同样本之间缺乏稳定的跨样本梯度共享
# ============================================================

def flatten_grads(params):
    grads = []
    for p in params:
        if not p.requires_grad:
            continue
        g = torch.zeros(p.numel()) if p.grad is None else p.grad.detach().reshape(-1).float().cpu()
        grads.append(g)
    return torch.cat(grads)


def pairwise_stats(G, name):
    """统计样本两两梯度方向，用于 Theta-space 和 Final Head。"""
    G = G.float()
    norms = G.norm(dim=1)
    valid = norms > 1e-10
    G = G[valid]
    G_norm = F.normalize(G, p=2, dim=1)

    cos_matrix = G_norm @ G_norm.T
    mask = torch.triu(torch.ones(len(G), len(G), dtype=torch.bool), diagonal=1)
    pair_cos = cos_matrix[mask]

    mean_g = G.mean(dim=0)
    R = mean_g.norm() / (G.norm(dim=1).mean() + 1e-12)

    print(f"\n[{name}]")
    print(f"  R                         : {R.item():.4f}")
    print(f"  Mean Pairwise Cosine      : {pair_cos.mean().item():.4f}")
    print(f"  Strong Conflict (< -0.5)  : {(pair_cos < -0.5).float().mean().item():.2%}")
    print(f"  Strong Alignment (> 0.5)  : {(pair_cos > 0.5).float().mean().item():.2%}")


def cross_sample_stats(G, name):
    """Leave-One-Out：统计其他样本的平均梯度对当前样本的影响。"""
    G = G.float()
    norms = G.norm(dim=1)
    valid = norms > 1e-10
    B = G.shape[0]

    mean_g = G.mean(dim=0)
    R = mean_g.norm() / (norms[valid].mean() + 1e-12)

    G_loo = (G.sum(dim=0, keepdim=True) - G) / (B - 1)
    loo_norms = G_loo.norm(dim=1)
    valid = valid & (loo_norms > 1e-10)

    dot = (G * G_loo).sum(dim=1)
    cosine = dot / (norms * loo_norms + 1e-12)

    print(f"\n[{name}]")
    print(f"  R                         : {R.item():.4f}")
    print(f"  LOO Mean Cosine           : {cosine[valid].mean().item():.4f}")
    print(f"  Cross-Harmed Ratio        : {(dot[valid] < 0).float().mean().item():.2%}")


def gradient_structure_diagnostic(model, H_batch, snr_batch, cap_loss, codebook):
    model.eval()
    B = H_batch.shape[0]
    final_layer = model.head.mlp[-1]

    modules = {
        "Preprocessor": list(model.preprocessor.parameters()),
        "FiLM": list(model.film_generator.parameters()),
        "Backbone": list(model.backbone.parameters()),
        "Head": list(model.head.parameters()),
        "Final Head": list(final_layer.parameters()),
    }

    theta_grads = []
    module_grads = {name: [] for name in modules}

    for i in range(B):
        model.zero_grad(set_to_none=True)

        H_i = H_batch[i:i+1]
        snr_i = snr_batch[i:i+1]

        theta_i = model(H_i, snr=snr_i)
        theta_i.retain_grad()

        W_i = codebook(theta_i)
        loss_i = cap_loss(W_i, H_i, theta_i, snr_i)
        loss_i.backward()

        # 转换到统一的物理相位坐标：
        # alpha_h/v = 2*pi*theta，alpha_phi = pi/2*theta_phi
        scale = torch.tensor([1/(2*torch.pi), 1/(2*torch.pi), 2/torch.pi, 1/(2*torch.pi), 1/(2*torch.pi), 2/torch.pi])
        theta_grads.append(theta_i.grad.detach().reshape(-1).float().cpu() * scale)

        for name, params in modules.items():
            module_grads[name].append(flatten_grads(params))

    G_theta = torch.stack(theta_grads)
    G_modules = {name: torch.stack(grads) for name, grads in module_grads.items()}

    print("=" * 64)
    print(f"Gradient Structure Diagnostic | B={B}")
    print(f"Orthogonal reference: 1/sqrt(B) = {1 / (B ** 0.5):.4f}")
    print("=" * 64)

    # --------------------------------------------------------
    # Part 1：输出参数梯度存在明显方向分化/冲突
    # --------------------------------------------------------
    print("\n=== Pairwise Gradient Structure ===")
    pairwise_stats(G_theta, "Theta-space")
    pairwise_stats(G_modules["Final Head"], "Final Head")

    # --------------------------------------------------------
    # Part 2：网络各层跨样本梯度是否存在共享方向
    # --------------------------------------------------------
    print("\n=== Cross-Sample Gradient Sharing (Leave-One-Out) ===")
    for name in ["Preprocessor", "FiLM", "Backbone", "Head", "Final Head"]:
        cross_sample_stats(G_modules[name], name)

    model.zero_grad(set_to_none=True)


# ============================================================
# Trained Model + Test Set
# ============================================================

B = 64
H_test = torch.stack([test_ds[i][0] for i in range(B)]).to(device)
snr_test = torch.stack([test_ds[i][1] for i in range(B)]).to(device)

gradient_structure_diagnostic(
    model=model,
    H_batch=H_test,
    snr_batch=snr_test,
    cap_loss=cap_loss,
    codebook=codebook
)
