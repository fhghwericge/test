if (step + 1) % self.gradient_accum_steps == 0 or (step + 1) == len(self.train_loader):
    unclipped_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        self.train_cfg.max_grad_norm
    )

    grad_norm = unclipped_norm.item()
    max_norm = self.train_cfg.max_grad_norm

    clip_coef = min(
        1.0,
        max_norm / (grad_norm + 1e-12)
    )

    was_clipped = grad_norm > max_norm

    if step % 20 == 0:
        print(
            f"Step {step:4d} | "
            f"GradNorm={grad_norm:9.4f} | "
            f"ClipCoef={clip_coef:8.6f} | "
            f"Clipped={was_clipped}"
        )

    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad(set_to_none=True)
