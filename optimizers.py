import torch


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor):
    """Polar Express sign method for orthogonalization."""
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Computed for num_iters=5, safety_factor=2e-2, cushion=2
    polar_express_coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]

    # Ensure spectral norm is at most 1.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction with low-rank second-moment estimator."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


class AdamSingleGPU(torch.optim.Optimizer):
    """
    Adam optimizer matching train_gpt.py DistAdam behavior.
    Uses lr_mul but NOT shape_mult for learning rate (DistAdam doesn't use shape scaling).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2**0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)

                wd_mul = getattr(p, "wd_mul", 1.0)
                eff_weight_decay = lr * wd * wd_mul
                if eff_weight_decay != 0:
                    mask = (update * p) > 0
                    update.addcmul_(p, mask, value=eff_weight_decay * lr)

                p.add_(update, alpha=-1.0)


class NorMuonSingleGPU(torch.optim.Optimizer):
    """
    NorMuon - MomentUm Orthogonalized by Newton-schulz with variance reduction
    """

    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95, weight_decay=0.0, nesterov=True):
        defaults = dict(
            lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay, nesterov=nesterov
        )
        super().__init__(params, defaults)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()
                if "second_momentum_buffer" in state:
                    state["second_momentum_buffer"].zero_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta2 = group["beta2"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                g = polar_express(g)

                is_gate = "gate" in getattr(p, "label", "")
                param_shape = p.shape
                if is_gate or param_shape[-2] >= param_shape[-1]:
                    red_dim = -1
                    buffer_shape = (*g.shape[:-1], 1)
                else:
                    red_dim = -2
                    buffer_shape = (*g.shape[:-2], 1, g.shape[-1])

                if "second_momentum_buffer" not in state:
                    state["second_momentum_buffer"] = torch.zeros(
                        buffer_shape, dtype=g.dtype, device=g.device
                    )
                second_momentum_buffer = state["second_momentum_buffer"]

                g = apply_normuon_variance_reduction(g, second_momentum_buffer, beta2, red_dim)

                shape_mult = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                eff_lr = lr * shape_mult * getattr(p, "lr_mul", 1.0)

                wd_mul = getattr(p, "wd_mul", 1.0)
                eff_wd = wd_mul * wd * lr
                mask = (g * p) >= 0
                if eff_wd != 0:
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)
                p.add_(g, alpha=-eff_lr)
