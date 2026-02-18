import torch


def create_optimizer(model, optimizer_config: dict, print_fn=print):
    # Collect parameter groups - keep logic centralized to match existing behavior.
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n and p.ndim >= 2]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 and "x0_lambda" not in n]
    x0_lambda_params = [p for n, p in model.named_parameters() if "x0_lambda" in n]
    head_params = [model.lm_head.weight]

    adam_cfg = optimizer_config["adam"]
    scalar_adam_cfg = optimizer_config.get("scalar_adam", adam_cfg)
    use_muon = optimizer_config.get("use_muon", True)

    optimizers = []
    scalar_opt = None
    matrix_opt = None
    matrix_optimizer_type = None

    if use_muon and len(hidden_matrix_params) > 0:
        # Use Muon/ARO/BAM for hidden matrix parameters if available.
        adam_param_groups = []
        if head_params + embed_params:
            adam_param_groups.append(
                {
                    "params": head_params + embed_params,
                    "lr": adam_cfg["lr"],
                    "betas": adam_cfg["betas"],
                    "eps": adam_cfg["eps"],
                    "weight_decay": adam_cfg["weight_decay"],
                }
            )
        adam_opt = Adam(adam_param_groups)

        all_scalar_params = scalar_params + x0_lambda_params
        if all_scalar_params:
            scalar_param_groups = [
                {
                    "params": all_scalar_params,
                    "lr": scalar_adam_cfg["lr"],
                    "betas": scalar_adam_cfg.get("betas", (0.9, 0.99)),
                    "eps": scalar_adam_cfg.get("eps", 1e-8),
                    "weight_decay": scalar_adam_cfg.get("weight_decay", 0.0),
                }
            ]
            scalar_opt = Adam(scalar_param_groups)
            print_fn(f"Created separate scalar optimizer for {len(all_scalar_params)} parameters")

        matrix_optimizer_type = optimizer_config.get("matrix_optimizer", "muon")
        if matrix_optimizer_type == "aro":
            aro_cfg = optimizer_config.get(
                "aro",
                {
                    "lr": 0.02,
                    "momentum": 0.95,
                    "momentum_min": 0.85,
                    "momentum_warmup_frac": 0.10,
                    "momentum_cooldown_frac": 0.10,
                    "nesterov": True,
                    "sinkhorn_iters": 5,
                },
            )
            matrix_opt = AROSinkhorn(
                hidden_matrix_params,
                lr=aro_cfg["lr"],
                momentum=aro_cfg["momentum"],
                weight_decay=aro_cfg.get("weight_decay", 0.0),
                nesterov=aro_cfg.get("nesterov", True),
                sinkhorn_iters=aro_cfg.get("sinkhorn_iters", 5),
            )
            print_fn(
                f"Using ARO-Sinkhorn optimizer for {len(hidden_matrix_params)} matrix parameters"
            )
        elif matrix_optimizer_type == "bam":
            bam_cfg = optimizer_config.get(
                "bam",
                {
                    "lr": 0.02,
                    "momentum": 0.95,
                    "momentum_min": 0.85,
                    "momentum_warmup_frac": 0.10,
                    "momentum_cooldown_frac": 0.10,
                    "nesterov": True,
                    "sink_steps": 1,
                },
            )
            matrix_opt = BAM(
                hidden_matrix_params,
                lr=bam_cfg["lr"],
                momentum=bam_cfg["momentum"],
                weight_decay=bam_cfg.get("weight_decay", 0.0),
                nesterov=bam_cfg.get("nesterov", True),
                sink_steps=bam_cfg.get("sink_steps", 1),
            )
            print_fn(f"Using BAM optimizer for {len(hidden_matrix_params)} matrix parameters")
        else:
            muon_cfg = optimizer_config.get(
                "muon",
                {
                    "lr": 0.02,
                    "momentum": 0.95,
                    "momentum_min": 0.85,
                    "momentum_warmup_frac": 0.144,
                    "momentum_cooldown_frac": 0.024,
                    "beta2": 0.95,
                    "nesterov": True,
                },
            )
            matrix_opt = NorMuon(
                hidden_matrix_params,
                lr=muon_cfg["lr"],
                momentum=muon_cfg["momentum"],
                beta2=muon_cfg.get("beta2", 0.95),
                weight_decay=muon_cfg.get("weight_decay", 0.0),
                nesterov=muon_cfg.get("nesterov", True),
            )
            print_fn(f"Using NorMuon optimizer for {len(hidden_matrix_params)} matrix parameters")

        optimizers = [adam_opt, matrix_opt]
        if scalar_opt:
            optimizers.append(scalar_opt)

    else:
        # Use a single Adam optimizer for all non-scalar parameters.
        adam_param_groups = []
        non_scalar_params = hidden_matrix_params + head_params + embed_params
        if non_scalar_params:
            adam_param_groups.append(
                {
                    "params": non_scalar_params,
                    "lr": adam_cfg["lr"],
                    "betas": adam_cfg["betas"],
                    "eps": adam_cfg["eps"],
                    "weight_decay": adam_cfg["weight_decay"],
                }
            )
        adam_opt = Adam(adam_param_groups)
        optimizers = [adam_opt]

        all_scalar_params = scalar_params + x0_lambda_params
        if all_scalar_params:
            scalar_param_groups = [
                {
                    "params": all_scalar_params,
                    "lr": scalar_adam_cfg["lr"],
                    "betas": scalar_adam_cfg.get("betas", (0.9, 0.99)),
                    "eps": scalar_adam_cfg.get("eps", 1e-8),
                    "weight_decay": 0.0,
                }
            ]
            scalar_opt = Adam(scalar_param_groups)
            optimizers.append(scalar_opt)

        print_fn("Using Adam optimizer for all parameters")

    return dict(
        adam_opt=adam_opt,
        matrix_opt=matrix_opt,
        scalar_opt=scalar_opt,
        optimizers=optimizers,
        matrix_optimizer_type=matrix_optimizer_type,
        hidden_matrix_params=hidden_matrix_params,
        embed_params=embed_params,
        scalar_params=scalar_params,
        x0_lambda_params=x0_lambda_params,
        head_params=head_params,
    )


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


class Adam(torch.optim.Optimizer):
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


class NorMuon(torch.optim.Optimizer):
    """
    NorMuon - MomentUm Orthogonalized by Newton-schulz with variance reduction
    https://arxiv.org/abs/2510.05491
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


@torch.compile(dynamic=False, fullgraph=True)
def sink_norm(G: torch.Tensor, steps: int = 1):
    """BAM: alternating row/column L2 normalization.
    Order depends on matrix shape (tall vs wide) for better conditioning.
    """
    for _ in range(steps):
        if G.shape[-2] > G.shape[-1]:
            G = G / (torch.linalg.vector_norm(G, ord=2, dim=-1, keepdim=True) + 1e-7)
            G = G / (torch.linalg.vector_norm(G, ord=2, dim=-2, keepdim=True) + 1e-7)
        else:
            G = G / (torch.linalg.vector_norm(G, ord=2, dim=-2, keepdim=True) + 1e-7)
            G = G / (torch.linalg.vector_norm(G, ord=2, dim=-1, keepdim=True) + 1e-7)
    return G


class BAM(torch.optim.Optimizer):
    """
    BAM - Balanced Axis Momentum
    https://github.com/knightron0/bam

    A simplified variant of Muon that replaces Newton-Schulz orthogonalization
    with cheap alternating row/column L2 normalization (SinkNorm).
    O(mn) per step vs O(mn*min(m,n)) for Newton-Schulz.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True, sink_steps=1):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, sink_steps=sink_steps
        )
        super().__init__(params, defaults)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

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

                g = sink_norm(g, steps=group["sink_steps"])

                shape_mult = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                eff_lr = lr * shape_mult * getattr(p, "lr_mul", 1.0)

                wd_mul = getattr(p, "wd_mul", 1.0)
                eff_wd = wd_mul * wd * lr
                mask = (g * p) >= 0
                if eff_wd != 0:
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)
                p.add_(g, alpha=-eff_lr)


@torch.compile(dynamic=False, fullgraph=True)
def sinkhorn_normalize(X: torch.Tensor, num_iters: int = 5):
    """SinkGD: simultaneous row and column L2 normalization, iterated."""
    X = X.float()
    for _ in range(num_iters):
        row_norms = X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        col_norms = X.norm(dim=-2, keepdim=True).clamp_min(1e-8)
        X = X / row_norms / col_norms
    return X


def shifted_cholesky_qr(A, eps=1e-7):
    """Shifted Cholesky QR factorization. Falls back to torch.linalg.qr on failure."""
    A = A.float()
    m = A.size(-2)
    P = A.mT @ A + eps * torch.eye(m, device=A.device, dtype=A.dtype)
    try:
        L = torch.linalg.cholesky(P)
        Q = torch.linalg.solve_triangular(L.mT, A.mT, upper=True).mT
        if not Q.isfinite().all():
            raise RuntimeError("Non-finite in SCQR")
        return Q
    except (RuntimeError, torch.linalg.LinAlgError):
        Q, _ = torch.linalg.qr(A)
        return Q


class AROSinkhorn(torch.optim.Optimizer):
    """
    ARO-Sinkhorn optimizer -- Adaptively Rotated Optimization with Sinkhorn normalization.
    https://arxiv.org/abs/2502.xxxxx

    Uses a learned rotation matrix per parameter to align gradients before
    applying Sinkhorn normalization, achieving better spectral conditioning
    than fixed orthogonalization methods.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        sinkhorn_iters=5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sinkhorn_iters=sinkhorn_iters,
        )
        super().__init__(params, defaults)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()
                if "rotation" in state:
                    state["rotation"].copy_(
                        torch.eye(p.size(-2), device=p.device, dtype=torch.float32)
                    )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            sinkhorn_iters = group["sinkhorn_iters"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize state on first step
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                if "rotation" not in state:
                    state["rotation"] = torch.eye(
                        p.size(-2), device=p.device, dtype=torch.float32
                    )

                buf = state["momentum_buffer"]
                R = state["rotation"]

                # Momentum EMA
                buf.lerp_(g, 1 - group["momentum"])
                M = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                M_float = M.float()

                # --- Rotation selection (first Sinkhorn call) ---
                D_tilde = sinkhorn_normalize(R.mT @ M_float, num_iters=sinkhorn_iters)
                A = M_float @ D_tilde.mT  # m x m cross-Gram
                R_new = shifted_cholesky_qr(A)
                state["rotation"] = R_new

                # --- Rotated update (second Sinkhorn call) ---
                update = sinkhorn_normalize(R_new.mT @ M_float, num_iters=sinkhorn_iters)
                delta = R_new @ update  # rotate back to original coords

                # Apply shape multiplier, lr_mul, masked weight decay (same as NorMuon)
                shape_mult = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                eff_lr = lr * shape_mult * getattr(p, "lr_mul", 1.0)

                wd_mul = getattr(p, "wd_mul", 1.0)
                eff_wd = wd_mul * wd * lr
                mask = (delta * p) >= 0
                if eff_wd != 0:
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)
                p.add_(delta.to(dtype=p.dtype), alpha=-eff_lr)
