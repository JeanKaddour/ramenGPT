import torch


def create_optimizer(model, optimizer_config: dict, print_fn=print):
    """
    Build optimizer state for dense and low-rank parameter groups.
    Low-rank updates are routed to a dedicated matrix optimizer when requested.
    """
    # Collect parameter groups - keep logic centralized to match existing behavior.
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n and p.ndim >= 2]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 and "x0_lambda" not in n]
    x0_lambda_params = [p for n, p in model.named_parameters() if "x0_lambda" in n]
    head_params = [model.lm_head.weight]
    low_rank_pairs = getattr(model, "low_rank_pairs", [])
    low_rank_pairs = [
        (pair[0], pair[1])
        for pair in low_rank_pairs
        if isinstance(pair, (tuple, list)) and len(pair) == 2
    ]
    low_rank_param_ids = {id(p) for pair in low_rank_pairs for p in pair}
    matrix_non_low_rank_params = [
        p for p in hidden_matrix_params if id(p) not in low_rank_param_ids
    ]

    adam_cfg = optimizer_config["adam"]
    scalar_adam_cfg = optimizer_config.get("scalar_adam", adam_cfg)
    use_muon = optimizer_config.get("use_muon", True)
    matrix_optimizer_type = optimizer_config.get("matrix_optimizer", "muon")
    is_spectron = matrix_optimizer_type == "spectron"

    optimizers = []
    scalar_opt = None
    matrix_opt = None
    scale_weight_decay_by_lr = optimizer_config.get("apply_lr_scale_to_weight_decay", False)
    if use_muon and len(hidden_matrix_params) > 0:
        # Use Muon/ARO/BAM for hidden matrix parameters if available.
        adam_param_groups = []
        if is_spectron and matrix_non_low_rank_params:
            adam_param_groups.append(
                {
                    "params": head_params + embed_params + matrix_non_low_rank_params,
                    "lr": adam_cfg["lr"],
                    "betas": adam_cfg["betas"],
                    "eps": adam_cfg["eps"],
                    "weight_decay": adam_cfg["weight_decay"],
                }
            )
        elif head_params + embed_params:
            adam_param_groups.append(
                {
                    "params": head_params + embed_params,
                    "lr": adam_cfg["lr"],
                    "betas": adam_cfg["betas"],
                    "eps": adam_cfg["eps"],
                    "weight_decay": adam_cfg["weight_decay"],
                }
            )
        adam_opt = Adam(adam_param_groups, scale_weight_decay_by_lr=scale_weight_decay_by_lr)

        all_scalar_params = scalar_params + x0_lambda_params
        if all_scalar_params:
            scalar_param_groups = [
                {
                    "params": all_scalar_params,
                    "lr": scalar_adam_cfg["lr"],
                    "betas": scalar_adam_cfg.get("betas", (0.9, 0.99)),
                    "eps": scalar_adam_cfg.get("eps", 1e-8),
                    "weight_decay": scalar_adam_cfg.get("weight_decay", 0.0),
                },
            ]
            scalar_opt = Adam(scalar_param_groups, scale_weight_decay_by_lr=scale_weight_decay_by_lr)
            print_fn(f"Created separate scalar optimizer for {len(all_scalar_params)} parameters")

        if matrix_optimizer_type == "spectron":
            if len(low_rank_pairs) == 0:
                print_fn(
                    "matrix_optimizer='spectron' requested, but no low-rank pairs were discovered; "
                    "falling back to Muon for hidden matrix parameters."
                )
                matrix_optimizer_type = "muon"
            else:
                spectron_cfg = optimizer_config.get(
                    "spectron",
                    {
                        "lr": 0.02,
                        "momentum": 0.95,
                        "momentum_min": 0.85,
                        "momentum_warmup_frac": 0.10,
                        "momentum_cooldown_frac": 0.10,
                        "beta2": 0.95,
                        "nesterov": True,
                        "power_iter_steps": 1,
                        "ns_iter_steps": 5,
                    },
                )
                matrix_opt = Spectron(
                    low_rank_pairs,
                    lr=spectron_cfg["lr"],
                    momentum=spectron_cfg["momentum"],
                    beta2=spectron_cfg.get("beta2", 0.95),
                    weight_decay=spectron_cfg.get("weight_decay", 0.0),
                    nesterov=spectron_cfg.get("nesterov", True),
                    power_iter_steps=spectron_cfg.get("power_iter_steps", 1),
                    ns_iter_steps=spectron_cfg.get("ns_iter_steps", 5),
                    scale_weight_decay_by_lr=scale_weight_decay_by_lr,
                )
                print_fn(
                    f"Using Spectron optimizer for {len(low_rank_pairs)} low-rank matrix pairs"
                )
        elif matrix_optimizer_type == "aro":
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
                scale_weight_decay_by_lr=scale_weight_decay_by_lr,
            )
            print_fn(
                f"Using ARO-Sinkhorn optimizer for {len(hidden_matrix_params)} matrix parameters"
            )
        elif matrix_optimizer_type == "lite":
            lite_cfg = optimizer_config.get(
                "lite",
                {
                    "lr": 0.02,
                    "momentum": 0.95,
                    "momentum_min": 0.85,
                    "momentum_warmup_frac": 0.10,
                    "momentum_cooldown_frac": 0.10,
                    "nesterov": True,
                    "ns_steps": 5,
                    "subspace_ratio": 0.1,
                    "lr_ratio": 2.0,
                    "beta_start": -0.25,
                    "beta_end": 1.0,
                    "beta_warmup_frac": 0.50,
                },
            )
            matrix_opt = LITE(
                hidden_matrix_params,
                lr=lite_cfg["lr"],
                momentum=lite_cfg["momentum"],
                weight_decay=lite_cfg.get("weight_decay", 0.0),
                nesterov=lite_cfg.get("nesterov", True),
                ns_steps=lite_cfg.get("ns_steps", 5),
                subspace_ratio=lite_cfg.get("subspace_ratio", 0.1),
                lr_ratio=lite_cfg.get("lr_ratio", 2.0),
                scale_weight_decay_by_lr=scale_weight_decay_by_lr,
            )
            print_fn(f"Using LITE optimizer for {len(hidden_matrix_params)} matrix parameters")
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
                scale_weight_decay_by_lr=scale_weight_decay_by_lr,
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
                scale_weight_decay_by_lr=scale_weight_decay_by_lr,
            )
            print_fn(f"Using NorMuon optimizer for {len(hidden_matrix_params)} matrix parameters")

        if matrix_opt is not None:
            optimizers = [adam_opt, matrix_opt]
        else:
            optimizers = [adam_opt]
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
        adam_opt = Adam(adam_param_groups, scale_weight_decay_by_lr=scale_weight_decay_by_lr)
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
            scalar_opt = Adam(scalar_param_groups, scale_weight_decay_by_lr=scale_weight_decay_by_lr)
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
def polar_express(G: torch.Tensor, num_iters: int | None = None):
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

    if num_iters is None:
        num_iters = len(polar_express_coeffs)
    for a, b, c in polar_express_coeffs[: num_iters]:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def mproj(M: torch.Tensor, sign_M: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Top-subspace projection: keeps eigenvalue > 1 subspace of M."""
    return (sign_M + polar_express(M - sign_M, num_iters=num_iters)) / 2


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


@torch.no_grad()
def _power_iteration_spectral_norm(matrix: torch.Tensor, vec_state: torch.Tensor | None, num_iters=1):
    if matrix.ndim != 2:
        raise ValueError("power iteration expects a 2D matrix")

    if matrix.numel() == 0:
        zero_vec = torch.zeros(matrix.size(0), device=matrix.device, dtype=torch.float32)
        return torch.tensor(0.0, device=matrix.device, dtype=torch.float32), zero_vec

    m_float = matrix.to(dtype=torch.float64)
    if not torch.isfinite(m_float).all():
        m_float = torch.nan_to_num(m_float, nan=0.0, posinf=0.0, neginf=0.0)

    if vec_state is None or vec_state.numel() != matrix.size(0):
        u = torch.randn(matrix.size(0), device=matrix.device, dtype=torch.float64)
    else:
        u = vec_state.to(dtype=torch.float64)

    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    if u.numel() == 0:
        return torch.tensor(0.0, device=matrix.device, dtype=torch.float32), torch.zeros_like(u)

    u_norm = u.norm()
    if not torch.isfinite(u_norm) or u_norm <= 0:
        u = torch.ones_like(u)
        u_norm = u.norm()
    u = u / (u_norm + 1e-12)

    for _ in range(max(1, int(num_iters))):
        v = torch.mv(m_float.T, u)
        if not torch.isfinite(v).all():
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v_norm = v.norm()
        if not torch.isfinite(v_norm) or v_norm <= 0:
            return torch.tensor(0.0, device=matrix.device, dtype=torch.float32), u
        v = v / (v_norm + 1e-12)

        u = torch.mv(m_float, v)
        if not torch.isfinite(u).all():
            u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u_norm = u.norm()
        if not torch.isfinite(u_norm) or u_norm <= 0:
            return torch.tensor(0.0, device=matrix.device, dtype=torch.float32), v
        u = u / (u_norm + 1e-12)

    sigma_vec = torch.mv(m_float, v)
    if not torch.isfinite(sigma_vec).all():
        sigma_vec = torch.nan_to_num(sigma_vec, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = sigma_vec.norm().to(dtype=torch.float64)
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma.to(dtype=torch.float32), u.to(dtype=torch.float32)


class Spectron(torch.optim.Optimizer):
    """
    Spectral renormalization + orthogonalization for low-rank matrix factors.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        beta2=0.95,
        weight_decay=0.0,
        nesterov=True,
        power_iter_steps=1,
        ns_iter_steps=5,
        scale_weight_decay_by_lr=False,
    ):
        if params is None:
            params = []
        pair_params = list(params)
        for pair in pair_params:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Spectron expects an iterable of parameter pairs.")
        param_groups = [{"params": [A, B]} for A, B in pair_params]
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta2=beta2,
            weight_decay=weight_decay,
            nesterov=nesterov,
            power_iter_steps=max(1, int(power_iter_steps)),
            ns_iter_steps=max(1, int(ns_iter_steps)),
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]
            power_iter_steps = group["power_iter_steps"]
            ns_iter_steps = group["ns_iter_steps"]

            pair = group["params"]
            if len(pair) != 2:
                raise RuntimeError("Spectron parameter groups must contain exactly two parameters.")
            A, B = pair
            if A.grad is None or B.grad is None:
                continue

            state_a = self.state[A]
            state_b = self.state[B]

            # Momentum buffers
            if "momentum_buffer" not in state_a:
                state_a["momentum_buffer"] = torch.zeros_like(A.grad)
            if "momentum_buffer" not in state_b:
                state_b["momentum_buffer"] = torch.zeros_like(B.grad)

            buf_a = state_a["momentum_buffer"]
            buf_b = state_b["momentum_buffer"]

            buf_a.lerp_(A.grad, 1 - group["momentum"])
            buf_b.lerp_(B.grad, 1 - group["momentum"])

            g_a = A.grad.lerp_(buf_a, group["momentum"]) if group["nesterov"] else buf_a
            g_b = B.grad.lerp_(buf_b, group["momentum"]) if group["nesterov"] else buf_b

            # Update orthogonality and estimate spectral norms.
            g_a = polar_express(g_a, num_iters=ns_iter_steps)
            g_b = polar_express(g_b, num_iters=ns_iter_steps)

            u_a = state_a.get("u_vec")
            u_b = state_b.get("u_vec")
            sigma_a, u_a = _power_iteration_spectral_norm(A.float(), u_a, power_iter_steps)
            sigma_b, u_b = _power_iteration_spectral_norm(B.float(), u_b, power_iter_steps)
            state_a["u_vec"] = u_a
            state_b["u_vec"] = u_b

            lr_mul = max(getattr(A, "lr_mul", 1.0), getattr(B, "lr_mul", 1.0))
            eff_lr = lr * lr_mul
            sigma_sum = sigma_a + sigma_b + 1.0
            if not torch.isfinite(sigma_sum):
                sigma_sum = torch.tensor(1.0, device=A.device, dtype=A.dtype)
            scale = eff_lr / sigma_sum.clamp_min(1e-8)

            wd_mul = max(getattr(A, "wd_mul", 1.0), getattr(B, "wd_mul", 1.0))
            eff_wd = wd * wd_mul
            if scale_weight_decay_by_lr:
                eff_wd *= lr

            delta_a = (g_a.to(dtype=A.dtype)) * scale
            delta_b = (g_b.to(dtype=B.dtype)) * scale

            if eff_wd != 0:
                mask_a = (delta_a * A) >= 0
                mask_b = (delta_b * B) >= 0
                A.addcmul_(A, mask_a, value=-eff_wd * scale)
                B.addcmul_(B, mask_b, value=-eff_wd * scale)

            A.add_(delta_a, alpha=-1.0)
            B.add_(delta_b, alpha=-1.0)


class Adam(torch.optim.Optimizer):
    """
    Adam optimizer matching train_gpt.py DistAdam behavior.
    Uses lr_mul but NOT shape_mult for learning rate (DistAdam doesn't use shape scaling).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        scale_weight_decay_by_lr=False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]

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
                if scale_weight_decay_by_lr:
                    eff_weight_decay *= lr
                if eff_weight_decay != 0:
                    mask = (update * p) > 0
                    update.addcmul_(p, mask, value=eff_weight_decay)

                p.add_(update, alpha=-1.0)


class NorMuon(torch.optim.Optimizer):
    """
    NorMuon - MomentUm Orthogonalized by Newton-schulz with variance reduction
    https://arxiv.org/abs/2510.05491
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        beta2=0.95,
        weight_decay=0.0,
        nesterov=True,
        scale_weight_decay_by_lr=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta2=beta2,
            weight_decay=weight_decay,
            nesterov=nesterov,
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
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
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]

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
                eff_wd = wd_mul * wd
                if scale_weight_decay_by_lr:
                    eff_wd *= lr
                mask = (g * p) >= 0
                if eff_wd != 0:
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)
                p.add_(g, alpha=-eff_lr)


class LITE(torch.optim.Optimizer):
    """
    LITE — Muon with flat-direction dynamics enhancement.
    arxiv.org/abs/2602.22681
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        ns_steps=5,
        subspace_ratio=0.1,
        lr_ratio=2.0,
        beta=0.0,
        scale_weight_decay_by_lr=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            subspace_ratio=subspace_ratio,
            lr_ratio=lr_ratio,
            beta=beta,
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
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
            ns_steps = group["ns_steps"]
            lr_ratio = group["lr_ratio"]
            beta = group["beta"]
            subspace_ratio = group["subspace_ratio"]
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                m, n = p.size(-2), p.size(-1)

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["subspace_threshold_ratio"] = 1.0 / (min(m, n) ** 0.5)
                buf = state["momentum_buffer"]

                # Clone gradient before Nesterov modifies it in-place
                g_orig = g.clone()

                # Momentum EMA
                buf.lerp_(g, 1 - group["momentum"])
                # Nesterov
                M = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # Sign of momentum (orthogonalized)
                m_ns = polar_express(M, num_iters=ns_steps)

                # Subspace projection
                k = int(subspace_ratio * min(m, n)) + 1
                k = min(k, min(m, n) - 1)
                thres_r = M.norm() * state["subspace_threshold_ratio"] + 1e-9
                P = mproj(M / thres_r, m_ns, ns_steps)

                # Adapt threshold
                if P.norm() > k**0.5:
                    state["subspace_threshold_ratio"] = min(
                        1.0, state["subspace_threshold_ratio"] * 1.05
                    )
                else:
                    state["subspace_threshold_ratio"] *= 0.95

                # Flat projector: I_n - P^T @ P
                P_flat = torch.eye(n, dtype=P.dtype, device=P.device) - P.mT @ P

                # Hessian damping from original gradient
                hd = polar_express(g_orig, num_iters=ns_steps)
                hd_flat = hd @ P_flat

                # Update: m_ns + beta * hd_flat, then amplify flat directions
                update = m_ns + beta * hd_flat
                update = update + (lr_ratio - 1) * (update @ P_flat)

                # Effective LR with shape multiplier
                shape_mult = max(1.0, m / n) ** 0.5
                eff_lr = lr * shape_mult * getattr(p, "lr_mul", 1.0)

                # Weight decay
                wd_mul = getattr(p, "wd_mul", 1.0)
                eff_wd = wd_mul * wd
                if scale_weight_decay_by_lr:
                    eff_wd *= lr

                # Standard masked weight decay
                update_p = update.to(dtype=p.dtype)
                if eff_wd != 0:
                    mask = (update_p * p) >= 0
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)

                # Extra flat-direction weight decay
                if eff_wd != 0 and lr_ratio != 1.0:
                    P_flat_p = P_flat.to(dtype=p.dtype)
                    p.add_(p @ P_flat_p, alpha=-eff_lr * (lr_ratio - 1) * eff_wd)

                # Apply update
                p.add_(update_p, alpha=-eff_lr)


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

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        nesterov=True,
        sink_steps=1,
        scale_weight_decay_by_lr=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sink_steps=sink_steps,
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
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
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]

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
                eff_wd = wd_mul * wd
                if scale_weight_decay_by_lr:
                    eff_wd *= lr
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
    https://arxiv.org/abs/2602.09006

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
        scale_weight_decay_by_lr=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sinkhorn_iters=sinkhorn_iters,
            scale_weight_decay_by_lr=scale_weight_decay_by_lr,
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
            scale_weight_decay_by_lr = group["scale_weight_decay_by_lr"]

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
                eff_wd = wd_mul * wd
                if scale_weight_decay_by_lr:
                    eff_wd *= lr
                mask = (delta * p) >= 0
                if eff_wd != 0:
                    p.addcmul_(p, mask, value=-eff_wd * eff_lr)
                p.add_(delta.to(dtype=p.dtype), alpha=-eff_lr)
