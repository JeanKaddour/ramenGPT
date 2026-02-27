import os
model_config = dict(
    vocab_size=50257,
    num_layers=6,
    num_heads=4,
    model_dim=256,
    head_dim=64,
    activation="relu_squared",
    mlp_init_std_scale=0.5,
    lm_head_init_std=0.005,
    embed_padding_multiple=128,
    eos_token_id=50256,
    value_embed_head_indices=[1, 2],
    value_embed_mid_layer_count=5,
    value_embed_tail_indices=[0, 1, 2],
    value_embed_gate_scale=2.0,
    skip_gate_scale=2.0,
    residual_first_layer_index=0,
    logits_softcap_scale=23.0,
    logits_softcap_shift=5.0,
    logits_softcap_divisor=7.5,
)

gating_config = dict(
    use_attn_gate=True,
    use_value_embed_gate=True,
    use_smear_gate=True,
    use_skip_gate=True,
    gate_input_dim=12,
)

rope_config = dict(
    type="yarn",           # "yarn", "half_rope", "rope", or "none"
    base_freq=1024,
    initial_attn_scale=0.1,
)

embed_config = dict(
    weight_tied=True,
    enable_embed_split=True,  # create untied embedding late in training if enabled
    split_frac=0.90,  # fraction of the full run reached when untying embeddings
)

skip_config = dict(
    skip_in_layers=[2],
    skip_out_layers=[4],
    backout_layer=5,
)

residual_connection_config = dict(
    mode="standard",
    num_streams=1,
    num_fracs=1,
    tanh=True,
    disable=None,
    sinkhorn_iters=10,
    sinkhorn_tau=0.05,
    mhc_h_res_proj="sinkhorn",
    ns_steps=5,
    ns_eps=1e-7,
    ns_coeffs=(3.0, -3.2, 1.2),
    mhc_residual_identity_mix=False,
    mhc_residual_alpha=0.01,
)

# Data layout matches the FineWeb shards used by baseline configs.
TRAIN_SEQ_LEN = 512
BATCH_SIZE_MULTIPLE = int(os.environ.get("BATCH_SIZE_MULTIPLE", "64"))
if BATCH_SIZE_MULTIPLE < 1:
    raise ValueError("BATCH_SIZE_MULTIPLE must be >= 1")

data_config = dict(
    train_files="data/fineweb10B/fineweb_train_*.bin",
    val_files="data/fineweb10B/fineweb_val_*.bin",
    val_tokens=TRAIN_SEQ_LEN * 8,
    train_seq_len=TRAIN_SEQ_LEN,
    val_seq_len=TRAIN_SEQ_LEN,
    batch_size_multiple=BATCH_SIZE_MULTIPLE,
    # Single micro-batch setup: do the whole mini-batch in one pass.
    train_micro_batch_tokens=TRAIN_SEQ_LEN * BATCH_SIZE_MULTIPLE,
)

batch_schedule_config = dict(
    schedule_type="stepped",
    # Keep these aligned with the stepped schedule for logging/compatibility.
    initial_grad_accum_steps=1,
    final_grad_accum_steps=8,
    warmup_frac=0.5,
    # Triple doubling across training: 1x -> 2x -> 4x -> 8x.
    batch_sizes=[1, 2, 4, 8],
    base_tokens=TRAIN_SEQ_LEN * BATCH_SIZE_MULTIPLE,
    transitions=[0.25, 0.50, 0.75],
)

window_schedule_config = dict(
    schedule=[3],
    final_ws=3,
    post_yarn_extension=6,
    transitions=[1.0],
)

training_config = dict(
    num_iterations=420,
    num_scheduled_iterations=320,
    cooldown_frac=0.40,
    val_loss_every=50,
    save_checkpoint=False,
    checkpoint_every=0,
    checkpoint_root="checkpoints",
    grad_clip_norm=1.0,
)

optimizer_config = dict(
    use_muon=True,
    matrix_optimizer="muon",  # "muon", "spectron", "aro", "bam", or "lite"
    apply_lr_scale_to_weight_decay=False,
    adam=dict(
        lr=0.006,
        betas=(0.65, 0.95),
        eps=1e-8,
        weight_decay=0.005,
    ),
    scalar_adam=dict(
        lr=0.006,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
    ),
    muon=dict(
        lr=0.02,
        weight_decay=1.2,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        beta2=0.95,
        nesterov=True,
    ),
    aro=dict(
        lr=0.02,
        weight_decay=1.2,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        nesterov=True,
        sinkhorn_iters=5,
    ),
    bam=dict(
        lr=0.02,
        weight_decay=1.2,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        nesterov=True,
        sink_steps=1,
    ),
    spectron=dict(
        lr=0.02,
        weight_decay=1.2,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        beta2=0.95,
        nesterov=True,
        power_iter_steps=1,
        ns_iter_steps=5,
    ),
    lite=dict(
        lr=0.02,
        weight_decay=1.2,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        nesterov=True,
        ns_steps=5,
        subspace_ratio=0.1,
        lr_ratio=2.0,
        beta_start=-0.25,
        beta_end=1.0,
        beta_warmup_frac=0.50,
    ),
    lr_multipliers=dict(
        embed=1.0,
        value_embed=75.0,
        c_proj=2.0,
        head=1.0,
        scalars=5.0,
        x0_lambdas=5.0,
        smear_gate=0.01,
        skip_gate=0.05,
    ),
    wd_multipliers=dict(
        value_embed=5.0,
        embed=150.0,
        head=150.0,
        scalars=0.0,
        x0_lambdas=0.0,
        smear_gate=0.0,
        skip_gate=0.0,
    ),
    freeze_scalars_on_transition=8,
)

low_rank_config = dict(
    enabled=False,
    rank_ratio=0.25,
    rank=None,
    min_rank=1,
    max_rank=None,
    apply_attention=True,
    apply_mlp=True,
)

# FlexAttention setup
attention_config = dict(
    block_size=128,
    attention_scale=None,
    max_window_size=2048,
)

attention_pattern_config = dict(
    block_mask_pattern="SSSSSS",
    value_embed_layers=[0, 1, 2, 3, 4, 5],
    num_value_embeds=3,
    skip_attention_layers=[],
)

lambda_config = dict(
    resid_lambdas_init=1.1,
    x0_lambdas_init=0.0,
    sa_lambdas_init=[0.5, 1.0],
    sa_lambdas_init_no_ve=[0.5, 1.0],
    smear_lambda_init=0.0,
    backout_lambda_init=0.5,
    skip_lambda_init=-1.5,
)

# Keep warmup to stabilize early steps
warmup_config = dict(
    warmup_steps=8,
    warmup_seq_len=256,
    lr_warmup_steps=16,
)

logging_config = dict(
    use_wandb=True,
    wandb_project="ramenGPT",
    wandb_run_name=None,
)

compilation_config = dict(
    compile_model=True,
    relaxed_compile=True,
)
