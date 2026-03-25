import os

NUM_LAYERS = 8
VALUE_EMBED_HEAD_INDICES = [1, 2]
VALUE_EMBED_MID_LAYER_COUNT = 5
VALUE_EMBED_TAIL_INDICES = [0, 1, 2]

model_config = dict(
    vocab_size=50257,
    num_layers=NUM_LAYERS,
    num_heads=4,
    num_kv_heads=4,
    model_dim=256,
    head_dim=64,
    activation="relu_squared",
    mlp_init_std_scale=0.5,
    lm_head_init_std=0.005,
    embed_padding_multiple=128,
    eos_token_id=50256,
    value_embed_head_indices=VALUE_EMBED_HEAD_INDICES,
    value_embed_mid_layer_count=VALUE_EMBED_MID_LAYER_COUNT,
    value_embed_tail_indices=VALUE_EMBED_TAIL_INDICES,
    value_embed_gate_scale=2.0,
    skip_gate_scale=2.0,
    residual_first_layer_index=0,
    logits_softcap_scale=23.0,
    logits_softcap_shift=5.0,
    logits_softcap_divisor=7.5,
    logits_softcap_mode="sigmoid",
    logits_tanh_cap=30.0,
    qk_gain_init=1.0,
    ln_scale=False,
)

gating_config = dict(
    use_attn_gate=True,
    use_value_embed_gate=True,
    use_smear_gate=True,
    use_skip_gate=True,
    gate_input_dim=12,
)

rope_config = dict(
    type="yarn",           # "yarn", "half_rope", "rope", "none", or "nope"
    base_freq=1024,
    initial_attn_scale=0.1,
    rope_dims=0,
)

embed_config = dict(
    weight_tied=True,
    enable_embed_split=True,  # create untied embedding late in training if enabled
    split_frac=0.90,  # fraction of the full run reached when untying embeddings
)

skip_config = dict(
    skip_in_layers=[2],
    skip_out_layers=[NUM_LAYERS - 2],
    backout_layer=NUM_LAYERS - 1,
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

canon_config = dict(
    enabled=False,
    set="ABCD",
    kernel=4,
    first_n=0,
    last_n=0,
    layers=(),
    bias=False,
    activation=False,
    residual=True,
    delta_gate=False,
    delta_gate_init=-4.0,
    use_fast_conv1d=True,
)

smear_config = dict(
    mode="ramen",
)

skip_topology_config = dict(
    mode="ramen",
)

xsa_config = dict(
    enabled=False,
    last_n=0,
    learnable_gate=False,
    gate_init=2.0,
)

boundary_delta_config = dict(
    enabled=False,
    first_n=0,
    gate_vector=False,
    gate_init=-4.0,
)

resid_mix_config = dict(
    enabled=False,
)

bigram_config = dict(
    enabled=False,
    vocab_size=0,
    dim=0,
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
    initial_grad_accum_steps=8,
    final_grad_accum_steps=8,
    warmup_frac=0.5,
    # Constant max batch from step 0.
    batch_sizes=[8],
    base_tokens=TRAIN_SEQ_LEN * BATCH_SIZE_MULTIPLE,
    transitions=[],
)

window_schedule_config = dict(
    schedule=[3],
    final_ws=3,
    post_yarn_extension=12,
    transitions=[1.0],
)

training_config = dict(
    num_iterations=500,
    num_scheduled_iterations=380,
    cooldown_frac=0.50,
    final_lr_ratio=0.1,
    val_loss_every=50,
    save_checkpoint=False,
    checkpoint_every=0,
    checkpoint_root="checkpoints",
    grad_clip_norm=1.0,
)

lr_scheduler_config = dict(
    scheduler_type="linear",
    warmup_steps=16,
    use_linear_warmup=True,
    cooldown_steps=int(training_config["num_iterations"] * training_config["cooldown_frac"]),
    final_lr_ratio=training_config["final_lr_ratio"],
    cosine_min_lr_ratio=0.1,
)

optimizer_config = dict(
    use_muon=True,
    matrix_optimizer="muon",  # "muon", "spectron", "aro", "bam", or "lite"
    apply_lr_scale_to_weight_decay=False,
    adam=dict(
        lr=0.01,
        betas=(0.80, 0.95),
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
        lr=0.015,
        weight_decay=0.8,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        beta2=0.95,
        nesterov=True,
        muon_split=False,       # per-head/group orthogonalization for attn+MLP weights
        muon_decorrelate=False, # Gram-Schmidt decorrelation across heads/groups
        mlp_split_groups=0,     # number of neuron groups for MLP split (0=disabled)
    ),
    aro=dict(
        lr=0.0003,
        weight_decay=0.1,
        momentum=0.95,
        momentum_min=0.85,
        momentum_warmup_frac=0.10,
        momentum_cooldown_frac=0.10,
        beta2=0.95,
        nesterov=True,
        sinkhorn_iters=5,
        rms_norm_target=0.2,
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

attn_variant_config = dict(
    variant="baseline",  # "baseline" | "relational_transport" | "hyper_attention"
    # Relational Transport params
    rt_num_relations=2,
    rt_copy_frac=0.5,
    rt_gate_bias_init=2.0,
    rt_per_head_gates=False,
    # Hyper-Attention params
    ha_rank=8,
    ha_plain_frac=0.5,
    ha_activation="silu",
)

low_rank_config = dict(
    enabled=False,
    mode="factorized",
    rank_ratio=0.25,
    rank=None,
    min_rank=1,
    max_rank=None,
    apply_attention=True,
    apply_mlp=True,
    noble_up_init_alpha=0.01,
    noble_lr_power=0.3,
    noble_mix_lr_power=0.45,
    noble_freq_lr_mul=3.0,
    noble_phase_lr_mul=5.0,
    noble_freq_min=0.8,
    noble_freq_max=1.2,
    noble_phase_std=0.1,
)

# FlexAttention setup
attention_config = dict(
    block_size=128,
    attention_scale=None,
    max_window_size=2048,
)

attention_pattern_config = dict(
    block_mask_pattern="S" * NUM_LAYERS,
    value_embed_layers=(
        VALUE_EMBED_HEAD_INDICES
        + [None] * (NUM_LAYERS - VALUE_EMBED_MID_LAYER_COUNT)
        + VALUE_EMBED_TAIL_INDICES
    ),
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

validation_inference = dict(
    enabled=False,
    num_samples=3,
    prompt_tokens=128,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    seed=42,
    stop_on_eos=True,
)

mtp_config = dict(
    enabled=False,
    schedule=[
        dict(mtp_weights_start=[1.0, 0.5, 0.25], mtp_weights_end=[1.0, 0.5, 0.0]),
        dict(mtp_weights_start=[1.0, 0.5], mtp_weights_end=[1.0, 0.0]),
        dict(mtp_weights_start=[1.0], mtp_weights_end=[1.0]),
    ],
    transitions=[1/3, 2/3],
)


compilation_config = dict(
    compile_model=True,
    relaxed_compile=True,
)
