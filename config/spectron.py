from config.base import *

low_rank_config.update(
    {
        "enabled": True,
        "rank_ratio": 0.25,
        "apply_attention": True,
        "apply_mlp": True,
    }
)

optimizer_config.update(
    {
        "matrix_optimizer": "spectron",
    }
)
