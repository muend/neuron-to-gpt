"""Validated model dimensions and conservative hardware presets."""

from dataclasses import asdict, dataclass

STAGES = ("bigram", "single_head", "multi_head", "feedforward", "residual", "layernorm")


@dataclass
class ModelConfig:
    vocab_size: int
    stage: str = "layernorm"
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 3
    block_size: int = 128
    dropout: float = .2
    initialize_small: bool = True

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage: {self.stage}")
        if min(self.vocab_size, self.n_embd, self.n_head, self.n_layer, self.block_size) < 1:
            raise ValueError("All dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.stage != "layernorm" and self.n_layer != 1:
            raise ValueError("Only the completed layernorm model may stack blocks")
        if self.stage in ("bigram", "single_head") and self.dropout:
            raise ValueError("Historical baseline stages do not use dropout")

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "ablation": dict(n_embd=64, n_head=4, n_layer=1, block_size=32, dropout=0.,
                     batch_size=32, steps=4000, learning_rate=.003, initialize_small=False),
    "t4": dict(n_embd=128, n_head=4, n_layer=3, block_size=128, dropout=.2,
               batch_size=64, steps=5000, learning_rate=.0003, initialize_small=True),
    "a100": dict(n_embd=256, n_head=8, n_layer=6, block_size=256, dropout=.2,
                 batch_size=64, steps=5000, learning_rate=.0003, initialize_small=True),
    "smoke": dict(n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=.2,
                  batch_size=4, steps=20, learning_rate=.001, initialize_small=True),
}
