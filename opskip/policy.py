import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Policy:
    model_family: str
    num_layers: int
    attention_only: tuple[int, ...] = ()
    ffn_only: tuple[int, ...] = ()
    freeze: tuple[int, ...] = ()
    freeze_scope: str = "visual"
    use_triton: bool = False
    mlp_chunk_size: int = 0
    # Legacy JSON field; use_triton is the master switch for all fused kernels.
    rmsnorm_backend: str = "torch"

    def __post_init__(self):
        if self.model_family not in {"qwen3_vl", "qwen2_5_vl", "llava", "llava_next"}:
            raise ValueError(f"Unsupported model family: {self.model_family}")
        if type(self.num_layers) is not int or self.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        seen = set()
        for name in ("attention_only", "ffn_only", "freeze"):
            layers = getattr(self, name)
            if not isinstance(layers, tuple):
                raise ValueError(f"{name} must be a tuple of layer indices")
            for index in layers:
                if type(index) is not int or not 0 <= index < self.num_layers:
                    raise ValueError(f"Invalid layer index in {name}: {index}")
                if index in seen:
                    raise ValueError(f"Layer {index} occurs more than once")
                seen.add(index)
        if self.freeze_scope not in {"visual", "prefix_through_visual"}:
            raise ValueError("freeze_scope must be visual or prefix_through_visual")
        if type(self.use_triton) is not bool:
            raise ValueError("use_triton must be a bool")
        if type(self.mlp_chunk_size) is not int or self.mlp_chunk_size < 0:
            raise ValueError("mlp_chunk_size must be a nonnegative integer")
        if self.rmsnorm_backend not in {"torch", "triton_rounded"}:
            raise ValueError("rmsnorm_backend must be torch or triton_rounded")
        if self.rmsnorm_backend == "triton_rounded" and self.model_family != "qwen2_5_vl":
            raise ValueError("Explicit RMSNorm module replacement is specific to Qwen2.5")

    @property
    def budget(self):
        return len(self.attention_only) + len(self.ffn_only) + len(self.freeze)

    def action(self, index):
        for name in ("freeze", "attention_only", "ffn_only"):
            if index in getattr(self, name):
                return name
        return "full"


def load_policy(path, *, use_triton=None, freeze_scope=None):
    if isinstance(path, Policy):
        return replace(path, **{k: v for k, v in dict(use_triton=use_triton, freeze_scope=freeze_scope).items() if v is not None})
    data = json.loads(Path(path).read_text())
    allowed = {"model_family", "num_layers", "attention_only", "ffn_only", "freeze",
               "freeze_scope", "use_triton", "mlp_chunk_size", "rmsnorm_backend", "budget", "checkpoint", "source", "preset"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
    fields = {key: value for key, value in data.items() if key in Policy.__dataclass_fields__}
    for key in ("attention_only", "ffn_only", "freeze"):
        fields[key] = tuple(fields.get(key, ()))
    fields.update({k: v for k, v in dict(use_triton=use_triton, freeze_scope=freeze_scope).items() if v is not None})
    policy = Policy(**fields)
    if "budget" in data and (type(data["budget"]) is not int or data["budget"] != policy.budget):
        raise ValueError(f"Declared budget {data['budget']} does not match {policy.budget} selected layers")
    return policy


def add_policy_options(parser):
    import argparse
    parser.add_argument("--use-triton", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable optional Triton RMSNorm/RoPE kernels (default: off)")
    parser.add_argument("--freeze-scope", choices=["visual", "prefix_through_visual"], default=None,
                        help="Freeze visual tokens only (default), or the entire prefix through the last visual token")
