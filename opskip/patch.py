import inspect
from functools import wraps
from types import MethodType

from .policy import load_policy
from .runtime import PrefillState, operator_forward


def replace_method(obj, name, function, changes):
    # Remember whether this was an instance override, so removal restores lookup semantics.
    changes.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
    setattr(obj, name, MethodType(function, obj))


def _patch_layer(layer, action, state, policy, changes):
    original = layer.forward
    signature = inspect.signature(original)
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    keyword_names = frozenset(signature.parameters)

    @wraps(original.__func__)
    def forward(self, *args, **kwargs):
        if state.visual_mask is None:
            return original(*args, **kwargs)
        if (policy.model_family == "llava_next" and state.nonvisual.numel() == 0
                and action != "freeze"):
            return original(*args, **kwargs)
        if (state.optimized and len(args) <= 1 and "kwargs" not in kwargs
                and (accepts_extra or kwargs.keys() <= keyword_names)
                and ((len(args) == 1 and "hidden_states" not in kwargs)
                     or (not args and "hidden_states" in kwargs))):
            values = dict(kwargs)
            hidden = args[0] if args else values.pop("hidden_states")
        else:
            bound = signature.bind(*args, **kwargs)
            values = dict(bound.arguments)
            hidden = values.pop("hidden_states")
        # Decode retains the upstream forward and the full per-layer visual KV cache.
        if hidden.shape[1] <= 1:
            return original(*args, **kwargs)
        extra = values.pop("kwargs", {})
        values.update(extra)
        return operator_forward(self, hidden, action, state, policy, values)

    replace_method(layer, "forward", forward, changes)


def _parallel_text_mlp(layer, state, changes):
    import torch
    mlp = layer.mlp
    original = mlp.forward
    @wraps(original.__func__)
    def forward(self, x):
        if state.visual_mask is None or x.shape[-2] <= 1 or not x.is_cuda:
            return original(x)
        parent = torch.cuda.current_stream(x.device)
        key = ("gate", x.device)
        if key not in state.streams: state.streams[key] = torch.cuda.Stream(device=x.device)
        stream = state.streams[key]
        stream.wait_stream(parent)
        with torch.cuda.stream(stream):
            gate = self.act_fn(self.gate_proj(x))
            x.record_stream(stream); self.gate_proj.weight.record_stream(stream)
            if self.gate_proj.bias is not None: self.gate_proj.bias.record_stream(stream)
        up = self.up_proj(x)
        parent.wait_stream(stream); gate.record_stream(parent)
        return self.down_proj(gate * up)
    replace_method(mlp, "forward", forward, changes)


def apply_opskip(model, policy, *, use_triton=None, freeze_scope=None):
    """Enable a policy on a dense SDPA model.
    It preserves the operator policy, full KV cache and upstream decode path.
    """
    policy = load_policy(policy, use_triton=use_triton, freeze_scope=freeze_scope)
    if hasattr(model, "_opskip_changes"):
        raise ValueError("Op-Skip is already installed; remove_opskip(model) before changing policy")
    family = policy.model_family
    if family == "qwen3_vl":
        from .qwen3_vl import setup
    elif family == "qwen2_5_vl":
        from .qwen2_5_vl import setup
    else:
        from .llava import setup
    state, changes = PrefillState(optimized=True, use_triton=policy.use_triton), []
    try:
        layers = setup(model, state, changes, family)
        if len(layers) != policy.num_layers:
            raise ValueError(f"Policy expects {policy.num_layers} layers; model has {len(layers)}")
        for index, layer in enumerate(layers):
            attn = layer.self_attn
            if getattr(attn.config, "_attn_implementation", None) != "sdpa":
                raise ValueError("Load the model with attn_implementation='sdpa'")
            if int(getattr(attn.config, "pretraining_tp", 1)) != 1:
                raise ValueError("Op-Skip expects pretraining_tp=1")
            action = policy.action(index)
            if action != "full":
                _patch_layer(layer, action, state, policy, changes)
                if family == "llava_next" and action in ("freeze", "attention_only"):
                    _parallel_text_mlp(layer, state, changes)
        if policy.use_triton and family == "qwen2_5_vl":
            from .kernels.exact import install_qwen25_rmsnorm
            for norm in model.modules():
                if norm.__class__.__name__ == "Qwen2RMSNorm":
                    install_qwen25_rmsnorm(norm, changes)
    except Exception:
        _restore(changes)
        raise
    model._opskip_changes = changes
    model._opskip_policy = policy
    return model


def _restore(changes):
    for obj, name, had_value, value in reversed(changes):
        if had_value:
            setattr(obj, name, value)
        else:
            delattr(obj, name)


def remove_opskip(model):
    """Restore the instance's original methods."""
    if hasattr(model, "_opskip_changes"):
        _restore(model._opskip_changes)
        del model._opskip_changes
        del model._opskip_policy
    return model
