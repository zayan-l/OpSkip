from dataclasses import dataclass, field
import torch

from .attention import cache_llava_next_kv, selected_attention


@dataclass
class PrefillState:
    optimized: bool = False
    use_triton: bool = False
    memo: dict = field(default_factory=dict)
    rotary_groups: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    suffix_start: int = 0
    visual_mask: object = None
    nonvisual: object = None
    suffix: object = None
    sequence_length: int = 0

    def reset(self, visual_mask=None, *, allow_empty_suffix=False):
        self.memo.clear()
        self.visual_mask = visual_mask
        self.nonvisual = self.suffix = None
        self.sequence_length = 0
        if visual_mask is None or not visual_mask.any():
            self.visual_mask = None
            return
        if visual_mask.ndim != 2 or visual_mask.shape[0] != 1:
            raise ValueError("Op-Skip currently supports batch_size=1")
        visual = visual_mask[0].nonzero().flatten()
        self.sequence_length = visual_mask.shape[1]
        end = int(visual[-1]) + 1
        self.suffix_start = end
        if end == self.sequence_length and not allow_empty_suffix:
            raise ValueError("Op-Skip requires text after the visual tokens")
        self.nonvisual = (~visual_mask[0]).nonzero().flatten()
        self.suffix = torch.arange(end, self.sequence_length, device=visual_mask.device)


def cached_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return cache.get_seq_length()
    if isinstance(cache, (tuple, list)) and len(cache):
        return cache[0][0].shape[-2]
    raise ValueError("Unsupported KV cache type")


def mlp_update(layer, hidden, chunk_size):
    if not chunk_size or hidden.shape[1] <= chunk_size:
        return layer.mlp(hidden)
    return torch.cat([layer.mlp(part) for part in hidden.split(chunk_size, dim=1)], dim=1)


def operator_forward(layer, hidden, action, state, policy, kwargs):
    if layer.training or torch.is_grad_enabled():
        raise RuntimeError("Op-Skip is inference-only; call model.eval() inside torch.inference_mode()")
    if hidden.shape[0] != 1 or hidden.shape[1] != state.sequence_length:
        raise ValueError("Visual positions do not match the decoder input")
    if kwargs.get("output_attentions", False):
        raise ValueError("Op-Skip does not materialize attention weights")
    family = policy.model_family
    attention_kwargs = dict(kwargs)
    attention_kwargs.pop("output_attentions", None)
    cache = attention_kwargs.pop("past_key_values", None)
    if cache is None:
        cache = attention_kwargs.pop("past_key_value", None)
    else:
        attention_kwargs.pop("past_key_value", None)
    nonvisual = state.nonvisual.to(hidden.device)
    if state.use_triton:
        from .kernels.exact import rmsnorm
        norm = rmsnorm
    else:
        norm = lambda module, x: module(x)
    prefix_freeze = action == "freeze" and policy.freeze_scope == "prefix_through_visual"
    frozen_queries = state.suffix if prefix_freeze else state.nonvisual
    normed = norm(layer.input_layernorm, hidden)
    if family == "llava_next" and action == "freeze" and frozen_queries.numel() == 0:
        # No suffix queries exist, but future decode tokens still need this
        # layer's prefix K/V. Preserve every hidden row without computing Q/O,
        # SDPA, post-attention normalization, or the FFN.
        cache_llava_next_kv(layer.self_attn, normed, past_key_value=cache,
                            state=state, **attention_kwargs)
        output = hidden
    elif action == "attention_only":
        cache_name = "past_key_values" if family == "qwen3_vl" else "past_key_value"
        attention_kwargs[cache_name] = cache
        delta = layer.self_attn(hidden_states=normed, **attention_kwargs)[0]
        post = hidden + delta
        # RMSNorm reduces within each token, so only normalize rows consumed by the MLP.
        selected_post = post.index_select(1, nonvisual)
        mlp_input = (norm(layer.post_attention_layernorm, selected_post) if state.optimized
                     else norm(layer.post_attention_layernorm, post).index_select(1, nonvisual))
        update = mlp_update(layer, mlp_input, policy.mlp_chunk_size)
        output = post if state.optimized else post.clone()
        output.index_copy_(1, nonvisual, selected_post + update)
    else:
        queries = frozen_queries.to(hidden.device) if action == "freeze" else nonvisual
        delta = selected_attention(layer.self_attn, normed, queries, family,
                                   past_key_value=cache, state=state,
                                   suffix_start=state.suffix_start if prefix_freeze else None,
                                   **attention_kwargs)
        post = hidden.clone()
        selected = hidden[:, state.suffix_start:] if state.optimized and prefix_freeze else hidden.index_select(1, queries)
        updated = selected + delta
        if action == "freeze":
            # Update text rows selected by the configured freeze scope.
            updated = updated + mlp_update(layer, norm(layer.post_attention_layernorm, updated), policy.mlp_chunk_size)
            if state.optimized and prefix_freeze:
                post[:, state.suffix_start:].copy_(updated)
            else:
                post.index_copy_(1, queries, updated)
            output = post
        else:
            post.index_copy_(1, queries, updated)
            update = mlp_update(layer, norm(layer.post_attention_layernorm, post), policy.mlp_chunk_size)
            output = post.add_(update) if state.optimized else post + update
    if family == "qwen3_vl":
        return output
    result = (output,)
    if kwargs.get("use_cache", False):
        result += (cache,)
    return result
