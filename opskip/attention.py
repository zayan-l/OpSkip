import torch
import torch.nn.functional as F


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def repeat_kv(x, groups):
    if groups == 1:
        return x
    batch, heads, length, dim = x.shape
    return x[:, :, None].expand(batch, heads, groups, length, dim).reshape(batch, heads * groups, length, dim)


def selected_mask(mask, positions, length, dtype, device):
    blocked = torch.arange(length, device=device)[None, :] > positions[:, None]
    result = torch.zeros((1, 1, len(positions), length), dtype=dtype, device=device)
    result.masked_fill_(blocked, torch.finfo(dtype).min)
    if mask is None:
        return result
    if mask.ndim == 2:
        return result.masked_fill(~mask[:, None, None, :length].bool(), torch.finfo(dtype).min)
    if mask.ndim != 4:
        raise ValueError("Op-Skip expects a 2D padding mask or 4D attention mask")
    selected = mask[..., :length]
    if selected.shape[-2] != 1:
        selected = selected.index_select(-2, positions)
    if selected.dtype == torch.bool:
        return result.masked_fill(~selected, torch.finfo(dtype).min)
    return torch.minimum(result, selected.to(dtype))


def cache_llava_next_kv(attn, hidden, *, position_ids=None, past_key_value=None,
                        cache_position=None, state=None, **kwargs):
    """Populate NeXT's prefix cache without queries or attention/FFN updates."""
    batch, length, _ = hidden.shape
    k = attn.k_proj(hidden).view(batch, length, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(batch, length, -1, attn.head_dim).transpose(1, 2)
    if position_ids is None:
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
    optimized = state is not None and state.optimized
    group = state.rotary_groups.get(id(attn.rotary_emb), id(attn.rotary_emb)) if optimized else None
    key = ("llama_rotary", group, id(position_ids), length, v.dtype, v.device)
    if optimized and key in state.memo:
        _, cos, sin, _, _ = state.memo[key]
    else:
        cos, sin = attn.rotary_emb(v, position_ids)
        if optimized:
            state.memo[key] = (position_ids, cos, sin, cos, sin)
    rotated = None
    if state is not None and state.use_triton:
        from .kernels.exact import rotary
        rotated = rotary(k, cos, sin)
    k = (k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
         if rotated is None else rotated)
    if past_key_value is not None:
        past_key_value.update(k, v, attn.layer_idx,
                              {"cos": cos, "sin": sin, "cache_position": cache_position})


def selected_attention(attn, hidden, positions, family, *, attention_mask=None,
                       position_ids=None, position_embeddings=None, past_key_value=None,
                       cache_position=None, state=None, suffix_start=None, **kwargs):
    optimized = state is not None and state.optimized
    batch, length, _ = hidden.shape
    dim = attn.head_dim
    query_hidden = hidden[:, suffix_start:] if optimized and suffix_start is not None else hidden.index_select(1, positions)
    if optimized and family == "llava_next" and hidden.is_cuda:
        parent = torch.cuda.current_stream(hidden.device)
        key = ("qk", hidden.device)
        if key not in state.streams:
            state.streams[key] = (torch.cuda.Stream(device=hidden.device), torch.cuda.Stream(device=hidden.device))
        qs, ks = state.streams[key]
        qs.wait_stream(parent); ks.wait_stream(parent)
        with torch.cuda.stream(qs):
            q = attn.q_proj(query_hidden)
            query_hidden.record_stream(qs)
            attn.q_proj.weight.record_stream(qs)
            if attn.q_proj.bias is not None: attn.q_proj.bias.record_stream(qs)
        with torch.cuda.stream(ks):
            k = attn.k_proj(hidden)
            hidden.record_stream(ks)
            attn.k_proj.weight.record_stream(ks)
            if attn.k_proj.bias is not None: attn.k_proj.bias.record_stream(ks)
        v = attn.v_proj(hidden)
        parent.wait_stream(qs); parent.wait_stream(ks)
        q.record_stream(parent); k.record_stream(parent)
        q = q.view(batch, len(positions), -1, dim)
        k = k.view(batch, length, -1, dim)
        v = v.view(batch, length, -1, dim).transpose(1, 2)
    else:
        q = attn.q_proj(query_hidden).view(batch, len(positions), -1, dim)
        k = attn.k_proj(hidden).view(batch, length, -1, dim)
        v = attn.v_proj(hidden).view(batch, length, -1, dim).transpose(1, 2)
    if family == "qwen3_vl":
        if state is not None and state.use_triton:
            from .kernels.exact import rmsnorm
            q, k = rmsnorm(attn.q_norm, q), rmsnorm(attn.k_norm, k)
        else:
            q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)

    if family in {"qwen3_vl", "qwen2_5_vl"}:
        if position_embeddings is None:
            raise ValueError("Qwen Op-Skip requires shared rotary position embeddings")
        cos, sin = position_embeddings
        if family == "qwen2_5_vl":
            sections = tuple(attn.rope_scaling["mrope_section"] * 2)
            key = ('mrope', id(cos), id(sin), sections)
            if optimized and key in state.memo:
                _, _, cos, sin = state.memo[key]
            else:
                source_cos, source_sin = cos, sin
                cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
                sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
                if optimized: state.memo[key] = (source_cos, source_sin, cos, sin)
        q, k = apply_selected_rotary(q, k, cos, sin, positions, state is not None and state.use_triton)
    else:
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
        group = state.rotary_groups.get(id(attn.rotary_emb), id(attn.rotary_emb)) if optimized else None
        key = ("llama_rotary", group, id(position_ids),
               length, v.dtype, v.device)
        if optimized and key in state.memo:
            _, cos, sin, ck, sk = state.memo[key]
        else:
            # 4.37 LLaMA uses a rotary table; 4.40 LLaMA accepts position_ids.
            if family == "llava":
                cos, sin = attn.rotary_emb(v, seq_len=length)
                ck, sk = cos[position_ids], sin[position_ids]
            else:
                cos, sin = attn.rotary_emb(v, position_ids)
                ck, sk = cos, sin
            if optimized:
                state.memo[key] = (position_ids, cos, sin, ck, sk)
        q, k = apply_selected_rotary(q, k, ck, sk, positions, state is not None and state.use_triton)

    if past_key_value is not None:
        if family == "qwen3_vl":
            k, v = past_key_value.update(k, v, attn.layer_idx)
        else:
            k, v = past_key_value.update(k, v, attn.layer_idx,
                                         {"cos": cos, "sin": sin, "cache_position": cache_position})
    k = repeat_kv(k, attn.num_key_value_groups)
    v = repeat_kv(v, attn.num_key_value_groups)
    key = (id(attention_mask), id(positions), k.shape[-2], q.dtype, q.device)
    if optimized and key in state.memo:
        mask = state.memo[key][2]
    else:
        mask = selected_mask(attention_mask, positions, k.shape[-2], q.dtype, q.device)
        if optimized:
            # Keep source tensors alive: an object ID must not be recycled
            # for a different layer's mask during this decoder invocation.
            state.memo[key] = (attention_mask, positions, mask)
    # SDPA accepts strided head/sequence dimensions when the feature axis is contiguous.
    # Keep the original layout conversion for unoptimized execution.
    q, k, v = ((x if optimized and x.stride(-1) == 1 else x.contiguous()) for x in (q, k, v))
    output = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=mask, dropout_p=0.0, is_causal=False)
    output = output.transpose(1, 2).contiguous().reshape(batch, len(positions), -1)
    return attn.o_proj(output)


def apply_selected_rotary(q, k, cos, sin, positions, use_triton):
    if use_triton:
        from .kernels.exact import rotary
        qr, kr = rotary(q, cos, sin, positions), rotary(k, cos, sin)
        if qr is not None and kr is not None:
            return qr, kr
    cq, sq = cos.index_select(1, positions).unsqueeze(1), sin.index_select(1, positions).unsqueeze(1)
    return (q * cq + rotate_half(q) * sq,
            k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1))
