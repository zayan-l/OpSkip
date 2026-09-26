from functools import wraps
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


FP_OPTIONS = ({'enable_fp_fusion': False}
              if triton is not None and int(triton.__version__.split('.')[0]) >= 3 else {})

if triton is not None:
    _rsqrt = getattr(tl, 'rsqrt', tl.math.rsqrt)
    @triton.jit(do_not_specialize=['N'])
    def _rope(X, C, S, P, Y, N, H: tl.constexpr, D: tl.constexpr,
              XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
              CS: tl.constexpr, INDEXED: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < H * N * D
        d = i % D
        n = (i // D) % N
        h = i // (D * N)
        pos = tl.load(P + n, n < N, 0) if INDEXED else n
        x = tl.load(X + h * XS0 + n * XS1 + d * XS2, valid, 0).to(tl.float32)
        other = (d + D // 2) % D
        rotated = tl.load(X + h * XS0 + n * XS1 + other * XS2, valid, 0).to(tl.float32)
        rotated = tl.where(d < D // 2, -rotated, rotated)
        c = tl.load(C + pos * CS + d, valid, 0).to(tl.float32)
        s = tl.load(S + pos * CS + d, valid, 0).to(tl.float32)
        a = (x * c).to(Y.dtype.element_ty).to(tl.float32)
        b = (rotated * s).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + i, a + b, valid)

    @triton.jit
    def _norm(X, W, Y, C: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + row * C + cols, cols < C, 0).to(tl.float32)
        var = tl.sum(x * x, 0) / C
        normalized = (x * _rsqrt(var + EPS)).to(Y.dtype.element_ty).to(tl.float32)
        w = tl.load(W + cols, cols < C, 0).to(tl.float32)
        tl.store(Y + row * C + cols, normalized * w, cols < C)


def _fused_rmsnorm(module, x):
    """Launch the same rounded kernel for all supported model families."""
    cols = x.shape[-1]
    out = torch.empty_like(x)
    _norm[(x.numel() // cols,)](x, module.weight, out, cols, module.variance_epsilon,
                              BLOCK=triton.next_power_of_2(cols),
                              num_warps=4 if cols <= 1024 else 8,
                              **FP_OPTIONS)
    return out


def rmsnorm(module, x):
    eligible = (triton is not None and x.is_cuda and x.is_contiguous()
                and x.dtype in (torch.float16, torch.bfloat16)
                and module.__class__.__name__ in ('Qwen3VLTextRMSNorm', 'LlamaRMSNorm')
                and module.weight.dtype == x.dtype and module.weight.is_contiguous()
                and x.numel() and not torch.is_grad_enabled())
    if not eligible:
        # Qwen2RMSNorm delegates to its installed wrapper, or to upstream when
        # the policy selects torch. Its wrapper never calls this dispatcher.
        return module(x)
    return _fused_rmsnorm(module, x)


def install_qwen25_rmsnorm(norm, changes):
    """Use the shared kernel on Qwen2.5 modules in prefill and decode."""
    from ..patch import replace_method
    original = norm.forward

    @wraps(original.__func__)
    def forward(self, hidden_states):
        weight = self.weight
        eligible = (hidden_states.is_cuda and weight.is_cuda
                    and hidden_states.dtype in {torch.float16, torch.bfloat16}
                    and weight.dtype == hidden_states.dtype and hidden_states.numel() > 0
                    and hidden_states.ndim >= 2 and hidden_states.is_contiguous()
                    and weight.is_contiguous() and weight.numel() == hidden_states.shape[-1]
                    and not torch.is_grad_enabled())
        if not eligible:
            return original(hidden_states)
        if triton is None:
            raise RuntimeError("Qwen2.5 rounded RMSNorm requires Triton; install it or set use_triton=False")
        return _fused_rmsnorm(self, hidden_states)

    replace_method(norm, "forward", forward, changes)


def rotary(x, cos, sin, positions=None):
    """Return fused selected/full RoPE, or None when layout is unsupported."""
    if not (triton is not None and x.is_cuda and x.shape[0] == 1
            and x.dtype in (torch.bfloat16, torch.float16)
            and cos.dtype == sin.dtype == x.dtype and cos.ndim == sin.ndim == 3
            and cos.shape[0] == sin.shape[0] == 1
            and cos.stride(-1) == sin.stride(-1) == 1
            and cos.stride(1) == sin.stride(1) and not torch.is_grad_enabled()):
        return None
    _, heads, n, dim = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rope[(triton.cdiv(x.numel(), 256),)](x, cos, sin, positions if positions is not None else x, out,
          n, heads, dim, *x.stride()[1:], cos.stride(1), positions is not None,
          BLOCK=256, num_warps=4, **FP_OPTIONS)
    return out
