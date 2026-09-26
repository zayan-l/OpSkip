import inspect
from functools import wraps

from .patch import replace_method
from .runtime import cached_length


def setup(model, state, changes, family):
    if model.config.model_type != "qwen3_vl":
        raise ValueError("This policy requires dense Qwen3-VL")
    decoder = model.model.language_model
    original = decoder.forward
    signature = inspect.signature(original)

    @wraps(original.__func__)
    def forward(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        mask = bound.get("visual_pos_masks")
        cache = bound.get("past_key_values")
        state.reset(mask if cached_length(cache) == 0 else None)
        try:
            return original(*args, **kwargs)
        finally:
            state.reset()

    replace_method(decoder, "forward", forward, changes)
    return decoder.layers

