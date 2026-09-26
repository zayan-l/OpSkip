import inspect
from functools import wraps

from .patch import replace_method
from .runtime import cached_length


def setup(model, state, changes, family):
    if model.config.model_type != "qwen2_5_vl":
        raise ValueError("This policy requires Qwen2.5-VL")
    original = model.forward
    signature = inspect.signature(original)

    @wraps(original.__func__)
    def forward(self, *args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        ids = values.get("input_ids")
        mask = None
        if cached_length(values.get("past_key_values")) == 0:
            if ids is not None:
                mask = (ids == self.config.image_token_id) | (ids == self.config.video_token_id)
                # The experimental Qwen2.5 implementation stores a single range,
                # including any separator/text rows between multiple visual blocks.
                if mask.shape[0] == 1 and mask.any():
                    visual = mask[0].nonzero().flatten()
                    mask[0, int(visual[0]):int(visual[-1]) + 1] = True
            elif values.get("pixel_values") is not None or values.get("pixel_values_videos") is not None:
                raise ValueError("Qwen2.5 Op-Skip requires input_ids to locate visual tokens")
        state.reset(mask)
        try:
            return original(*args, **kwargs)
        finally:
            state.reset()

    replace_method(model, "forward", forward, changes)
    return model.model.layers
