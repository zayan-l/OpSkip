import torch
from PIL import Image


def load_example(family, checkpoint, image_path, prompt, device="cuda:0"):
    image = Image.open(image_path).convert("RGB")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if family in {"qwen3_vl", "qwen2_5_vl"}:
        from transformers import AutoProcessor
        if family == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration as Model
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Model
        model = Model.from_pretrained(checkpoint, torch_dtype=dtype,
                                      attn_implementation="sdpa", device_map=device).eval()
        processor = AutoProcessor.from_pretrained(checkpoint)
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [{"type": "image", "image": image},
                     {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

        def decode(result):
            return processor.batch_decode(result[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        return model, dict(inputs), decode
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates

    tokenizer, model, image_processor, _ = load_pretrained_model(
        checkpoint, None, get_model_name_from_path(checkpoint),
        device_map=device, attn_implementation="sdpa")
    model.eval()
    token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        token = DEFAULT_IM_START_TOKEN + token + DEFAULT_IM_END_TOKEN
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], token + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)
    images = process_images([image], image_processor, model.config)
    dtype = next(model.parameters()).dtype
    if isinstance(images, list):
        images = [x.to(device=device, dtype=dtype) for x in images]
    else:
        images = images.to(device=device, dtype=dtype)
    inputs = {"inputs": ids, "images": images, "image_sizes": [image.size]}

    def decode(result):
        # Original LLaVA generates from inputs_embeds and returns generated IDs.
        return tokenizer.batch_decode(result, skip_special_tokens=True)[0].strip()

    return model, inputs, decode


def decoder_for(model, family):
    return model.model.language_model if family == "qwen3_vl" else model.model

