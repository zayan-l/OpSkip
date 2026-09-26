"""Run one image with a fixed Op-Skip policy."""

import argparse
from opskip.policy import add_policy_options
import torch
from opskip import apply_opskip, load_policy
from opskip.inference import load_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    add_policy_options(parser)
    args = parser.parse_args()
    policy = load_policy(args.policy, use_triton=args.use_triton, freeze_scope=args.freeze_scope)
    model, inputs, decode = load_example(policy.model_family, args.model, args.image, args.prompt, args.device)
    apply_opskip(model, policy)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1,
                                 max_new_tokens=args.max_new_tokens, use_cache=True)
    print(decode(output))


if __name__ == "__main__":
    main()

