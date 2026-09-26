import argparse
from opskip.policy import add_policy_options
import importlib
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .policy import load_policy


MODELS = {
    "qwen3_vl": ("qwen3_vl", "Qwen3_VL"),
    "qwen2_5_vl": ("qwen2_5_vl", "Qwen2_5_VL"),
    "llava": ("llava", "Llava"),
    "llava_next": ("llava", "Llava"),
}
TASKS = "pope,gqa,mmmu_val,textvqa_val,scienceqa_img,mmbench_en_dev,mme,ai2d,vizwiz_vqa_val,ocrbench"


def run_evaluator(module, cli):
    """Propagate failures swallowed by the upstream CLI to the job exit code."""
    original_single = module.cli_evaluate_single
    original_argv = sys.argv
    failures = []

    def evaluate_single(*args, **kwargs):
        try:
            return original_single(*args, **kwargs)
        except Exception as error:
            failures.append(error)
            raise

    module.cli_evaluate_single = evaluate_single
    try:
        sys.argv = ["lmms_eval", *cli]
        module.cli_evaluate()
        if failures:
            raise RuntimeError("lmms-eval failed; evaluation is incomplete") from failures[0]
    finally:
        module.cli_evaluate_single = original_single
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=MODELS, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face ID or local checkpoint")
    parser.add_argument("--policy", required=True, help="Policy JSON, or 'off' for vanilla")
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved arguments without loading a model")
    add_policy_options(parser)
    args = parser.parse_args()
    policy = None if args.policy == "off" else load_policy(args.policy, use_triton=args.use_triton, freeze_scope=args.freeze_scope)
    if policy is not None and policy.model_family != args.family:
        parser.error("Policy model_family does not match --family")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    model_id, class_name = MODELS[args.family]
    model_args = f"pretrained={args.model},attn_implementation=sdpa,device_map={args.device},use_cache=True"
    if args.family.startswith("qwen"):
        for name in ("min_pixels", "max_pixels"):
            value = getattr(args, name)
            if value is not None:
                model_args += f",{name}={value}"
    elif args.min_pixels is not None or args.max_pixels is not None:
        parser.error("Pixel bounds are only supported by Qwen")
    if args.family == "llava_next":
        model_args += ",conv_template=vicuna_v1,truncate_context=False"
    for value in (args.model, args.device):
        if "," in value:
            parser.error("Model paths and device names cannot contain commas (lmms-eval argument syntax)")
    cli = ["--model", model_id, "--model_args", model_args, "--tasks", args.tasks,
           "--batch_size", "1", "--device", args.device, "--output_path", args.output,
           "--log_samples", "--force_simple"]
    if args.limit is not None:
        cli += ["--limit", str(args.limit)]
    resolved = {"family": args.family, "checkpoint": args.model,
                "optimized_prefill": policy is not None,
                "optimized_prefill_backend": "fused_selected_v3" if policy is not None else None,
                "empty_suffix_freeze": "kv_only" if args.family == "llava_next" and policy is not None else None,
                "policy": {**asdict(policy), "budget": policy.budget} if policy else None,
                "policy_source": args.policy, "lmms_eval_args": cli}
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    import torch

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"[OpSkip] Initializing CUDA before evaluator imports: {device}", flush=True)
        torch.cuda.set_device(device)
        print("[OpSkip] CUDA initialization completed", flush=True)

    from . import apply_opskip
    import lmms_eval.models as models

    module = importlib.import_module(f"lmms_eval.models.simple.{model_id}")
    if "zipdemo" in inspect.getsource(module).lower():
        raise RuntimeError("Install a clean upstream lmms-eval v0.6.1; this installation contains experimental model changes")
    base = getattr(module, class_name)

    class OpSkipModel(base):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            if self.accelerator.num_processes != 1:
                raise ValueError("Use one Python process for this evaluation entry point")
            if policy is not None:
                apply_opskip(self.model, policy)

    original_get_model = models.get_model

    def get_model(name, force_simple=False):
        return OpSkipModel if name == model_id else original_get_model(name, force_simple)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "opskip_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    models.get_model = get_model
    try:
        import lmms_eval.__main__ as evaluator_cli
        run_evaluator(evaluator_cli, cli)
    finally:
        models.get_model = original_get_model


if __name__ == "__main__":
    main()
