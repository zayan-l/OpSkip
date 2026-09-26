import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from opskip import apply_opskip, load_policy, remove_opskip
from opskip.inference import decoder_for, load_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="outputs/latency.json")
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be nonnegative")
    policy = load_policy(args.policy)
    model, inputs, _ = load_example(policy.model_family, args.model, args.image, args.prompt, args.device)
    decoder = decoder_for(model, policy.model_family)
    records = {}

    def sync():
        if args.device != "cpu":
            torch.cuda.synchronize(args.device)

    with torch.inference_mode():
        for mode in ("vanilla", "opskip"):
            if mode == "opskip":
                apply_opskip(model, policy)
            samples = []
            timing = {}

            def start(module, call_args):
                sync()
                timing["start"] = time.perf_counter()

            def finish(module, call_args, result):
                sync()
                timing["decoder_ms"] = (time.perf_counter() - timing["start"]) * 1000

            for i in range(args.warmup + args.repeats):
                # Two passes separate decoder-hook overhead from end-to-end latency.
                hooks = [decoder.register_forward_pre_hook(start), decoder.register_forward_hook(finish)]
                try:
                    model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                sync()
                begin = time.perf_counter()
                model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True)
                sync()
                latency = (time.perf_counter() - begin) * 1000
                if i >= args.warmup:
                    samples.append({"first_token_ms": latency, "decoder_prefill_ms": timing["decoder_ms"]})
            records[mode] = {"optimized_prefill": mode == "opskip",
                             "optimized_prefill_backend": "fused_selected_v3" if mode == "opskip" else None,
                             "samples": samples, **{
                key: statistics.median(row[key] for row in samples)
                for key in ("first_token_ms", "decoder_prefill_ms")}}
        remove_opskip(model)
    output = {"model": args.model, "policy": json.loads(Path(args.policy).read_text()),
              "image": args.image, "prompt": args.prompt, "torch": torch.__version__,
              "device": args.device, "gpu": torch.cuda.get_device_name(args.device) if args.device != "cpu" else None,
              "warmup": args.warmup, "repeats": args.repeats, "aggregation": "median",
              "first_token_scope": "generate(max_new_tokens=1), including vision encoder; excluding loading and CPU preprocessing",
              "decoder_scope": "one text decoder prefill, separately instrumented", "results": records}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
