#!/usr/bin/env python3
"""Re-measure a training run's saved checkpoints with the current probe design.

train_emergence.py originally measured with a SINGLE 50-token probe block; per-head
scores from one block wobble ±0.01–0.02 — the scale of the pre-ignition signal being
read. This script replays every saved checkpoint for a tag through the upgraded
measurement (N_PROBE averaged blocks + prev-token scores) and rewrites the run's CSV
and score-matrix files in the new schema, preserving the logged train_loss. Run it
BEFORE resuming training with the new schema (the training process rewrites the CSV
from memory, clobbering external edits).

  python 12_train_emergence/remeasure.py                    # tiny-4L256-mix30
  python 12_train_emergence/remeasure.py tiny-4L256         # the pure-TinyStories run
"""
import csv, sys, torch
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent
OUT, CKPT = HERE / "results", HERE / "checkpoints"
TAG = sys.argv[1] if len(sys.argv) > 1 else "tiny-4L256-mix30"
SEQ, P, N_PROBE = 50, 1, 16                    # must match train_emergence.py
FIELDS = ["step", "tokens", "train_loss", "loss_copy1", "loss_copy2",
          "top_score", "top5_mean", "prev_top", "prev_top5"]

CSV_PATH = OUT / f"training_{TAG}.csv"
PT_PATH = OUT / f"training_scores_{TAG}.pt"
PREV_PT_PATH = OUT / f"training_prev_{TAG}.pt"

ckpts = sorted(CKPT.glob(f"{TAG}_step*.pt"),
               key=lambda p: int(p.stem.rsplit("step", 1)[1]))
assert ckpts, f"no checkpoints found for tag {TAG}"

# weights_only=False: our own files (holds optimizer state + config); never load
# untrusted .pt files this way.
state = torch.load(ckpts[0], map_location=DEV, weights_only=False)
cfg = HookedTransformerConfig.from_dict(state["cfg"])
model = HookedTransformer(cfg).to(DEV)
model.eval()
EOS = model.tokenizer.eos_token_id
NL, NH = cfg.n_layers, cfg.n_heads

# identical probe construction to train_emergence.py (same seed: block 0 of this
# draw is bit-identical to the original single-probe stimulus)
torch.manual_seed(0)
rand = torch.randint(0, cfg.d_vocab, (N_PROBE, SEQ))
probe = torch.cat([torch.full((N_PROBE, 1), EOS, dtype=torch.long), rand, rand], dim=1).to(DEV)


def measure():
    loss = model(probe, return_type="loss", loss_per_token=True).float().cpu()
    _, cache = model.run_with_cache(probe, names_filter=lambda n: n.endswith("hook_pattern"))
    ind = torch.zeros(NL, NH)
    prev = torch.zeros(NL, NH)
    for L in range(NL):
        patt = cache[f"blocks.{L}.attn.hook_pattern"]
        ind[L] = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
        prev[L] = patt.diagonal(offset=-1, dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
    l1 = loss[:, P:P + SEQ - 1].mean().item()
    l2 = loss[:, P + SEQ - 1:P + 2 * SEQ - 1].mean().item()
    return l1, l2, ind, prev


old = {}
if CSV_PATH.exists():
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            old[int(row["step"])] = row

done, score_mats, prev_mats = {}, {}, {}
for p in ckpts:
    step = int(p.stem.rsplit("step", 1)[1])
    state = torch.load(p, map_location=DEV, weights_only=False)
    model.load_state_dict(state["model"])
    l1, l2, ind, prev = measure()
    top5 = ind.flatten().topk(5).values
    ptop5 = prev.flatten().topk(5).values
    tokens = old.get(step, {}).get("tokens", "")
    done[step] = {"step": step, "tokens": tokens,
                  "train_loss": old.get(step, {}).get("train_loss", "nan"),
                  "loss_copy1": l1, "loss_copy2": l2,
                  "top_score": top5[0].item(), "top5_mean": top5.mean().item(),
                  "prev_top": ptop5[0].item(), "prev_top5": ptop5.mean().item()}
    score_mats[step] = ind
    prev_mats[step] = prev
    print(f"step {step:>6}: 1st-copy {l1:5.2f}   2nd-copy {l2:5.2f}   "
          f"top induction {top5[0]:.3f}   top prev {ptop5[0]:.3f}")

torch.save(score_mats, PT_PATH)
torch.save(prev_mats, PREV_PT_PATH)
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for s in sorted(done):
        w.writerow(done[s])
print(f"rewrote {CSV_PATH.name} ({len(done)} rows), {PT_PATH.name}, {PREV_PT_PATH.name}")
