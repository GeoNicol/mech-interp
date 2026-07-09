#!/usr/bin/env python3
"""
emergence.py — watch induction heads being BORN: the in-context learning phase change.

Olsson et al. (2022) found that induction heads don't grow gradually: they emerge in a
sudden phase change partway through training, and in-context learning ability appears at
exactly the same moment — key evidence that induction heads cause in-context learning.
EleutherAI's Pythia models publish checkpoints throughout training, so this experiment
replays our induction-score + loss-cliff measurements across training time:

  for each checkpoint: load, measure (a) per-head induction scores, (b) 1st- and
  2nd-copy loss on the repeated random sequence. Plot everything against training step.

The signature to look for: 2nd-copy loss and top induction score are flat and boring,
then both snap within a few thousand steps of each other — the phase change.

Progress is checkpointed to results/emergence_{TAG}.csv (+ a .pt with the full score
matrices), so an interrupted run resumes where it left off.

  python 07_emergence/emergence.py                          # pythia-160m (default)
  python 07_emergence/emergence.py EleutherAI/pythia-410m   # any Pythia size

Outputs land in results/ next to this script.
"""
import csv, sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-160m"
TAG = MODEL.split("/")[-1].replace(".", "_")

# Pythia checkpoints: powers of 2 up to 512, then every 1000 steps. One step = ~2M tokens,
# and the induction bump is expected around 2-5B tokens (~step 1000-2500), so the grid is
# dense there and sparse elsewhere.
STEPS = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 5000, 8000, 16000, 32000, 64000, 143000]

CSV_PATH = OUT / f"emergence_{TAG}.csv"
PT_PATH = OUT / f"emergence_scores_{TAG}.pt"

# ── resume support: skip checkpoints already measured ─────────────────────────
done = {}
if CSV_PATH.exists():
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            done[int(row["step"])] = {k: float(v) for k, v in row.items()}
score_mats = torch.load(PT_PATH) if PT_PATH.exists() else {}

def measure(step):
    """Load one checkpoint, return (loss 1st copy, loss 2nd copy, induction score matrix)."""
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step,
                                              device=DEV, dtype=torch.float32)
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    torch.manual_seed(0)                       # same stimulus at every checkpoint
    bos = model.tokenizer.bos_token_id
    if bos is None:
        bos = model.tokenizer.eos_token_id
    prefix = [[bos]] if bos is not None else [[]]
    P = len(prefix[0])
    rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))
    tokens = torch.cat([torch.tensor(prefix, dtype=torch.long), rand, rand], dim=1).to(DEV)

    loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True)[0].float().cpu()
    _, cache = model.run_with_cache(tokens)
    induction = torch.zeros(nL, nH)
    for L in range(nL):
        patt = cache[f"blocks.{L}.attn.hook_pattern"]
        induction[L] = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1)[0].cpu()
    del cache, model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    l1 = loss[slice(P, P + SEQ - 1)].mean().item()          # within the 1st copy (skip prefix pred)
    l2 = loss[slice(P + SEQ - 1, P + 2 * SEQ - 1)].mean().item()  # the 2nd copy
    return l1, l2, induction

# ── sweep the checkpoints ─────────────────────────────────────────────────────
print(f"model: {MODEL}   checkpoints: {len(STEPS)} ({len(done)} already done)")
for step in STEPS:
    if step in done:
        continue
    l1, l2, mat = measure(step)
    top5 = mat.flatten().topk(5).values
    row = {"step": step, "loss_copy1": l1, "loss_copy2": l2,
           "top_score": top5[0].item(), "top5_mean": top5.mean().item()}
    done[step] = row
    score_mats[step] = mat
    torch.save(score_mats, PT_PATH)
    with open(CSV_PATH, "w", newline="") as f:               # rewrite whole CSV each time (tiny)
        w = csv.DictWriter(f, fieldnames=["step", "loss_copy1", "loss_copy2", "top_score", "top5_mean"])
        w.writeheader()
        for s in sorted(done):
            w.writerow(done[s])
    print(f"  step {step:>6}: 1st-copy loss {l1:.2f}   2nd-copy loss {l2:.2f}   "
          f"top induction score {row['top_score']:.2f}")

rows = [done[s] for s in sorted(done)]
steps = [r["step"] for r in rows]
x = [max(s, 1) for s in steps]   # log axis; step 0 plotted at 1

# ── chart 1: the phase change ─────────────────────────────────────────────────
# Loss on both copies (left axis) and top induction score (right axis) vs training step.
# The moment the score snaps up, the 2nd-copy loss snaps down — while 1st-copy loss
# barely moves. That co-timing is the evidence that induction heads drive in-context
# learning.
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.plot(x, [r["loss_copy1"] for r in rows], color="#9FAAAD", lw=1.6, marker="o", ms=4, label="loss, 1st copy (no lookback possible)")
ax1.plot(x, [r["loss_copy2"] for r in rows], color="#0E9E76", lw=2.0, marker="o", ms=4, label="loss, 2nd copy (in-context learning)")
ax1.set_xscale("log")
ax1.set_xlabel("training step (log scale, ~2M tokens/step)")
ax1.set_ylabel("loss (nats)")
ax2 = ax1.twinx()
ax2.plot(x, [r["top_score"] for r in rows], color="#C0392B", lw=2.0, marker="s", ms=4, label="top induction score")
ax2.plot(x, [r["top5_mean"] for r in rows], color="#C0392B", lw=1.2, ls="--", marker="s", ms=3, alpha=0.6, label="top-5 mean induction score")
ax2.set_ylabel("induction score", color="#C0392B")
ax2.set_ylim(0, 1)
h1, l1_ = ax1.get_legend_handles_labels()
h2, l2_ = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1_ + l2_, loc="center left", fontsize=8)
ax1.set_title(f"The phase change: induction heads and in-context learning appear together\n{MODEL}", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / f"emergence_12_phase_{TAG}.png", dpi=120); print(f"→ results/emergence_12_phase_{TAG}.png")

# ── chart 2: heatmap filmstrip of the birth ───────────────────────────────────
# Four snapshots of the full (layer, head) score matrix: before, during, after the
# transition, and fully trained.
picks = [s for s in [256, 1000, 4000, 143000] if s in score_mats]
if len(picks) >= 2:
    vmax = max(score_mats[s].max().item() for s in picks)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.8), squeeze=False)
    fig.subplots_adjust(top=0.78)
    for i, s in enumerate(picks):
        ax = axes[0][i]
        im = ax.imshow(score_mats[s], cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"step {s}", fontsize=10)
        ax.set_xlabel("head")
        if i == 0:
            ax.set_ylabel("layer")
    fig.colorbar(im, ax=[axes[0][i] for i in range(len(picks))], label="induction score", fraction=0.03)
    fig.suptitle(f"Birth of the induction heads — {MODEL}", fontweight="bold")
    fig.savefig(OUT / f"emergence_13_filmstrip_{TAG}.png", dpi=120, bbox_inches="tight"); print(f"→ results/emergence_13_filmstrip_{TAG}.png")
