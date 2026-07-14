#!/usr/bin/env python3
"""
train_emergence.py — train a tiny GPT from scratch and watch induction heads being born.

Experiment 07 replayed the phase change from Pythia's PUBLISHED checkpoints; this one
removes the training wheels. A 4-layer, ~29M-parameter HookedTransformer (untied GPT-2
vocab embeddings dominate the count; ~3M live in the blocks) is trained from
scratch on TinyStories on the local GPU, with log-spaced checkpoints (dense through the
expected transition, sparse after). At every checkpoint the experiment-01 measurement
runs in-line — 1st/2nd-copy loss and per-head induction scores on the same repeated
random block — appending to a resumable CSV, so the phase change can be watched LIVE
from another terminal while training:  tail -f results/training_tiny-4L256.csv

Training the model as a HookedTransformer (not nanoGPT etc.) means every downstream tool
in this repo — the exp-07 charts, capture_emergence.py, the 3D viewer — can consume the
checkpoints without conversion.

This is the repo's first script that trains (grads ON) — everything else is inference.

  python 12_train_emergence/train_emergence.py        # full run: 30k steps ≈ 490M tokens
  python 12_train_emergence/train_emergence.py 300    # short run (outputs tagged _s300)

Resumable: reruns load the newest checkpoint for the tag and continue; CSV rows for
already-measured steps are kept. checkpoints/ is gitignored (only CSV + score matrices
+ charts are committed); TinyStories lives in the HF datasets cache like all weights.
"""
import csv, math, sys, time, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

OUT = Path(__file__).parent / "results"
CKPT = Path(__file__).parent / "checkpoints"
OUT.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── the model: smallest comfortable home for an induction circuit ─────────────
# 2 layers is the theoretical minimum (prev-token head → induction head needs one
# composition step); 4 gives it room so the circuit's *placement* is a finding, not
# forced. The GPT-2 tokenizer keeps the stimulus identical to experiment 01.
N_LAYERS, D_MODEL, N_HEADS, N_CTX = 4, 256, 8, 512
BATCH = 32                                     # sequences/step → 16,384 tokens/step
DEFAULT_STEPS = 30_000                         # ≈ 490M tokens ≈ one TinyStories epoch
LR, WARMUP, WD, CLIP = 1e-3, 200, 0.1, 1.0
SEQ, THRESH = 50, 0.3                          # experiment-01 measurement knobs

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEPS
TAG = f"tiny-{N_LAYERS}L{D_MODEL}" + ("" if TOTAL_STEPS == DEFAULT_STEPS else f"_s{TOTAL_STEPS}")
TOK_PER_STEP = BATCH * N_CTX

# log-spaced checkpoints: powers of two + halfway points, always incl. 0 and the end
CKPT_STEPS = sorted({0, TOTAL_STEPS,
                     *(2 ** k for k in range(5, 15)),
                     *(3 * 2 ** k for k in range(7, 14))})
CKPT_STEPS = [s for s in CKPT_STEPS if s <= TOTAL_STEPS]

CSV_PATH = OUT / f"training_{TAG}.csv"
PT_PATH = OUT / f"training_scores_{TAG}.pt"
FIELDS = ["step", "tokens", "train_loss", "loss_copy1", "loss_copy2", "top_score", "top5_mean"]

cfg = HookedTransformerConfig(
    n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_MODEL // N_HEADS,
    d_mlp=4 * D_MODEL, n_ctx=N_CTX, act_fn="gelu", normalization_type="LN",
    tokenizer_name="gpt2", seed=0)
model = HookedTransformer(cfg).to(DEV)
n_params = sum(p.numel() for p in model.parameters())
EOS = model.tokenizer.eos_token_id

# ── stimulus (same as experiments 01/07: gpt2 bos == eos, so P = 1) ───────────
torch.manual_seed(0)
P = 1
rand = torch.randint(0, cfg.d_vocab, (1, SEQ))
probe = torch.cat([torch.tensor([[EOS]], dtype=torch.long), rand, rand], dim=1).to(DEV)

def measure():
    """Experiment-01 measurement on the current weights (grads off, train mode off)."""
    model.eval()
    with torch.no_grad():
        loss = model(probe, return_type="loss", loss_per_token=True)[0].float().cpu()
        _, cache = model.run_with_cache(probe, names_filter=lambda n: n.endswith("hook_pattern"))
        ind = torch.zeros(N_LAYERS, N_HEADS)
        for L in range(N_LAYERS):
            patt = cache[f"blocks.{L}.attn.hook_pattern"]
            ind[L] = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1)[0].cpu()
    model.train()
    l1 = loss[P:P + SEQ - 1].mean().item()
    l2 = loss[P + SEQ - 1:P + 2 * SEQ - 1].mean().item()
    return l1, l2, ind

# ── data: TinyStories, shuffled once, tokenized lazily into a rolling buffer ──
ds = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=0)
n_stories = len(ds)
buf, story_i = [], 0

def next_batch():
    # HookedTransformer's return_type="loss" shifts targets internally, so a batch is
    # exactly N_CTX tokens (not the nanoGPT-style N_CTX+1 input/target pair)
    global story_i
    need = BATCH * N_CTX
    while len(buf) < need:
        j = story_i % n_stories
        texts = ds[j:min(j + 64, n_stories)]["text"]
        story_i += len(texts)
        for ids in model.tokenizer(texts, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
            buf.append(EOS)
    x = torch.tensor(buf[:need], dtype=torch.long).view(BATCH, N_CTX)
    del buf[:need]
    return x

def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)

# ── resume: newest checkpoint for this tag + already-measured CSV rows ────────
done = {}
if CSV_PATH.exists():
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            done[int(row["step"])] = {k: float(v) for k, v in row.items()}
score_mats = torch.load(PT_PATH) if PT_PATH.exists() else {}
start_step, ema = 0, None
ckpts_on_disk = sorted(CKPT.glob(f"{TAG}_step*.pt"),
                       key=lambda p: int(p.stem.rsplit("step", 1)[1]))
if ckpts_on_disk:
    # weights_only=False: the checkpoint holds optimizer state + config (our own file)
    state = torch.load(ckpts_on_disk[-1], map_location=DEV, weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    start_step, story_i, ema = state["step"], state["story_i"], state["ema"]
    print(f"resumed {ckpts_on_disk[-1].name} (step {start_step}, story {story_i})")

def checkpoint(step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "story_i": story_i, "ema": ema, "cfg": cfg.to_dict()},
               CKPT / f"{TAG}_step{step}.pt")
    l1, l2, ind = measure()
    top5 = ind.flatten().topk(5).values
    done[step] = {"step": step, "tokens": step * TOK_PER_STEP,
                  "train_loss": round(ema if ema is not None else float("nan"), 4),
                  "loss_copy1": l1, "loss_copy2": l2,
                  "top_score": top5[0].item(), "top5_mean": top5.mean().item()}
    score_mats[step] = ind
    torch.save(score_mats, PT_PATH)
    with open(CSV_PATH, "w", newline="") as f:   # rewrite whole CSV each time (tiny)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in sorted(done):
            w.writerow(done[s])
    print(f"  ckpt step {step:>6} ({step * TOK_PER_STEP / 1e6:7.1f}M tok): "
          f"train {done[step]['train_loss']:5.2f}   1st-copy {l1:5.2f}   "
          f"2nd-copy {l2:5.2f}   top induction {top5[0]:.2f}")

# ── train ─────────────────────────────────────────────────────────────────────
print(f"model: {TAG} ({n_params/1e6:.1f}M params, {N_LAYERS}L x {N_HEADS}H, d={D_MODEL})   "
      f"data: TinyStories ({n_stories} stories)   device: {DEV}")
print(f"steps: {start_step} -> {TOTAL_STEPS} ({TOK_PER_STEP} tok/step)   "
      f"checkpoints at {[s for s in CKPT_STEPS if s >= start_step]}")
model.train()
if start_step == 0 and 0 in CKPT_STEPS and 0 not in done:
    checkpoint(0)                                # the untrained baseline
t0, tok0 = time.time(), 0
for step in range(start_step + 1, TOTAL_STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = next_batch().to(DEV)
    with torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
        loss = model(x, return_type="loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    tok0 += TOK_PER_STEP
    if step % 50 == 0:
        print(f"step {step:>6}/{TOTAL_STEPS}   loss {ema:5.3f}   "
              f"lr {lr_at(step):.2e}   {tok0/(time.time()-t0)/1e3:5.1f}k tok/s", flush=True)
        t0, tok0 = time.time(), 0
    if step in CKPT_STEPS:
        checkpoint(step)

# ── chart 1: the phase change (same layout as experiment 07) ──────────────────
rows = [done[s] for s in sorted(done)]
x = [max(r["tokens"], 1) for r in rows]          # log axis; step 0 plotted at 1
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.plot(x, [r["loss_copy1"] for r in rows], color="#9FAAAD", lw=1.6, marker="o", ms=4,
         label="loss, 1st copy (no lookback possible)")
ax1.plot(x, [r["loss_copy2"] for r in rows], color="#0E9E76", lw=2.0, marker="o", ms=4,
         label="loss, 2nd copy (in-context learning)")
ax1.plot(x, [r["train_loss"] for r in rows], color="#5d7ba3", lw=1.2, ls=":", marker=".",
         ms=3, label="training loss (TinyStories)")
ax1.set_xscale("log")
ax1.set_xlabel("training tokens (log scale)")
ax1.set_ylabel("loss (nats)")
ax2 = ax1.twinx()
ax2.plot(x, [r["top_score"] for r in rows], color="#C0392B", lw=2.0, marker="s", ms=4,
         label="top induction score")
ax2.plot(x, [r["top5_mean"] for r in rows], color="#C0392B", lw=1.2, ls="--", marker="s",
         ms=3, alpha=0.6, label="top-5 mean induction score")
ax2.set_ylabel("induction score", color="#C0392B")
ax2.set_ylim(0, 1)
h1, l1_ = ax1.get_legend_handles_labels()
h2, l2_ = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1_ + l2_, loc="center left", fontsize=8)
ax1.set_title(f"The phase change, in a model trained HERE\n"
              f"{TAG}: {n_params/1e6:.1f}M params on TinyStories", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / f"training_27_phase_{TAG}.png", dpi=120)
print(f"→ results/training_27_phase_{TAG}.png")

# ── chart 2: heatmap filmstrip of the birth ───────────────────────────────────
steps_done = sorted(score_mats)
if len(steps_done) >= 4:
    picks = [steps_done[round(i * (len(steps_done) - 1) / 3)] for i in range(4)]
    vmax = max(score_mats[s].max().item() for s in picks)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.0), squeeze=False)
    fig.subplots_adjust(top=0.74)
    for i, s in enumerate(picks):
        ax = axes[0][i]
        im = ax.imshow(score_mats[s], cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"step {s} ({s * TOK_PER_STEP / 1e6:.0f}M tok)", fontsize=10)
        ax.set_xlabel("head")
        if i == 0:
            ax.set_ylabel("layer")
    fig.colorbar(im, ax=[axes[0][i] for i in range(len(picks))],
                 label="induction score", fraction=0.03)
    fig.suptitle(f"Birth of the induction heads — {TAG} (trained locally)", fontweight="bold")
    fig.savefig(OUT / f"training_28_filmstrip_{TAG}.png", dpi=120, bbox_inches="tight")
    print(f"→ results/training_28_filmstrip_{TAG}.png")

best = max(rows, key=lambda r: r["top_score"])
print(f"final: 2nd-copy loss {rows[0]['loss_copy2']:.2f} → {rows[-1]['loss_copy2']:.2f}, "
      f"top induction score {rows[0]['top_score']:.2f} → {rows[-1]['top_score']:.2f} "
      f"(peak {best['top_score']:.2f} at step {int(best['step'])})")
