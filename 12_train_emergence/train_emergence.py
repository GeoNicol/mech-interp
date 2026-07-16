#!/usr/bin/env python3
"""
train_emergence.py — train a tiny GPT from scratch and watch induction heads being born.

Experiment 07 replayed the phase change from Pythia's PUBLISHED checkpoints; this one
removes the training wheels. A 4-layer, ~29M-parameter HookedTransformer (untied GPT-2
vocab embeddings dominate the count; ~3M live in the blocks) is trained from
scratch on a dosable mix of TinyStories / wikitext-103 / synthetic repeated-random-token
rows (WEB_FRAC + SYNTH_FRAC knobs; the run history is a data-pressure dose-response —
see the knob comment below) on the local GPU, with log-spaced weight checkpoints and
dense probe measurement every MEASURE_EVERY steps. At every checkpoint the experiment-01 measurement
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
# Data-pressure dose-response, one knob per run. TinyStories alone (tag tiny-4L256)
# never grew induction heads — loss 1.60 with ZERO in-context copying; its ~7k-token
# world is memorizable from weights, so lookback never pays (prev-token heads DO form:
# 0.57 — the upstream half is free, the composition is not). 30% wikitext-103
# (tiny-4L256-mix30) tripled the top score to ~0.065 but stalled mid-ramp: natural-text
# pressure alone is too dilute at this scale (literature puts the transition at
# ~2.5B+ tokens; this budget is 491M). SYNTH_FRAC injects sequences of literally
# random tokens with a repeating block (variable period 16–256, the probe task
# itself) — copying is the only way to predict them, so induction pressure is
# maximal by construction while the wikitext bulk still teaches real language.
WEB_FRAC = 0.85                                # fraction of batch tokens from wikitext-103
SYNTH_FRAC = 0.15                              # fraction of batch rows: repeated random blocks
                                               # (remainder, if any, comes from TinyStories)

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEPS
TAG = (f"tiny-{N_LAYERS}L{D_MODEL}"
       + (f"-mix{round(WEB_FRAC * 100)}" if WEB_FRAC else "")
       + (f"-syn{round(SYNTH_FRAC * 100)}" if SYNTH_FRAC else "")
       + ("" if TOTAL_STEPS == DEFAULT_STEPS else f"_s{TOTAL_STEPS}"))
TOK_PER_STEP = BATCH * N_CTX

# log-spaced checkpoints: powers of two + halfway points, always incl. 0 and the end
CKPT_STEPS = sorted({0, TOTAL_STEPS,
                     *(2 ** k for k in range(5, 15)),
                     *(3 * 2 ** k for k in range(7, 14))})
CKPT_STEPS = [s for s in CKPT_STEPS if s <= TOTAL_STEPS]

MEASURE_EVERY = 256   # dense probe measurement (a forward pass — pennies); weight saves
                      # stay log-spaced at CKPT_STEPS (each is ~350 MB of model+optimizer)
N_PROBE = 16          # probe blocks averaged per measurement — a single 50-token block
                      # wobbles ±0.01–0.02 per head, the scale of the effect being read

CSV_PATH = OUT / f"training_{TAG}.csv"
PT_PATH = OUT / f"training_scores_{TAG}.pt"
PREV_PT_PATH = OUT / f"training_prev_{TAG}.pt"
FIELDS = ["step", "tokens", "train_loss", "loss_copy1", "loss_copy2",
          "top_score", "top5_mean", "prev_top", "prev_top5"]

cfg = HookedTransformerConfig(
    n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_MODEL // N_HEADS,
    d_mlp=4 * D_MODEL, n_ctx=N_CTX, act_fn="gelu", normalization_type="LN",
    tokenizer_name="gpt2", seed=0)
model = HookedTransformer(cfg).to(DEV)
n_params = sum(p.numel() for p in model.parameters())
EOS = model.tokenizer.eos_token_id

# ── stimulus (same as experiments 01/07: gpt2 bos == eos, so P = 1) ───────────
# N_PROBE independent repeated blocks, scores averaged. Same seed as always: the
# first block of the (N_PROBE, SEQ) draw is bit-identical to the original (1, SEQ)
# single-probe stimulus, so the old probe is literally row 0 of the new one.
torch.manual_seed(0)
P = 1
rand = torch.randint(0, cfg.d_vocab, (N_PROBE, SEQ))
probe = torch.cat([torch.full((N_PROBE, 1), EOS, dtype=torch.long), rand, rand], dim=1).to(DEV)

def measure():
    """Experiment-01 measurement on the current weights (grads off, train mode off).
    Returns copy-1/copy-2 loss plus per-head induction AND prev-token scores, all
    averaged over the N_PROBE blocks. The prev-token score (offset −1 diagonal, the
    experiment-03 measurement) separates 'upstream half never formed' from 'formed
    but the composition never closed'."""
    model.eval()
    with torch.no_grad():
        loss = model(probe, return_type="loss", loss_per_token=True).float().cpu()
        _, cache = model.run_with_cache(probe, names_filter=lambda n: n.endswith("hook_pattern"))
        ind = torch.zeros(N_LAYERS, N_HEADS)
        prev = torch.zeros(N_LAYERS, N_HEADS)
        for L in range(N_LAYERS):
            patt = cache[f"blocks.{L}.attn.hook_pattern"]
            ind[L] = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
            prev[L] = patt.diagonal(offset=-1, dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
    model.train()
    l1 = loss[:, P:P + SEQ - 1].mean().item()
    l2 = loss[:, P + SEQ - 1:P + 2 * SEQ - 1].mean().item()
    return l1, l2, ind, prev

# ── data: three dosable sources — TinyStories / wikitext-103 / synthetic ──────
# repeated-random-token rows. Text sources are shuffled once and tokenized lazily
# into per-source rolling buffers; synthetic rows are generated per step.
STORY_FRAC = 1.0 - WEB_FRAC - SYNTH_FRAC
assert STORY_FRAC > -1e-9, "WEB_FRAC + SYNTH_FRAC must not exceed 1"
ds_story = (load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=0)
            if STORY_FRAC > 1e-9 else None)
ds_web = (load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train").shuffle(seed=0)
          if WEB_FRAC else None)
n_stories = len(ds_story) if ds_story is not None else 0
buf_s, buf_w, story_i, web_i = [], [], 0, 0

def synth_rows(step, n):
    """n sequences of random tokens with a repeating block (variable period): the
    induction task in its purest form. Seeded by step, so resume is deterministic
    and rows never repeat across steps."""
    g = torch.Generator().manual_seed(step)
    rows = []
    for _ in range(n):
        L = int(torch.randint(16, 257, (1,), generator=g))
        block = torch.randint(0, cfg.d_vocab, (L,), generator=g)
        rows.append(block.repeat(-(-N_CTX // L))[:N_CTX])
    return torch.stack(rows)

def fill(buf, ds, i, need):
    """Top up `buf` to `need` tokens from dataset `ds` starting at row `i`."""
    n = len(ds)
    while len(buf) < need:
        j = i % n
        texts = ds[j:min(j + 64, n)]["text"]
        i += len(texts)
        texts = [t for t in texts if t.strip()]    # wikitext-raw has many blank rows
        for ids in model.tokenizer(texts, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
            buf.append(EOS)
    return i

def next_batch(step):
    # HookedTransformer's return_type="loss" shifts targets internally, so a batch is
    # exactly N_CTX tokens (not the nanoGPT-style N_CTX+1 input/target pair).
    # Synthetic sequences must be whole rows (their repetition is row-aligned);
    # the text sources share the remaining rows at their token fractions.
    global story_i, web_i
    n_syn = round(BATCH * SYNTH_FRAC)
    need = (BATCH - n_syn) * N_CTX
    need_w = min(int(BATCH * N_CTX * WEB_FRAC), need)
    if need_w:
        web_i = fill(buf_w, ds_web, web_i, need_w)
    if need - need_w:
        story_i = fill(buf_s, ds_story, story_i, need - need_w)
    x = torch.tensor(buf_w[:need_w] + buf_s[:need - need_w],
                     dtype=torch.long).view(BATCH - n_syn, N_CTX)
    del buf_w[:need_w], buf_s[:need - need_w]
    if n_syn:
        x = torch.cat([x, synth_rows(step, n_syn)])
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
            r = {k: float(v) for k, v in row.items()}
            # step/tokens are written as ints by measure_row; keep them ints here so
            # the CSV round-trips identically across repeated resumes
            r["step"], r["tokens"] = int(r["step"]), int(r["tokens"])
            done[r["step"]] = r
score_mats = torch.load(PT_PATH) if PT_PATH.exists() else {}
prev_mats = torch.load(PREV_PT_PATH) if PREV_PT_PATH.exists() else {}
start_step, ema = 0, None
ckpts_on_disk = sorted(CKPT.glob(f"{TAG}_step*.pt"),
                       key=lambda p: int(p.stem.rsplit("step", 1)[1]))
if ckpts_on_disk:
    # weights_only=False: the checkpoint holds optimizer state + config. Fine for
    # files THIS script wrote; never load untrusted .pt files this way (arbitrary
    # code execution via pickle).
    state = torch.load(ckpts_on_disk[-1], map_location=DEV, weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    start_step, story_i, ema = state["step"], state["story_i"], state["ema"]
    web_i = state.get("web_i", 0)
    print(f"resumed {ckpts_on_disk[-1].name} (step {start_step}, story {story_i}, web {web_i})")

def save_ckpt(step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "story_i": story_i, "web_i": web_i, "ema": ema,
                "cfg": cfg.to_dict()},
               CKPT / f"{TAG}_step{step}.pt")

def measure_row(step):
    l1, l2, ind, prev = measure()
    top5 = ind.flatten().topk(5).values
    ptop5 = prev.flatten().topk(5).values
    done[step] = {"step": step, "tokens": step * TOK_PER_STEP,
                  "train_loss": round(ema if ema is not None else float("nan"), 4),
                  "loss_copy1": l1, "loss_copy2": l2,
                  "top_score": top5[0].item(), "top5_mean": top5.mean().item(),
                  "prev_top": ptop5[0].item(), "prev_top5": ptop5.mean().item()}
    score_mats[step] = ind
    prev_mats[step] = prev
    torch.save(score_mats, PT_PATH)
    torch.save(prev_mats, PREV_PT_PATH)
    with open(CSV_PATH, "w", newline="") as f:   # rewrite whole CSV each time (tiny)
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in sorted(done):
            w.writerow(done[s])
    print(f"  meas step {step:>6} ({step * TOK_PER_STEP / 1e6:7.1f}M tok): "
          f"train {done[step]['train_loss']:5.2f}   1st-copy {l1:5.2f}   "
          f"2nd-copy {l2:5.2f}   top induction {top5[0]:.2f}   prev {ptop5[0]:.2f}")

# ── train ─────────────────────────────────────────────────────────────────────
parts = []
if ds_story is not None:
    parts.append(f"TinyStories ({n_stories} stories, {STORY_FRAC:.0%})")
if WEB_FRAC:
    parts.append(f"wikitext-103 ({len(ds_web)} rows, {WEB_FRAC:.0%})")
if SYNTH_FRAC:
    parts.append(f"synthetic repeated-random rows ({SYNTH_FRAC:.0%})")
data_desc = " + ".join(parts)
print(f"model: {TAG} ({n_params/1e6:.1f}M params, {N_LAYERS}L x {N_HEADS}H, d={D_MODEL})   "
      f"data: {data_desc}   device: {DEV}")
print(f"steps: {start_step} -> {TOTAL_STEPS} ({TOK_PER_STEP} tok/step)   "
      f"checkpoints at {[s for s in CKPT_STEPS if s >= start_step]}")
model.train()
if start_step == 0 and 0 not in done:
    save_ckpt(0)                                 # the untrained baseline
    measure_row(0)
t0, tok0 = time.time(), 0
for step in range(start_step + 1, TOTAL_STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = next_batch(step).to(DEV)
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
        save_ckpt(step)
    if step % MEASURE_EVERY == 0 or step in CKPT_STEPS:
        measure_row(step)

# ── chart 1: the phase change — two stacked panels, shared log-x, annotated ──
# The ~120 dense measurements draw as clean lines (no per-point markers); the
# annotations carry the story: the ignition window, the peak, the final values.
# Two panels instead of twinned y-axes: loss and attention score never share a
# scale, and the shared x-axis still lines the ignition band up across both.
rows = [done[s] for s in sorted(done) if s > 0]  # step 0 would stretch the log axis
xs = [r["tokens"] for r in rows]
sc = [r["top_score"] for r in rows]

# ignition window: last measurement below 0.1 before the first crossing of 0.5
ign = next((i for i in range(1, len(sc)) if sc[i] >= 0.5 and sc[i - 1] < 0.5), None)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.08})
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.grid(True, color="#E8E8E4", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ign is not None:
        j = ign - 1
        while j > 0 and sc[j] > 0.1:
            j -= 1
        ax.axvspan(xs[j], xs[ign], color="#F5C356", alpha=0.3, zorder=0)

ax1.plot(xs, [r["loss_copy1"] for r in rows], color="#2A78D6", lw=2,
         label="1st copy — no lookback possible (control)")
ax1.plot(xs, [r["loss_copy2"] for r in rows], color="#0E9E76", lw=2.4,
         label="2nd copy — in-context learning")
for key, col, fmt in (("loss_copy1", "#2A78D6", "{:.1f}"), ("loss_copy2", "#0E9E76", "{:.2f}")):
    ax1.annotate(fmt.format(rows[-1][key]), (xs[-1], rows[-1][key]), xytext=(5, 0),
                 textcoords="offset points", color=col, fontsize=9, va="center",
                 fontweight="bold", annotation_clip=False)
if ign is not None:
    ax1.text(xs[ign], 1.01, "induction heads ignite", transform=ax1.get_xaxis_transform(),
             ha="center", fontsize=9, color="#9C6B1F", fontweight="bold")
ax1.set_ylabel("probe loss (nats)")
ax1.legend(loc="lower left", fontsize=8.5, frameon=False)

ax2.plot(xs, sc, color="#C0392B", lw=2.4, label="top induction score")
ax2.plot(xs, [r["prev_top"] for r in rows], color="#E67E22", lw=1.6, ls="-.",
         label="top prev-token score")
ax2.axhline(THRESH, color="#B9B9B4", lw=1, ls="--")
ax2.text(xs[0], THRESH + 0.03, f"induction threshold {THRESH}", fontsize=7.5, color="#8A8A85")
peak_i = max(range(len(sc)), key=lambda i: sc[i])
ax2.plot([xs[peak_i]], [sc[peak_i]], "o", color="#C0392B", ms=6)
ax2.annotate(f"peak {sc[peak_i]:.2f}", (xs[peak_i], sc[peak_i]), xytext=(0, 8),
             textcoords="offset points", ha="center", fontsize=9, color="#C0392B")
ax2.annotate(f"{rows[-1]['prev_top']:.2f}", (xs[-1], rows[-1]["prev_top"]), xytext=(5, 0),
             textcoords="offset points", color="#E67E22", fontsize=9, va="center",
             fontweight="bold", annotation_clip=False)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("attention score")
ax2.set_xlabel("training tokens (log scale)")
ax2.legend(loc="upper left", fontsize=8.5, frameon=False)

mix_desc = " + ".join(p for p in [
    f"{STORY_FRAC:.0%} TinyStories" if ds_story is not None else "",
    f"{WEB_FRAC:.0%} wikitext-103" if WEB_FRAC else "",
    f"{SYNTH_FRAC:.0%} synthetic repeats" if SYNTH_FRAC else ""] if p)
ax1.set_title(f"The phase change, in a model trained HERE\n"
              f"{TAG}: {n_params/1e6:.1f}M params — {mix_desc}", fontweight="bold", pad=18)
fig.savefig(OUT / f"training_27_phase_{TAG}.png", dpi=120, bbox_inches="tight")
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
        ys, hs = (score_mats[s] >= THRESH).nonzero(as_tuple=True)   # dots mark heads
        ax.scatter(hs, ys, s=36, facecolors="none", edgecolors="white", lw=1.3)  # >= THRESH
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
