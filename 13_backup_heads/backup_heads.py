#!/usr/bin/env python3
"""
backup_heads.py — can training pressure CREATE backup induction heads?

Experiment 05 found that gpt2 has no backup induction heads (one Hydra round, zero
recruits), while Qwen3-1.7B does. Where does such redundancy come from? This experiment
tries to manufacture it: take experiment 12's converged tiny model — whose entire
induction circuit is two heads, L3H0+L3H1, everything else dark — and CONTINUE training
it on the same data mix while randomly zero-ablating those two heads for half the rows
of every batch (targeted dropout at hook_z, the same hook the repo ablates everywhere).
Whenever the pair is dropped, the copy loss on synthetic repeated rows reappears — so
for the first time since ignition there is live gradient pressure toward building a
SECOND induction circuit. If backup heads are born, redundancy is an adaptation to
unreliability; if half a run of maximal pressure can't recruit one, backups are
expensive and gpt2's missing safety net looks like the default, not an accident.

Both conditions are measured at every probe: the exp-01 measurement on the intact model
("clean") and with the pair fully zeroed ("abl", the exp-05 measurement inside the
broken model). The headline curve is the ablated 2nd-copy loss: it starts at ~11 nats
(the model is blind without its only circuit) and falls only if a backup grows.

Like experiment 12 this script trains (grads ON); measure() wraps torch.no_grad().

  python 13_backup_heads/backup_heads.py        # 6000 steps ≈ 98M tokens — the same
                                                # budget the original pair ignited in
  python 13_backup_heads/backup_heads.py 300    # short run (outputs tagged _s300)

Requires experiment 12's final checkpoint (12_train_emergence/checkpoints/, gitignored)
— run train_emergence.py to completion first. Resumable: reruns load the newest OWN
checkpoint for the tag and continue; CSV rows for already-measured steps are kept.
"""
import csv, sys, time, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from transformer_lens import HookedTransformer, HookedTransformerConfig

HERE = Path(__file__).parent
SRC = HERE.parent / "12_train_emergence"
OUT = HERE / "results"
CKPT = HERE / "checkpoints"
OUT.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SRC_TAG = "tiny-4L256-mix85-syn15"             # the experiment-12 run being continued
BATCH, N_CTX = 32, 512                         # must match experiment 12
DEFAULT_STEPS = 6_000                          # ≈ 98M tokens — the original ignition budget
LR, WARMUP, WD, CLIP = 3e-4, 100, 0.1, 1.0     # constant LR: the cosine already decayed
ABLATE_P = 0.5                                 # fraction of rows per batch with the pair dropped
SEQ, THRESH = 50, 0.3                          # experiment-01 measurement knobs
WEB_FRAC, SYNTH_FRAC = 0.85, 0.15              # same diet the circuit ignited on

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEPS
TAG = (f"{SRC_TAG}-abl{round(ABLATE_P * 100)}"
       + ("" if TOTAL_STEPS == DEFAULT_STEPS else f"_s{TOTAL_STEPS}"))
TOK_PER_STEP = BATCH * N_CTX

CKPT_STEPS = sorted({0, TOTAL_STEPS,
                     *(2 ** k for k in range(5, 15)),
                     *(3 * 2 ** k for k in range(7, 14))})
CKPT_STEPS = [s for s in CKPT_STEPS if s <= TOTAL_STEPS]
MEASURE_EVERY = 256
N_PROBE = 16

CSV_PATH = OUT / f"backup_{TAG}.csv"
PT_PATH = OUT / f"backup_scores_{TAG}.pt"       # ablated-condition score mats (the hunt)
CLEAN_PT_PATH = OUT / f"backup_scores_clean_{TAG}.pt"
FIELDS = ["step", "tokens", "train_loss", "l2_clean", "l2_abl",
          "pair_score", "backup_top_clean", "backup_top_abl", "prev_top"]

# ── the patient: experiment 12's converged model + its induction pair ─────────
src_ckpts = sorted(SRC.glob(f"checkpoints/{SRC_TAG}_step*.pt"),
                   key=lambda p: int(p.stem.rsplit("step", 1)[1]))
assert src_ckpts, f"no experiment-12 checkpoints for {SRC_TAG} — run train_emergence.py first"
# weights_only=False: our own files (optimizer state + config); never load untrusted
# .pt files this way (arbitrary code execution via pickle).
src_state = torch.load(src_ckpts[-1], map_location=DEV, weights_only=False)
cfg = HookedTransformerConfig.from_dict(src_state["cfg"])
model = HookedTransformer(cfg).to(DEV)
EOS = model.tokenizer.eos_token_id
NL, NH = cfg.n_layers, cfg.n_heads

src_scores = torch.load(SRC / f"results/training_scores_{SRC_TAG}.pt")
final_mat = src_scores[max(src_scores)]
TARGETS = [(L, H) for L in range(NL) for H in range(NH) if final_mat[L, H] >= THRESH]
assert TARGETS, f"{SRC_TAG} has no induction heads >= {THRESH} — nothing to ablate"
BY_LAYER = {}
for L, H in TARGETS:
    BY_LAYER.setdefault(L, []).append(H)
target_mask = torch.zeros(NL, NH, dtype=torch.bool)
for L, H in TARGETS:
    target_mask[L, H] = True

def make_kill(heads, row_mask=None):
    """Zero the given heads' outputs at hook_z — for every row (measurement) or for
    the rows where row_mask is True (train-time dropout)."""
    def kill(z, hook):                          # z: [batch, pos, head, d_head]
        if row_mask is None:
            z[:, :, heads] = 0
        else:
            for h in heads:
                z[row_mask, :, h] = 0
        return z
    return kill

KILL_ALL = [(f"blocks.{L}.attn.hook_z", make_kill(hs)) for L, hs in BY_LAYER.items()]

# ── stimulus + measurement (experiment 01, in both conditions) ────────────────
torch.manual_seed(0)
P = 1
rand = torch.randint(0, cfg.d_vocab, (N_PROBE, SEQ))
probe = torch.cat([torch.full((N_PROBE, 1), EOS, dtype=torch.long), rand, rand], dim=1).to(DEV)

def measure_cond(hooks):
    with model.hooks(fwd_hooks=hooks):
        loss = model(probe, return_type="loss", loss_per_token=True).float().cpu()
        _, cache = model.run_with_cache(probe, names_filter=lambda n: n.endswith("hook_pattern"))
    ind = torch.zeros(NL, NH)
    prev = torch.zeros(NL, NH)
    for L in range(NL):
        patt = cache[f"blocks.{L}.attn.hook_pattern"]
        ind[L] = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
        prev[L] = patt.diagonal(offset=-1, dim1=-2, dim2=-1).float().mean(-1).mean(0).cpu()
    l2 = loss[:, P + SEQ - 1:P + 2 * SEQ - 1].mean().item()
    return l2, ind, prev

def measure():
    model.eval()
    with torch.no_grad():
        l2_clean, ind_c, prev_c = measure_cond([])
        l2_abl, ind_a, _ = measure_cond(KILL_ALL)
    model.train()
    return l2_clean, l2_abl, ind_c, ind_a, prev_c

# ── data: identical three-source mix to experiment 12 ─────────────────────────
ds_web = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train").shuffle(seed=0)
buf_w, web_i = [], 0

def synth_rows(step, n):
    # seed offset 1_000_000: never repeats a row experiment 12 already trained on
    g = torch.Generator().manual_seed(1_000_000 + step)
    rows = []
    for _ in range(n):
        L = int(torch.randint(16, 257, (1,), generator=g))
        block = torch.randint(0, cfg.d_vocab, (L,), generator=g)
        rows.append(block.repeat(-(-N_CTX // L))[:N_CTX])
    return torch.stack(rows)

def fill(buf, ds, i, need):
    n = len(ds)
    while len(buf) < need:
        j = i % n
        texts = ds[j:min(j + 64, n)]["text"]
        i += len(texts)
        texts = [t for t in texts if t.strip()]
        for ids in model.tokenizer(texts, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
            buf.append(EOS)
    return i

def next_batch(step):
    global web_i
    n_syn = round(BATCH * SYNTH_FRAC)
    need = (BATCH - n_syn) * N_CTX
    web_i = fill(buf_w, ds_web, web_i, need)
    x = torch.tensor(buf_w[:need], dtype=torch.long).view(BATCH - n_syn, N_CTX)
    del buf_w[:need]
    return torch.cat([x, synth_rows(step, n_syn)])

def lr_at(step):
    return LR * min(1.0, (step + 1) / WARMUP)

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)

# ── resume: newest OWN checkpoint, else start from experiment 12's final ──────
done = {}
if CSV_PATH.exists():
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            r = {k: float(v) for k, v in row.items()}
            r["step"], r["tokens"] = int(r["step"]), int(r["tokens"])
            done[r["step"]] = r
abl_mats = torch.load(PT_PATH) if PT_PATH.exists() else {}
clean_mats = torch.load(CLEAN_PT_PATH) if CLEAN_PT_PATH.exists() else {}
start_step, ema = 0, None
own_ckpts = sorted(CKPT.glob(f"{TAG}_step*.pt"),
                   key=lambda p: int(p.stem.rsplit("step", 1)[1]))
if own_ckpts:
    state = torch.load(own_ckpts[-1], map_location=DEV, weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    start_step, web_i, ema = state["step"], state["web_i"], state["ema"]
    print(f"resumed {own_ckpts[-1].name} (step {start_step}, web {web_i})")
else:
    model.load_state_dict(src_state["model"])
    web_i = src_state.get("web_i", 0)           # keep eating wikitext where 12 stopped
    print(f"continuing from {src_ckpts[-1].name} (web {web_i})")

def save_ckpt(step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "web_i": web_i, "ema": ema, "cfg": cfg.to_dict()},
               CKPT / f"{TAG}_step{step}.pt")

def measure_row(step):
    l2_clean, l2_abl, ind_c, ind_a, prev_c = measure()
    others_c = ind_c.masked_fill(target_mask, 0)
    others_a = ind_a.masked_fill(target_mask, 0)
    done[step] = {"step": step, "tokens": step * TOK_PER_STEP,
                  "train_loss": round(ema if ema is not None else float("nan"), 4),
                  "l2_clean": l2_clean, "l2_abl": l2_abl,
                  "pair_score": ind_c[target_mask].mean().item(),
                  "backup_top_clean": others_c.max().item(),
                  "backup_top_abl": others_a.max().item(),
                  "prev_top": prev_c.flatten().max().item()}
    abl_mats[step] = ind_a
    clean_mats[step] = ind_c
    torch.save(abl_mats, PT_PATH)
    torch.save(clean_mats, CLEAN_PT_PATH)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in sorted(done):
            w.writerow(done[s])
    print(f"  meas step {step:>5} ({step * TOK_PER_STEP / 1e6:6.1f}M tok): "
          f"train {done[step]['train_loss']:5.2f}   2nd-copy clean {l2_clean:5.2f} / "
          f"abl {l2_abl:5.2f}   pair {done[step]['pair_score']:.2f}   "
          f"backup {done[step]['backup_top_abl']:.2f}")

# ── train: same objective, but the pair flickers ──────────────────────────────
n_params = sum(p.numel() for p in model.parameters())
print(f"model: {TAG} ({n_params/1e6:.1f}M params, {NL}L x {NH}H)   "
      f"targets: {['L%dH%d' % t for t in TARGETS]} dropped on {ABLATE_P:.0%} of rows   "
      f"device: {DEV}")
print(f"steps: {start_step} -> {TOTAL_STEPS} ({TOK_PER_STEP} tok/step)   "
      f"checkpoints at {[s for s in CKPT_STEPS if s >= start_step]}")
model.train()
if start_step == 0 and 0 not in done:
    save_ckpt(0)                                 # the intact experiment-12 endpoint
    measure_row(0)
t0, tok0 = time.time(), 0
for step in range(start_step + 1, TOTAL_STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = next_batch(step).to(DEV)
    g = torch.Generator().manual_seed(2_000_000 + step)   # resume-deterministic dropout
    row_mask = (torch.rand(BATCH, generator=g) < ABLATE_P).to(DEV)
    hooks = [(f"blocks.{L}.attn.hook_z", make_kill(hs, row_mask))
             for L, hs in BY_LAYER.items()]
    with model.hooks(fwd_hooks=hooks):
        with torch.autocast(DEV, torch.bfloat16, enabled=DEV == "cuda"):
            loss = model(x, return_type="loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    tok0 += TOK_PER_STEP
    if step % 50 == 0:
        print(f"step {step:>5}/{TOTAL_STEPS}   loss {ema:5.3f}   "
              f"{tok0/(time.time()-t0)/1e3:5.1f}k tok/s", flush=True)
        t0, tok0 = time.time(), 0
    if step in CKPT_STEPS:
        save_ckpt(step)
    if step % MEASURE_EVERY == 0 or step in CKPT_STEPS:
        measure_row(step)

# ── chart 1: does the broken model heal? (two panels, shared linear-x) ────────
rows = [done[s] for s in sorted(done)]
xs = [r["tokens"] / 1e6 for r in rows]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.08})
for ax in (ax1, ax2):
    ax.grid(True, color="#E8E8E4", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.plot(xs, [r["l2_clean"] for r in rows], color="#2A78D6", lw=2,
         label="2nd-copy loss, pair intact")
ax1.plot(xs, [r["l2_abl"] for r in rows], color="#0E9E76", lw=2.4,
         label="2nd-copy loss, pair zero-ablated")
for key, col in (("l2_clean", "#2A78D6"), ("l2_abl", "#0E9E76")):
    ax1.annotate(f"{rows[-1][key]:.2f}", (xs[-1], rows[-1][key]), xytext=(5, 0),
                 textcoords="offset points", color=col, fontsize=9, va="center",
                 fontweight="bold", annotation_clip=False)
ax1.set_ylabel("probe loss (nats)")
ax1.legend(loc="center right", fontsize=8.5, frameon=False)

ax2.plot(xs, [r["backup_top_abl"] for r in rows], color="#C0392B", lw=2.4,
         label="top backup induction score (pair ablated)")
ax2.plot(xs, [r["pair_score"] for r in rows], color="#E67E22", lw=1.6, ls="-.",
         label="original pair, mean score (intact)")
ax2.axhline(THRESH, color="#B9B9B4", lw=1, ls="--")
ax2.text(xs[0], THRESH + 0.03, f"induction threshold {THRESH}", fontsize=7.5, color="#8A8A85")
cross = next((i for i, r in enumerate(rows) if r["backup_top_abl"] >= THRESH), None)
if cross is not None:
    ax2.axvline(xs[cross], color="#C0392B", lw=1, ls=":")
    ax2.annotate(f"backup crosses threshold\n({rows[cross]['tokens']/1e6:.0f}M tok)",
                 (xs[cross], THRESH), xytext=(8, 40), textcoords="offset points",
                 fontsize=8.5, color="#C0392B")
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("attention score")
ax2.set_xlabel("additional training tokens (M)")
ax2.legend(loc="center left", fontsize=8.5, frameon=False)
ax1.set_title(f"Can pressure create redundancy?\n"
              f"{TAG}: {'+'.join('L%dH%d' % t for t in TARGETS)} dropped on "
              f"{ABLATE_P:.0%} of training rows", fontweight="bold", pad=12)
fig.savefig(OUT / f"backup_29_recovery_{TAG}.png", dpi=120, bbox_inches="tight")
print(f"→ results/backup_29_recovery_{TAG}.png")

# ── chart 2: filmstrip of the hunt (ablated-condition scores, pair greyed) ────
steps_done = sorted(abl_mats)
if len(steps_done) >= 4:
    picks = [steps_done[round(i * (len(steps_done) - 1) / 3)] for i in range(4)]
    vmax = max(max(abl_mats[s].masked_fill(target_mask, 0).max().item() for s in picks), THRESH)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#B9B9B4")                     # the ablated pair, greyed out
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.0), squeeze=False)
    fig.subplots_adjust(top=0.74)
    for i, s in enumerate(picks):
        ax = axes[0][i]
        m = abl_mats[s].masked_fill(target_mask, float("nan"))
        im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        ys, hs = ((abl_mats[s] >= THRESH) & ~target_mask).nonzero(as_tuple=True)
        ax.scatter(hs, ys, s=36, facecolors="none", edgecolors="white", lw=1.3)
        ax.set_title(f"+{s * TOK_PER_STEP / 1e6:.0f}M tok", fontsize=10)
        ax.set_xlabel("head")
        if i == 0:
            ax.set_ylabel("layer")
    fig.colorbar(im, ax=[axes[0][i] for i in range(len(picks))],
                 label="induction score (pair ablated)", fraction=0.03)
    fig.suptitle(f"Hunting for a backup circuit — {TAG} (grey = the ablated pair)",
                 fontweight="bold")
    fig.savefig(OUT / f"backup_30_filmstrip_{TAG}.png", dpi=120, bbox_inches="tight")
    print(f"→ results/backup_30_filmstrip_{TAG}.png")

first, last = rows[0], rows[-1]
print(f"final: ablated 2nd-copy loss {first['l2_abl']:.2f} → {last['l2_abl']:.2f}, "
      f"top backup score {first['backup_top_abl']:.2f} → {last['backup_top_abl']:.2f}, "
      f"pair {first['pair_score']:.2f} → {last['pair_score']:.2f}")
