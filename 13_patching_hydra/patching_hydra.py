#!/usr/bin/env python3
"""
patching_hydra.py — does the Hydra keep spawning when you ablate it *faithfully*?

Experiment 05 hunted backup induction heads by ITERATIVE ZERO-ABLATION: kill the induction
heads, re-score every survivor inside the broken model, recruit whatever crosses THRESH,
repeat. Only Qwen3-1.7B recruited backups (round 2: L21H9, L8H5, L18H4; 2.9x -> 4.5x).

But zero-ablation is off-distribution — it shoves all-zeros into the residual stream, and
LayerNorm then renormalises what's left, which can AMPLIFY surviving heads and manufacture
apparent "self-repair" that the model never actually does on real inputs. Experiment 06
already showed the SINGLE-round damage is the same under mean-ablation (3.0x ~ 2.9x), so the
first-round redundancy is genuine. This experiment asks the harder, iterative question:

    is the RECRUITMENT of backup heads — the Hydra growing a new head each round — a real
    computational property, or an artifact of how we removed the primary heads?

To find out we re-run the exact exp-05 hunt under three interventions and compare:

  zero   : head output := 0                       (exp-05; off-distribution)
  mean   : head output := its mean over a ref batch  (exp-06; in-distribution constant)
  patch  : head output := its activation on a matched CORRUPT run  (resample ablation)

The corrupt run is  [BOS] r0..r49 s0..s49  — same first copy as the stimulus, but a FRESH
random second half, so there is nothing to copy and each induction head emits its genuine
"no valid target" output. Because attention is causal and the first halves are byte-identical,
patching is a literal no-op on the first copy and surgically alters ONLY the induction region:
the cleanest possible null. This is the gold-standard causal intervention (causal-scrubbing /
ACDC lineage) — the head is replaced by a real activation it actually produces, never an
off-distribution constant.

Read-out:
  same recruits under all three  -> the Hydra is real; self-repair survives faithful ablation
  recruits vanish under patch    -> the backups were a zero/mean-ablation artifact

  python 13_patching_hydra/patching_hydra.py Qwen/Qwen3-1.7B   # the one with the Hydra
  python 13_patching_hydra/patching_hydra.py                   # gpt2 (default; should stay flat)
  python 13_patching_hydra/patching_hydra.py Qwen/Qwen3-1.7B 0.2   # 2nd arg overrides THRESH

Outputs land in results/ next to this script.
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
MAX_ROUNDS = 10   # safety stop; real runs converge much sooner
REF_BATCH = 16    # reference sequences used to estimate each head's mean output (for mean-ablation)
METHODS = ["zero", "mean", "patch"]
COLORS = {"zero": "#C0392B", "mean": "#E8A33D", "patch": "#2E86C1"}

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
if THRESH != 0.3:
    TAG += f"_t{int(THRESH * 100)}"   # non-default threshold gets its own output files
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01–06) ─────────────────────────────────────────────────────
try:
    model = HookedTransformer.from_pretrained(MODEL, device=DEV, dtype=DTYPE)
except Exception:
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(MODEL, device=DEV, dtype=DTYPE)
    model.enable_compatibility_mode()
nL, nH = model.cfg.n_layers, model.cfg.n_heads

torch.manual_seed(0)
bos = model.tokenizer.bos_token_id
if bos is None:
    bos = model.tokenizer.eos_token_id
prefix = [[bos]] if bos is not None else [[]]
P = len(prefix[0])
prefix_t = torch.tensor(prefix, dtype=torch.long)
rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))                 # the first copy r0..r49
tokens = torch.cat([prefix_t, rand, rand], dim=1).to(DEV)           # stimulus: [BOS] r r
reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)

# The matched corrupt run for patch/resample ablation: same first copy, fresh random second
# half. Nothing to copy in the 2nd half -> induction heads emit their genuine "no target"
# output there; first-half activations are byte-identical to the stimulus (causal attention),
# so patching only ever changes the induction region.
g_corrupt = torch.Generator().manual_seed(1)
s_half = torch.randint(0, model.cfg.d_vocab, (1, SEQ), generator=g_corrupt)
corrupt_tokens = torch.cat([prefix_t, rand, s_half], dim=1).to(DEV)  # [BOS] r s

def stripe_score(patt):
    return patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1)[0].cpu()

def make_seq(seed):   # fresh repeated-random sequence for the mean-ablation reference batch
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, model.cfg.d_vocab, (1, SEQ), generator=g)
    return torch.cat([prefix_t, r, r], dim=1).to(DEV)

# ── clean baseline: scores + which layers actually expose attention patterns ──
_, cache = model.run_with_cache(tokens)
attn_layers = [L for L in range(nL) if f"blocks.{L}.attn.hook_pattern" in cache.cache_dict]
clean_scores = torch.zeros(nL, nH)
for L in attn_layers:
    clean_scores[L] = stripe_score(cache[f"blocks.{L}.attn.hook_pattern"])
del cache

# ── replacement activations for the two non-zero interventions ────────────────
# patch: the corrupt run's per-head z, cached once (a fixed reference forward pass) ---------
_, corrupt_cache = model.run_with_cache(corrupt_tokens)
corrupt_z = {L: corrupt_cache[f"blocks.{L}.attn.hook_z"].detach() for L in attn_layers}
del corrupt_cache

# mean: each head's average z over REF_BATCH fresh repeated-random sequences ----------------
# Any head can be recruited, so we estimate the mean for every head in every attention layer.
sums = {L: torch.zeros(nH, model.cfg.d_head) for L in attn_layers}
count = 0
def accumulate(z, hook):   # z: [1, pos, head, d_head]
    sums[hook.layer()] += z[0].float().sum(0).cpu()
    return z
acc_hooks = [(f"blocks.{L}.attn.hook_z", accumulate) for L in attn_layers]
for i in range(REF_BATCH):
    seq = make_seq(1000 + i)
    model.run_with_hooks(seq, return_type=None, fwd_hooks=acc_hooks)
    count += seq.shape[1]
mean_z = {L: (sums[L] / count).to(DEV, DTYPE) for L in attn_layers}   # [nH, d_head] per layer

# ── one intervened forward pass: returns 2nd-copy loss + every head's re-scored induction ──
# `dead` heads are edited according to `method`; the induction score of EVERY head is
# re-measured inside the resulting model — that re-scoring is what exposes backups.
def intervened_pass(dead, method):
    by_layer = defaultdict(list)
    for L, H in dead:
        by_layer[L].append(H)
    def edit_z(z, hook):
        L = hook.layer()
        for h in by_layer[L]:
            if method == "zero":
                z[:, :, h, :] = 0.0
            elif method == "mean":
                z[:, :, h, :] = mean_z[L][h]
            else:  # patch / resample
                z[:, :, h, :] = corrupt_z[L][:, :, h, :]
        return z
    captured = {}
    def capture(patt, hook):
        captured[hook.layer()] = patt.detach()
    hooks = [(f"blocks.{L}.attn.hook_z", edit_z) for L in by_layer] + \
            [(f"blocks.{L}.attn.hook_pattern", capture) for L in attn_layers]
    loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True,
                                fwd_hooks=hooks)[0].float().cpu()
    scores = torch.zeros(nL, nH)
    for L in attn_layers:
        scores[L] = stripe_score(captured[L])
    return loss[reg].mean().item(), scores

clean_loss, _ = intervened_pass([], "zero")   # no dead heads = clean forward pass
init_heads = [(L, H) for L in range(nL) for H in range(nH) if clean_scores[L, H] >= THRESH]
print(f"model: {MODEL}  ({nL} layers x {nH} heads, {len(attn_layers)} attention layers)")
print(f"clean 2nd-copy loss {clean_loss:.3f};  {len(init_heads)} induction head(s) >= {THRESH}: "
      + ", ".join(f"L{L}H{H}" for L, H in init_heads))

# ── the hunt, run once per intervention from the identical starting point ─────
# Round 1 always ablates the SAME initial induction heads (scored on the clean model); rounds
# 2+ diverge only because the intervention differs — that divergence is the whole experiment.
def hydra_hunt(method):
    dead, scores = set(), clean_scores.clone()
    losses, recruits_per_round, snapshots = [clean_loss], [], []
    for r in range(1, MAX_ROUNDS + 1):
        new = [(L, H) for L in range(nL) for H in range(nH)
               if scores[L, H] >= THRESH and (L, H) not in dead]
        snapshots.append((scores.clone(), set(dead), list(new)))   # scores that produced `new`
        if not new:
            break
        new.sort(key=lambda p: -scores[p])
        dead |= set(new)
        loss, scores = intervened_pass(sorted(dead), method)
        losses.append(loss)
        recruits_per_round.append(new)
    return {"losses": losses, "recruits": recruits_per_round, "dead": dead, "snapshots": snapshots}

results = {}
for method in METHODS:
    R = hydra_hunt(method)
    results[method] = R
    tail = " ".join(f"r{i+2}:+{len(rc)}" for i, rc in enumerate(R["recruits"][1:])) or "none"
    print(f"\n[{method:>5}] rounds={len(R['recruits'])}  dead={len(R['dead'])}  "
          f"final {R['losses'][-1]:.3f} ({R['losses'][-1]/clean_loss:.1f}x)  backups after r1: {tail}")
    for i, rc in enumerate(R["recruits"][1:], start=2):
        print(f"          round {i} recruited: " + ", ".join(f"L{L}H{H}" for L, H in rc))

print("\nverdict:")
for method in METHODS:
    n_backup = sum(len(rc) for rc in results[method]["recruits"][1:])
    print(f"  {method:>5}: {n_backup} backup head(s) recruited beyond round 1")
print("  -> if 'patch' recruits ~as many backups as 'zero', the Hydra survives faithful "
      "ablation (real self-repair); if it recruits far fewer, the backups were a zeroing artifact.")

# ── chart 1: loss trajectory per round, the three interventions overlaid ──────
# Round 1 removes the SAME primary heads for every method (one shared label); rounds 2+ are
# the backup recruitment — the actual story — so those get per-method, vertically staggered
# labels ("+k" backups) that don't collide when two methods land on the same loss.
# Backup-count labels sit in three fixed vertical bands so they never collide, even when two
# methods land on the same loss (they nearly coincide at low thresholds).
YOFF = {"zero": (10, 12), "mean": (10, -2), "patch": (10, -18)}
fig, ax = plt.subplots(figsize=(8, 5))
for method in METHODS:
    L = results[method]["losses"]
    ax.plot(range(len(L)), L, "-o", color=COLORS[method], lw=2, ms=7, label=method, zorder=3)
    for i in range(2, len(L)):   # rounds >= 2 = backups recruited that round
        k = len(results[method]["recruits"][i - 1])
        ax.annotate(f"+{k}", (i, L[i]), textcoords="offset points", xytext=YOFF[method],
                    fontsize=9, fontweight="bold", color=COLORS[method])
if len(results["zero"]["losses"]) > 1:   # one shared label for the identical round-1 removal
    ax.annotate(f"−{len(init_heads)} primary heads", (1, results["zero"]["losses"][1]),
                textcoords="offset points", xytext=(8, -16), fontsize=8, color="#54534C")
ax.axhline(clean_loss, color="#0E9E76", ls="--", lw=1.2, zorder=1)
ax.text(0.02, clean_loss, " clean", color="#0E9E76", va="bottom", fontsize=9,
        transform=ax.get_yaxis_transform())
ax.set_yscale("log")
ax.set_xlabel("ablation round  (round 1 = the original induction heads)")
ax.set_ylabel("2nd-copy loss (nats, log scale)")
ax.set_title(f"Does the Hydra keep spawning under faithful ablation?\n{MODEL}", fontweight="bold")
ax.legend(title="head removed via", frameon=False)
ax.margins(x=0.08)
fig.tight_layout(); fig.savefig(OUT / f"patch_29_trajectory_{TAG}.png", dpi=120)
print(f"\n→ results/patch_29_trajectory_{TAG}.png")

# ── chart 2: the money panel — re-scored heatmap right after round-1 ablation ─
# For each method: induction score of every head AFTER the primary heads were removed. Grey =
# removed; ● = recruited from this heatmap (round 2). Backups appearing here (and only under
# some methods) is the Hydra effect being method-dependent, made visible.
def rescored_panel(R):
    return R["snapshots"][1] if len(R["snapshots"]) > 1 else R["snapshots"][0]
panels = {m: rescored_panel(results[m]) for m in METHODS}
vmax = max(s.max().item() for s, _, _ in panels.values())
cmap = plt.cm.viridis.copy(); cmap.set_bad("#D9D9D9")
fig, axes = plt.subplots(1, len(METHODS), figsize=(4.6 * len(METHODS), 5.8), squeeze=False)
fig.subplots_adjust(top=0.82)
for ax, method in zip(axes[0], METHODS):
    s, dead_then, recruits = panels[method]
    disp = s.clone()
    for L, H in dead_then:
        disp[L, H] = float("nan")
    im = ax.imshow(disp, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for L, H in recruits:
        ax.text(H, L, "●", ha="center", va="center", color="white", fontsize=8)
    n_back = sum(len(rc) for rc in results[method]["recruits"][1:])
    ax.set_title(f"{method}-ablation\n{len(dead_then)} removed, {n_back} backup(s) recruited", fontsize=10)
    ax.set_xlabel("head")
axes[0][0].set_ylabel("layer")
fig.colorbar(im, ax=list(axes[0]), label="induction score (re-scored in broken model)", fraction=0.02)
fig.suptitle(f"Backup induction heads after removing the primary circuit ({MODEL})\n"
             "grey = removed, ● = recruited next round", fontweight="bold")
fig.savefig(OUT / f"patch_30_backups_{TAG}.png", dpi=120, bbox_inches="tight")
print(f"→ results/patch_30_backups_{TAG}.png")
