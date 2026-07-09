#!/usr/bin/env python3
"""
iterative_ablation.py — hunt the Hydra: how many rounds of ablation does the circuit survive?

Experiment 01 found that ablating Qwen3-1.7B's induction heads only hurts 2.9x — far less
than gpt2's 36.5x. The suspicion (the "Hydra effect" / self-repair): zero-ablation measures
the TOTAL effect after the network re-routes through backup heads, not the circuit's real
contribution. If backups exist, they should be findable:

  round 0 : score all heads clean, flag induction heads, ablate them
  round r : re-score every remaining head INSIDE the ablated model — backup heads that
            took over now show high induction scores — flag them, add to the ablation
            set, re-measure. Repeat until no head clears the threshold.

The trajectory of 2nd-copy loss per round, and the number of rounds/heads needed to kill
the cliff for good, quantify the redundancy that a single ablation hides.

  python 05_iterative_ablation/iterative_ablation.py Qwen/Qwen3-1.7B   # the interesting one
  python 05_iterative_ablation/iterative_ablation.py                   # gpt2 (default)

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
# Induction-score threshold (default matches 01). "Hydra is out of heads" always means
# "out of heads ABOVE this bar" — lower it (2nd CLI arg) to chase the sub-threshold tail
# where diffuse compensation lives.
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
MAX_ROUNDS = 10   # safety stop; real runs converge much sooner

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
if THRESH != 0.3:
    TAG += f"_t{int(THRESH * 100)}"   # non-default threshold gets its own output files
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01–04) ─────────────────────────────────────────────────────
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
rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))
tokens = torch.cat([torch.tensor(prefix, dtype=torch.long), rand, rand], dim=1).to(DEV)
reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)

def stripe_score(patt):
    return patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1).float().mean(-1)[0].cpu()

# One ablated forward pass that returns BOTH the 2nd-copy loss and every remaining head's
# induction score measured inside the broken model — the re-scoring is what exposes
# backup heads that only activate once the primary circuit is gone.
def ablated_pass(dead):
    by_layer = defaultdict(list)
    for L, H in dead:
        by_layer[L].append(H)
    def zero_heads(z, hook):
        for h in by_layer[hook.layer()]:
            z[:, :, h, :] = 0.0
        return z
    captured = {}
    def capture(patt, hook):
        captured[hook.layer()] = patt.detach()
    hooks = [(f"blocks.{L}.attn.hook_z", zero_heads) for L in by_layer] + \
            [(f"blocks.{L}.attn.hook_pattern", capture) for L in attn_layers]
    loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True,
                                fwd_hooks=hooks)[0].float().cpu()
    scores = torch.zeros(nL, nH)
    for L in attn_layers:
        scores[L] = stripe_score(captured[L])
    return loss[reg].mean().item(), scores

# ── round 0: clean baseline ───────────────────────────────────────────────────
_, cache = model.run_with_cache(tokens)
attn_layers = [L for L in range(nL) if f"blocks.{L}.attn.hook_pattern" in cache.cache_dict]
scores = torch.zeros(nL, nH)
for L in attn_layers:
    scores[L] = stripe_score(cache[f"blocks.{L}.attn.hook_pattern"])
del cache

clean_loss, _ = ablated_pass([])   # no heads dead = clean forward pass
print(f"model: {MODEL}  ({nL} layers x {nH} heads, {len(attn_layers)} attention layers)")
print(f"round 0 (clean): 2nd-copy loss {clean_loss:.3f}")

# ── the hunt: ablate, re-score, recruit, repeat ───────────────────────────────
dead = set()
losses, recruits_per_round = [clean_loss], []
snapshots = []   # one (scores, dead-at-scoring-time, recruits-found) per scoring pass, for the heatmaps
for r in range(1, MAX_ROUNDS + 1):
    new = [(L, H) for L in range(nL) for H in range(nH)
           if scores[L, H] >= THRESH and (L, H) not in dead]
    snapshots.append((scores.clone(), set(dead), list(new)))
    if not new:
        print(f"round {r}: no remaining head scores >= {THRESH} "
              f"(max remaining: {max((scores[L,H].item() for L in attn_layers for H in range(nH) if (L,H) not in dead), default=0):.2f}) — Hydra is out of heads.")
        break
    new.sort(key=lambda p: -scores[p])
    dead |= set(new)
    loss, scores = ablated_pass(sorted(dead))   # re-scored inside the newly broken model
    losses.append(loss)
    recruits_per_round.append(new)
    print(f"round {r}: recruited {len(new)} head(s): " +
          ", ".join(f"L{L}H{H}" for L, H in new[:12]) + ("..." if len(new) > 12 else "") +
          f"   -> cumulative {len(dead)} dead, 2nd-copy loss {loss:.3f} ({loss/clean_loss:.1f}x)")

print(f"\nsummary: single-round ablation gives {losses[1]/clean_loss:.1f}x; "
      f"full iterative ablation gives {losses[-1]/clean_loss:.1f}x "
      f"after {len(recruits_per_round)} round(s) / {len(dead)} heads — "
      f"the gap between those numbers IS the self-repair.")

# ── chart: the Hydra dies in rounds ───────────────────────────────────────────
# Loss per round (log scale — the interesting jumps span orders of magnitude), each bar
# annotated with how many heads were newly recruited that round.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = range(len(losses))
bars = ax.bar(xs, losses, color=["#0E9E76"] + ["#C0392B"] * (len(losses) - 1), width=0.6)
for i, b in enumerate(bars[1:], start=1):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.05,
            f"+{len(recruits_per_round[i-1])} heads", ha="center", fontsize=9, color="#54534C")
ax.set_yscale("log")
ax.set_xticks(list(xs))
ax.set_xticklabels(["clean"] + [f"round {i}" for i in range(1, len(losses))])
ax.set_ylabel("2nd-copy loss (nats, log scale)")
ax.set_title(f"Hydra hunt ({MODEL}): iterative ablation until no induction head remains", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / f"hydra_9_rounds_{TAG}.png", dpi=120); print(f"→ results/hydra_9_rounds_{TAG}.png")

# Chart 2 — the re-scoring heatmaps: one panel per scoring pass. Ablated heads are greyed
# out; white dots mark the heads recruited FROM that panel's scores. The Hydra effect is
# panel 2: heads that were dark in panel 1 light up once the primary circuit is dead.
cmap = plt.cm.viridis.copy()
cmap.set_bad("#D9D9D9")   # dead heads render grey
vmax = max(s.max().item() for s, _, _ in snapshots)
n = len(snapshots)
fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2), squeeze=False)
for i, (s, dead_then, recruits) in enumerate(snapshots):
    disp = s.clone()
    for L, H in dead_then:
        disp[L, H] = float("nan")
    ax = axes[0][i]
    im = ax.imshow(disp, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for L, H in recruits:
        ax.text(H, L, "●", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("round 0: clean scores" if i == 0 else f"re-scored after round {i} ablation\n({len(dead_then)} heads dead)",
                 fontsize=10)
    ax.set_xlabel("head")
    if i == 0:
        ax.set_ylabel("layer")
fig.colorbar(im, ax=[axes[0][i] for i in range(n)], label="induction score", fraction=0.02)
fig.suptitle(f"Hydra heatmaps ({MODEL}): grey = ablated, ● = recruited from this panel", fontweight="bold")
fig.savefig(OUT / f"hydra_10_heatmaps_{TAG}.png", dpi=120, bbox_inches="tight"); print(f"→ results/hydra_10_heatmaps_{TAG}.png")
