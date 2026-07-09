#!/usr/bin/env python3
"""
prev_token_heads.py — break the induction circuit from the OTHER end (head composition).

Induction heads can't work alone. The full circuit has two parts:
  1. a PREVIOUS-TOKEN head (early layer) writes "the token before me was X" into each
     position's residual stream;
  2. the INDUCTION head (later layer) queries for "positions whose previous token matches
     my current token" and copies what follows — reading what head 1 wrote (K-composition).

Experiments 01/02 ablated part 2. This one ablates part 1 and checks BOTH effects:
  behavioural : does 2nd-copy loss rise?
  mechanistic : do the induction heads' own attention stripes collapse? (the smoking gun —
                we never touched the induction heads, yet their attention pattern breaks
                because the signal they query for is gone)

  python 03_prev_token_heads/prev_token_heads.py                  # gpt2-small (default)
  python 03_prev_token_heads/prev_token_heads.py Qwen/Qwen3.5-2B

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
SEQ = 50           # length of the random block that gets repeated
THRESH = 0.3       # induction-score threshold (same as 01/02)
# Previous-token score threshold — stricter than THRESH since mild attention to the
# previous token is common. Overridable as a 2nd CLI arg to probe dose-response: if the
# signal is spread across many weak heads, lowering the threshold should deepen the break.
PREV_THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
if PREV_THRESH != 0.5:
    TAG += f"_p{int(PREV_THRESH * 100)}"   # non-default threshold gets its own output files
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01/02) ─────────────────────────────────────────────────────
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

def per_pos_loss(toks, fwd_hooks=()):
    return model.run_with_hooks(toks, return_type="loss", loss_per_token=True,
                                fwd_hooks=list(fwd_hooks))[0].float().cpu()

# ── measure: score every head on BOTH diagonals ───────────────────────────────
# Induction score: attention on the offset -(SEQ-1) stripe (as in 01).
# Previous-token score: attention on the offset -1 stripe — a head that always looks at
# the immediately preceding token scores ~1 there.
loss_clean = per_pos_loss(tokens)
_, cache = model.run_with_cache(tokens)

def stripe_score(patt, offset):
    return patt.diagonal(offset=offset, dim1=-2, dim2=-1).float().mean(-1)[0].cpu()

induction, prevtok = torch.zeros(nL, nH), torch.zeros(nL, nH)
attn_layers = []
for L in range(nL):
    key = f"blocks.{L}.attn.hook_pattern"
    if key not in cache.cache_dict:
        continue
    attn_layers.append(L)
    induction[L] = stripe_score(cache[key], -(SEQ - 1))
    prevtok[L] = stripe_score(cache[key], -1)
del cache

ind_heads = [(L, H) for L in range(nL) for H in range(nH) if induction[L, H] >= THRESH]
# A head can score on both diagonals; exclude induction heads from the prev-token list so
# the ablation only ever touches the UPSTREAM half of the circuit — otherwise the loss
# effect would trivially include direct damage to the induction heads themselves.
prev_heads = [(L, H) for L in range(nL) for H in range(nH)
              if prevtok[L, H] >= PREV_THRESH and (L, H) not in set(ind_heads)]

print(f"model: {MODEL}  ({nL} layers x {nH} heads, {len(attn_layers)} attention layers)")
print(f"induction heads (score>={THRESH}): {len(ind_heads)}")
print(f"prev-token heads (score>={PREV_THRESH}, excl. induction heads): " +
      ", ".join(f"L{L}H{H}={prevtok[L, H]:.2f}" for L, H in prev_heads))
if not prev_heads:
    print("no dedicated previous-token heads found above threshold — nothing to ablate.")
    sys.exit(0)

# ── break upstream: ablate prev-token heads, watch the induction heads ────────
# Zero the prev-token heads' z (same surgical cut as 01) while simultaneously CAPTURING
# the attention patterns in the induction heads' layers during the same ablated forward
# pass — so we can re-score the induction stripes in the broken model.
by_layer = defaultdict(list)
for L, H in prev_heads:
    by_layer[L].append(H)
def zero_prev(z, hook):
    for h in by_layer[hook.layer()]:
        z[:, :, h, :] = 0.0
    return z

captured = {}
def capture(patt, hook):
    captured[hook.layer()] = patt.detach()

ind_layers = sorted({L for L, _ in ind_heads})
hooks = [(f"blocks.{L}.attn.hook_z", zero_prev) for L in by_layer] + \
        [(f"blocks.{L}.attn.hook_pattern", capture) for L in ind_layers]
loss_abl = per_pos_loss(tokens, hooks)

# Re-score the induction stripe of each induction head from the captured (ablated) patterns.
ind_scores_clean = torch.tensor([induction[L, H] for L, H in ind_heads])
ind_scores_abl = torch.tensor([stripe_score(captured[L], -(SEQ - 1))[H] for L, H in ind_heads])

reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)
ind_clean, ind_abl = loss_clean[reg].mean().item(), loss_abl[reg].mean().item()

print(f"2nd-copy loss           clean: {ind_clean:.3f}   prev-token heads ablated: {ind_abl:.3f}"
      f"   -> {ind_abl/ind_clean:.1f}x")
print(f"mean induction score    clean: {ind_scores_clean.mean():.3f}   prev-token heads ablated: "
      f"{ind_scores_abl.mean():.3f}   -> the induction heads' attention "
      f"{'collapsed' if ind_scores_abl.mean() < 0.5 * ind_scores_clean.mean() else 'largely survived'}"
      f" (we never touched them)")

# ── chart ─────────────────────────────────────────────────────────────────────
# Chart 1 — composition: each induction head's stripe score, clean vs upstream-ablated.
# If bars collapse, the induction heads demonstrably DEPEND on the prev-token heads.
labels = [f"L{L}H{H}" for L, H in ind_heads]
xpos = torch.arange(len(ind_heads))
fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(ind_heads) + 2), 4.5))
ax.bar(xpos - 0.2, ind_scores_clean, width=0.4, color="#0E9E76", label="clean")
ax.bar(xpos + 0.2, ind_scores_abl, width=0.4, color="#C0392B", label="prev-token heads ablated")
ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("induction score")
ax.set_title(f"Composition ({MODEL}): induction heads break when their upstream signal is cut", fontweight="bold")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / f"composition_5_scores_{TAG}.png", dpi=120); print(f"→ results/composition_5_scores_{TAG}.png")

# Chart 2 — the behavioural echo: loss curves with only the UPSTREAM heads removed.
x = range(1, len(loss_clean) + 1)
cliff = P + SEQ - 0.5
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, loss_clean, color="#0E9E76", lw=1.8, label="intact")
ax.plot(x, loss_abl, color="#C0392B", lw=1.8, label=f"{len(prev_heads)} prev-token heads ablated")
ax.axvline(cliff, color="#9FAAAD", ls="--", lw=1)
ax.set_title(f"Upstream cut ({MODEL}): ablating prev-token heads also hurts in-context learning", fontweight="bold")
ax.set_xlabel("token position"); ax.set_ylabel("loss (nats)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / f"composition_6_loss_{TAG}.png", dpi=120); print(f"→ results/composition_6_loss_{TAG}.png")
