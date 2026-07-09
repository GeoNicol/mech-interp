#!/usr/bin/env python3
"""
layer_knockout.py — which layers FEED the induction heads? (hybrid-model circuit tracing)

Experiment 03 found that Qwen3.5-2B has no previous-token heads in its 6 softmax layers,
yet its induction heads have crisp stripes — suggesting the upstream "the token before me
was X" signal is carried by the 18 LINEAR-attention layers, which have no attention
pattern to score. You can't score a linear layer, but you can still cut it.

This experiment knocks out one attention sublayer at a time (softmax layers: zero every
head's z; linear layers: zero the sublayer output) and, in the same forward pass, measures
  stripe : the induction heads' mean induction score (their patterns are captured live in
           the ablated model). Zeroing z happens AFTER the same layer's pattern is
           computed, so a layer's own knockout doesn't touch its own stripe — this metric
           is pure downstream dependence.
  loss   : 2nd-copy loss (the behavioural echo).

Reading the profile: a layer whose knockout collapses the stripes is upstream
infrastructure the induction heads depend on. For an all-softmax model the profile should
spike at the known prev-token layers; for Qwen3.5 the hypothesis says early LINEAR layers
should spike.

  python 04_layer_knockout/layer_knockout.py                  # gpt2-small (default)
  python 04_layer_knockout/layer_knockout.py Qwen/Qwen3.5-2B

Outputs land in results/ next to this script.
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50
THRESH = 0.3

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01–03) ─────────────────────────────────────────────────────
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

def stripe_score(patt, offset=-(SEQ - 1)):
    return patt.diagonal(offset=offset, dim1=-2, dim2=-1).float().mean(-1)[0].cpu()

# ── baseline: find the induction heads and classify every layer ───────────────
loss_clean = per_pos_loss(tokens)
_, cache = model.run_with_cache(tokens)

induction = torch.zeros(nL, nH)
layer_kind = {}   # L -> "softmax" | "linear", plus the hook name to knock it out
for L in range(nL):
    if f"blocks.{L}.attn.hook_pattern" in cache.cache_dict:
        layer_kind[L] = ("softmax", f"blocks.{L}.attn.hook_z")
        induction[L] = stripe_score(cache[f"blocks.{L}.attn.hook_pattern"])
    elif f"blocks.{L}.linear_attn.hook_out" in cache.cache_dict:
        layer_kind[L] = ("linear", f"blocks.{L}.linear_attn.hook_out")
    else:
        raise RuntimeError(f"layer {L}: no attn or linear_attn hook found in cache")
del cache

ind_heads = [(L, H) for L in range(nL) for H in range(nH) if induction[L, H] >= THRESH]
ind_layers = sorted({L for L, _ in ind_heads})
reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)
clean_loss = loss_clean[reg].mean().item()
clean_stripe = torch.tensor([induction[L, H] for L, H in ind_heads]).mean().item()

n_lin = sum(1 for k, _ in layer_kind.values() if k == "linear")
print(f"model: {MODEL}  ({nL} layers: {nL - n_lin} softmax, {n_lin} linear)")
print(f"induction heads: {len(ind_heads)} in layers {ind_layers}")
print(f"clean 2nd-copy loss {clean_loss:.3f}, clean mean induction stripe {clean_stripe:.3f}")

# ── knockout sweep: one attention sublayer at a time ──────────────────────────
# For each layer: zero its attention sublayer output, capture the induction layers'
# patterns during the same pass, and record downstream stripe health + 2nd-copy loss.
def zero_out(t, hook):
    return torch.zeros_like(t)

stripes, losses = [], []
for L in range(nL):
    captured = {}
    def capture(patt, hook):
        captured[hook.layer()] = patt.detach()
    hooks = [(layer_kind[L][1], zero_out)] + \
            [(f"blocks.{M}.attn.hook_pattern", capture) for M in ind_layers]
    loss = per_pos_loss(tokens, hooks)
    stripes.append(torch.tensor([stripe_score(captured[M])[H] for M, H in ind_heads]).mean().item())
    losses.append(loss[reg].mean().item())
    print(f"  knockout L{L:2d} ({layer_kind[L][0]:7s}): mean stripe {stripes[-1]:.3f}   2nd-copy loss {losses[-1]:.3f}")

# ── chart: the dependency profile ─────────────────────────────────────────────
# Two stacked panels over layer index. Softmax layers orange, linear layers blue,
# induction layers hatched; dashed line = clean baseline. Deep stripe dips at LINEAR
# layers are the finding: pattern-less layers the induction circuit depends on.
colors = ["#E8A33D" if layer_kind[L][0] == "softmax" else "#4C72B0" for L in range(nL)]
hatch = ["//" if L in ind_layers else "" for L in range(nL)]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, 0.42 * nL + 2), 7), sharex=True)

b1 = ax1.bar(range(nL), stripes, color=colors)
for bar, h in zip(b1, hatch):
    bar.set_hatch(h)
ax1.axhline(clean_stripe, color="#0E9E76", ls="--", lw=1.2, label=f"clean ({clean_stripe:.2f})")
ax1.set_ylabel("mean induction stripe\nafter knockout")
ax1.set_title(f"Layer-knockout profile ({MODEL}): which layers feed the induction heads?", fontweight="bold")
ax1.legend()

b2 = ax2.bar(range(nL), losses, color=colors)
for bar, h in zip(b2, hatch):
    bar.set_hatch(h)
ax2.axhline(clean_loss, color="#0E9E76", ls="--", lw=1.2, label=f"clean ({clean_loss:.2f})")
ax2.set_ylabel("2nd-copy loss\nafter knockout")
ax2.set_xlabel("knocked-out layer  (orange = softmax attention, blue = linear attention, hatched = has induction heads)")
ax2.legend()

fig.tight_layout(); fig.savefig(OUT / f"knockout_7_profile_{TAG}.png", dpi=120); print(f"→ results/knockout_7_profile_{TAG}.png")
