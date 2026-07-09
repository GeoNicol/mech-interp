#!/usr/bin/env python3
"""
random_control.py — control experiment: is the ablation damage SPECIFIC to induction heads?

Experiment 01 showed that zero-ablating the induction heads destroys in-context copying.
But that alone doesn't prove specificity — maybe removing ANY k attention heads hurts this
much. This control repeats the ablation with k randomly chosen NON-induction heads, drawn
from the same population (layers that actually have attention), over several trials:
  serve   : load the model (same as 01)
  measure : induction scores -> the k induction heads (same as 01)
  control : ablate k random non-induction heads, TRIALS times, and compare against
            ablating the k induction heads
  chart   : clean vs random-ablation (all trials) vs induction-ablation, on the 2nd copy

If random ablations barely move the loss while the induction ablation devastates it, the
effect is specific to the circuit — the standard control that separates "these heads
implement the behaviour" from "the model just dislikes losing heads".

  python 02_random_control/random_control.py                    # gpt2-small (default)
  python 02_random_control/random_control.py Qwen/Qwen3.5-2B

Outputs land in results/ next to this script.
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"   # charts go here, not the cwd
OUT.mkdir(exist_ok=True)

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50          # length of the random block that gets repeated
THRESH = 0.3      # induction-score threshold for "this is an induction head"
TRIALS = 10       # number of independent random-head ablations

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01) ────────────────────────────────────────────────────────
try:
    model = HookedTransformer.from_pretrained(MODEL, device=DEV, dtype=DTYPE)
except Exception:
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(MODEL, device=DEV, dtype=DTYPE)
    model.enable_compatibility_mode()
nL, nH = model.cfg.n_layers, model.cfg.n_heads

# Same seeded stimulus as 01 so results are directly comparable across experiments.
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

# ── measure: find the induction heads (same scoring as 01) ────────────────────
loss_clean = per_pos_loss(tokens)
_, cache = model.run_with_cache(tokens)

induction = torch.zeros(nL, nH)
attn_layers = []
for L in range(nL):
    key = f"blocks.{L}.attn.hook_pattern"
    if key not in cache.cache_dict:
        continue
    attn_layers.append(L)
    patt = cache[key]
    stripe = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1)
    induction[L] = stripe.float().mean(-1)[0].cpu()
del cache

top = [(L, H) for L in range(nL) for H in range(nH) if induction[L, H] >= THRESH]
k = len(top)

# ── control: ablate k induction heads vs k random non-induction heads ─────────
# One reusable ablation runner: takes any list of (layer, head) pairs, zeroes their
# per-head output (hook_z, pre-W_O — same surgical cut as experiment 01), and returns
# the mean loss over the 2nd copy only.
reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)
def ablated_2ndcopy_loss(heads):
    by_layer = defaultdict(list)
    for L, H in heads:
        by_layer[L].append(H)
    def zero_heads(z, hook):
        for h in by_layer[hook.layer()]:
            z[:, :, h, :] = 0.0
        return z
    hooks = [(f"blocks.{L}.attn.hook_z", zero_heads) for L in by_layer]
    return per_pos_loss(tokens, hooks)[reg].mean().item()

ind_clean = loss_clean[reg].mean().item()
ind_abl = ablated_2ndcopy_loss(top)

# The fair comparison pool: every head in a layer that has real attention (matters for
# hybrid models, where linear-attention layers have no hook_z), minus the circuit itself.
pool = [(L, H) for L in attn_layers for H in range(nH) if (L, H) not in set(top)]

# Each trial gets its own seeded generator (seeds 1..TRIALS) so trials are independent
# of each other AND of the stimulus seed (0), and the whole experiment reproduces exactly.
rand_losses = []
for t in range(TRIALS):
    g = torch.Generator().manual_seed(1 + t)
    pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:k].tolist()]
    rand_losses.append(ablated_2ndcopy_loss(pick))
rand_t = torch.tensor(rand_losses)

print(f"model: {MODEL}  ({nL} layers x {nH} heads, {len(attn_layers)} attention layers)")
print(f"k = {k} heads ablated per run; random pool = {len(pool)} heads; {TRIALS} random trials")
print(f"2nd-copy loss   clean:              {ind_clean:.3f}")
print(f"2nd-copy loss   random heads:       {rand_t.mean():.3f} ± {rand_t.std():.3f}   (min {rand_t.min():.3f}, max {rand_t.max():.3f})")
print(f"2nd-copy loss   induction heads:    {ind_abl:.3f}")
print(f"-> induction ablation is {(ind_abl - rand_t.mean()) / rand_t.std():.0f} standard deviations above the random-ablation mean")

# ── chart: the specificity picture ─────────────────────────────────────────────
# Three bars (clean / random / induction) with the individual random trials overlaid as
# dots — the gap between the last two bars is the whole result.
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = [ind_clean, rand_t.mean().item(), ind_abl]
ax.bar(range(3), bars, color=["#0E9E76", "#9FAAAD", "#C0392B"], width=0.6)
ax.scatter([1] * TRIALS, rand_losses, color="#54534C", zorder=3, s=18, label=f"{TRIALS} random trials")
ax.set_xticks(range(3))
ax.set_xticklabels(["clean", f"{k} random heads\nablated", f"{k} induction heads\nablated"])
ax.set_ylabel("2nd-copy loss (nats)")
ax.set_title(f"Specificity control ({MODEL}): only the induction heads matter", fontweight="bold")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / f"control_4_random_{TAG}.png", dpi=120); print(f"→ results/control_4_random_{TAG}.png")
