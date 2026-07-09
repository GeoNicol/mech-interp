#!/usr/bin/env python3
"""
mean_ablation.py — is the self-repair real, or an artifact of the zero?

Zero-ablation (01/05) shoves an off-distribution activation (all zeros) into the residual
stream. That alone can fake "self-repair": LayerNorm renormalises what's left, amplifying
surviving components, and downstream layers see statistics they never saw in training.
Mean ablation is the standard fix — replace each ablated head's output with its AVERAGE
output over a reference batch, which deletes the head's information content while keeping
the residual-stream statistics in-distribution.

Comparing the two on the same induction heads separates the hypotheses:
  mean ≈ zero : the robustness measured in 01/05 is genuine functional redundancy
  mean >> zero: part of the apparent self-repair was an artifact of zeroing

The reference batch is fresh repeated-random sequences (same task distribution as the
stimulus, different seeds), so the mean is "what this head typically emits on this task".

  python 06_mean_ablation/mean_ablation.py Qwen/Qwen3-1.7B   # the model with the Hydra
  python 06_mean_ablation/mean_ablation.py                   # gpt2 (default)

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
THRESH = 0.3
REF_BATCH = 16    # reference sequences used to estimate each head's mean output

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as 01–05) ─────────────────────────────────────────────────────
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

def per_pos_loss(toks, fwd_hooks=()):
    return model.run_with_hooks(toks, return_type="loss", loss_per_token=True,
                                fwd_hooks=list(fwd_hooks))[0].float().cpu()

def make_seq(seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, model.cfg.d_vocab, (1, SEQ), generator=g)
    return torch.cat([torch.tensor(prefix, dtype=torch.long), r, r], dim=1).to(DEV)

# ── find the induction heads (same scoring as 01) ─────────────────────────────
loss_clean = per_pos_loss(tokens)
_, cache = model.run_with_cache(tokens)
induction = torch.zeros(nL, nH)
for L in range(nL):
    key = f"blocks.{L}.attn.hook_pattern"
    if key not in cache.cache_dict:
        continue
    stripe = cache[key].diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1)
    induction[L] = stripe.float().mean(-1)[0].cpu()
del cache

top = [(L, H) for L in range(nL) for H in range(nH) if induction[L, H] >= THRESH]
by_layer = defaultdict(list)
for L, H in top:
    by_layer[L].append(H)

# ── estimate each induction head's mean output on the task distribution ───────
# Run REF_BATCH fresh repeated-random sequences (seeds disjoint from the stimulus) and
# average each head's z over batch and position -> one [d_head] vector per head. This is
# "what the head typically emits here", the in-distribution replacement for zero.
sums, count = {}, 0
def accumulate(z, hook):   # z: [1, pos, head, d_head]
    L = hook.layer()
    for h in by_layer[L]:
        sums[(L, h)] = sums.get((L, h), 0) + z[0, :, h, :].float().sum(0).cpu()
    return z
acc_hooks = [(f"blocks.{L}.attn.hook_z", accumulate) for L in by_layer]
for i in range(REF_BATCH):
    seq = make_seq(1000 + i)
    model.run_with_hooks(seq, return_type=None, fwd_hooks=acc_hooks)
    count += seq.shape[1]
mean_z = {k: (v / count).to(DEV, DTYPE) for k, v in sums.items()}

# ── ablate the same heads two ways ────────────────────────────────────────────
def zero_heads(z, hook):
    for h in by_layer[hook.layer()]:
        z[:, :, h, :] = 0.0
    return z
def mean_heads(z, hook):
    for h in by_layer[hook.layer()]:
        z[:, :, h, :] = mean_z[(hook.layer(), h)]
    return z

loss_zero = per_pos_loss(tokens, [(f"blocks.{L}.attn.hook_z", zero_heads) for L in by_layer])
loss_mean = per_pos_loss(tokens, [(f"blocks.{L}.attn.hook_z", mean_heads) for L in by_layer])

c = loss_clean[reg].mean().item()
z = loss_zero[reg].mean().item()
m = loss_mean[reg].mean().item()
print(f"model: {MODEL}  ({len(top)} induction heads ablated, mean from {REF_BATCH} ref sequences)")
print(f"2nd-copy loss   clean: {c:.3f}   zero-ablated: {z:.3f} ({z/c:.1f}x)   mean-ablated: {m:.3f} ({m/c:.1f}x)")
verdict = ("mean ≈ zero -> the robustness under zero-ablation is genuine functional redundancy"
           if abs(m - z) / max(z - c, 1e-6) < 0.25 else
           ("mean >> zero -> part of the apparent self-repair was a zeroing artifact" if m > z else
            "mean << zero -> zeroing itself caused extra damage beyond removing the heads' information"))
print(f"verdict: {verdict}")

# ── chart ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = [c, m, z]
ax.bar(range(3), bars, color=["#0E9E76", "#E8A33D", "#C0392B"], width=0.6)
for i, v in enumerate(bars):
    ax.text(i, v * 1.02, f"{v:.2f}", ha="center", fontsize=10, color="#54534C")
ax.set_ylim(top=max(bars) * 1.15)
ax.set_xticks(range(3))
ax.set_xticklabels(["clean", f"mean-ablated\n({len(top)} heads)", f"zero-ablated\n({len(top)} heads)"])
ax.set_ylabel("2nd-copy loss (nats)")
ax.set_title(f"Mean vs zero ablation: is the self-repair real?\n{MODEL}", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / f"meanabl_11_compare_{TAG}.png", dpi=120); print(f"→ results/meanabl_11_compare_{TAG}.png")
