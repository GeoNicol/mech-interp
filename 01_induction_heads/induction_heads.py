#!/usr/bin/env python3
"""
induction_heads.py — find, score, and causally break the induction circuit.

Mechanistic-interpretability's canonical first experiment, run as serve -> measure -> break -> chart:
  serve   : load GPT-2-small with hooks (TransformerLens HookedTransformer)
  measure : per-position loss on a REPEATED random sequence (in-context learning shows up as a
            loss cliff on the 2nd copy) + a per-head "induction score" from attention patterns
  break   : ablate the induction heads and re-measure -> the cliff vanishes (causal proof)
  chart   : (1) the induction cliff  (2) the layer x head induction-head heatmap  (3) ablation A/B

Runs in seconds on a 12 GB GPU. From WSL with the venv active:
  source /root/.venv/bin/activate
  python 01_induction_heads/induction_heads.py                    # gpt2-small (default)
  python 01_induction_heads/induction_heads.py Qwen/Qwen3-1.7B    # any TransformerLens-supported model

Outputs land in results/ next to this script.
"""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"   # charts/token dumps go here, not the cwd
OUT.mkdir(exist_ok=True)

# Global setup: this is a pure-inference experiment, so gradients are disabled everywhere
# (halves memory, speeds up the forward passes). SEQ and THRESH are the only two knobs.
torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50          # length of the random block that gets repeated
THRESH = 0.3      # induction-score threshold for "this is an induction head"

# Model comes from the command line (default gpt2-small). TAG makes every output file
# model-specific so runs on different models sit side by side instead of overwriting.
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
# gpt2-small fits comfortably in fp32; larger models load in bf16 to stay inside 12 GB.
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve ────────────────────────────────────────────────────────────────────
# Load the model through TransformerLens: same weights as HuggingFace, but every
# internal activation (attention patterns, per-head outputs, ...) is exposed as a named
# hook point we can read (run_with_cache) or overwrite (run_with_hooks) later.
# Newer architectures (e.g. Qwen3.5) aren't in HookedTransformer's registry — fall back
# to TransformerBridge (the TL 3.x loader) with compatibility mode, which re-exposes the
# classic hook names ("pattern", blocks.L.attn.hook_z) so the rest of the script is unchanged.
try:
    model = HookedTransformer.from_pretrained(MODEL, device=DEV, dtype=DTYPE)
except Exception:
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(MODEL, device=DEV, dtype=DTYPE)
    # no_processing skips TransformerLens's fold-LN / centre-unembed reparameterisation.
    # It's output-preserving for what we measure (attention patterns + hook_z + loss) and
    # avoids cloning the huge unembed — which OOMs a 12 GB GPU on big hybrids like
    # Qwen3.5-4B (32L, d_vocab 248k). Peak then ~9.6 GB instead of >12 GB.
    model.enable_compatibility_mode(no_processing=True)
nL, nH = model.cfg.n_layers, model.cfg.n_heads

# Build the stimulus: [BOS] r0..r49 r0..r49 — a random block repeated once. Random tokens
# carry no learnable statistics, so the 2nd copy is only predictable by looking back at
# what followed each token last time (i.e. by induction). Seeded for reproducible charts.
# Some tokenizers (e.g. Qwen) define no BOS token — fall back to EOS, or no prefix at all.
torch.manual_seed(0)
bos = model.tokenizer.bos_token_id
if bos is None:
    bos = model.tokenizer.eos_token_id
prefix = [[bos]] if bos is not None else [[]]
P = len(prefix[0])                                  # prefix length (1 or 0)
rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))
tokens = torch.cat([torch.tensor(prefix, dtype=torch.long), rand, rand], dim=1).to(DEV)  # [1, P+2*SEQ]

# Dump the exact token sequence to a text file so the stimulus can be inspected:
# position, token id, and the decoded string (repr'd so whitespace/bytes are visible).
with open(OUT / f"tokens_{TAG}.txt", "w", encoding="utf-8") as f:
    f.write(f"# stimulus for {MODEL}: {'[BOS/EOS] + ' if P else ''}{SEQ} random tokens, repeated once\n")
    f.write(f"# pos\ttoken_id\tdecoded\n")
    for i, t in enumerate(tokens[0].tolist()):
        role = "prefix" if i < P else ("copy1" if i < P + SEQ else "copy2")
        f.write(f"{i}\t{t}\t{role}\t{model.tokenizer.decode([t])!r}\n")
print(f"→ results/tokens_{TAG}.txt")

# One forward pass -> loss at every position (loss[i] = loss predicting token i+1).
# Optional fwd_hooks let the same function measure the ablated model in the "break" step.
def per_pos_loss(toks, fwd_hooks=()):
    return model.run_with_hooks(toks, return_type="loss", loss_per_token=True,
                                fwd_hooks=list(fwd_hooks))[0].float().cpu()   # [P+2*SEQ-1]

# ── measure: the behaviour, and which heads cause it ──────────────────────────
# Two passes over the same tokens: the clean per-position loss (shows the cliff on the
# 2nd copy), and a cached run that records every attention pattern for head scoring.
loss_clean = per_pos_loss(tokens)
_, cache = model.run_with_cache(tokens)

# Induction score per head: average attention on the stripe where a destination token attends
# back to the token that FOLLOWED its previous occurrence (offset = -(SEQ-1)). A head doing
# "A B ... A -> attend to B" puts its attention mass exactly on that diagonal, so the stripe
# mean is ~1 for a perfect induction head and ~0 for anything else.
# Hybrid models (e.g. Qwen3.5) interleave linear-attention layers that have NO attention
# pattern — those layers are skipped (left at score 0) and only true-attention layers scored.
induction = torch.zeros(nL, nH)
attn_layers = []
for L in range(nL):
    key = f"blocks.{L}.attn.hook_pattern"
    if key not in cache.cache_dict:
        continue
    attn_layers.append(L)
    patt = cache[key]                                            # [1, head, dest, src]
    stripe = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1)  # [1, head, diag]
    induction[L] = stripe.float().mean(-1)[0].cpu()              # .float(): accurate mean under bf16
if len(attn_layers) < nL:
    print(f"hybrid architecture: only {len(attn_layers)}/{nL} layers have attention patterns: {attn_layers}")

# Collect the heads above threshold, strongest first — this list IS the induction circuit,
# and is what gets ablated next.
top = [(L, H, induction[L, H].item()) for L in range(nL) for H in range(nH)
       if induction[L, H] >= THRESH]
top.sort(key=lambda x: -x[2])

# ── break: ablate the induction heads, re-measure ─────────────────────────────
# Causal test: if these heads merely correlate with the behaviour, removing them changes
# little; if they implement it, the 2nd-copy loss should snap back up to 1st-copy levels.
heads_by_layer = defaultdict(list)
for L, H, _ in top:
    heads_by_layer[L].append(H)

# hook_z is the per-head attention output BEFORE the heads are mixed by W_O, so zeroing
# z[:, :, h, :] deletes head h's entire contribution while leaving every other head intact.
# One hook function serves all layers — hook.layer() tells it which heads to zero here.
def zero_heads(z, hook):                # z: [batch, pos, head, d_head]
    for h in heads_by_layer[hook.layer()]:
        z[:, :, h, :] = 0.0
    return z
hooks = [(f"blocks.{L}.attn.hook_z", zero_heads) for L in heads_by_layer]
loss_abl = per_pos_loss(tokens, hooks)

# induction region = predictions of the 2nd copy. loss[i] predicts token i+1, and the 2nd
# copy spans token positions P+SEQ .. P+2*SEQ-1, so its losses live at P+SEQ-1 .. P+2*SEQ-2.
reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)
ind_clean, ind_abl = loss_clean[reg].mean().item(), loss_abl[reg].mean().item()

print(f"model: {MODEL}  ({nL} layers x {nH} heads)")
print(f"induction heads (score>={THRESH}): " +
      ", ".join(f"L{L}H{H}={s:.2f}" for L, H, s in top))
print(f"in-context (2nd-copy) loss   clean: {ind_clean:.3f}   ablated: {ind_abl:.3f}"
      f"   -> {ind_abl/ind_clean:.1f}x worse when the induction heads are removed")

# ── chart ─────────────────────────────────────────────────────────────────────
x = range(1, len(loss_clean) + 1)   # loss[i] predicts the token at position i+1
cliff = P + SEQ - 0.5               # x-coordinate where the 2nd copy begins

# Chart 1 — the induction cliff: clean per-position loss, with a dashed line marking where
# the 2nd copy starts. The drop after that line is in-context learning made visible.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, loss_clean, color="#0E9E76", lw=1.8)
ax.axvline(cliff, color="#9FAAAD", ls="--", lw=1)
ax.text(cliff + SEQ / 2, ax.get_ylim()[1]*0.9, "2nd copy →\nin-context learning", ha="center", fontsize=9, color="#54534C")
ax.set_title(f"Induction cliff: per-position loss on a repeated random sequence\n{MODEL}", fontweight="bold")
ax.set_xlabel("token position"); ax.set_ylabel("loss (nats)")
fig.tight_layout(); fig.savefig(OUT / f"induction_1_cliff_{TAG}.png", dpi=120); print(f"→ results/induction_1_cliff_{TAG}.png")

# Chart 2 — the circuit map: induction score for all layer x head pairs, with a white dot
# on every head above THRESH (the ones that were ablated).
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(induction, cmap="viridis", aspect="auto")
for L, H, s in top:
    ax.text(H, L, "●", ha="center", va="center", color="white", fontsize=9)
ax.set_title(f"Induction score by (layer, head) — the circuit\n{MODEL}", fontweight="bold")
ax.set_xlabel("head"); ax.set_ylabel("layer"); fig.colorbar(im, label="induction score")
fig.tight_layout(); fig.savefig(OUT / f"induction_2_heatmap_{TAG}.png", dpi=120); print(f"→ results/induction_2_heatmap_{TAG}.png")

# Chart 3 — the A/B: clean vs ablated loss overlaid. The ablated curve staying high on the
# 2nd copy (no cliff) is the causal proof that the flagged heads carry in-context learning.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, loss_clean, color="#0E9E76", lw=1.8, label="intact")
ax.plot(x, loss_abl, color="#C0392B", lw=1.8, label="induction heads ablated")
ax.axvline(cliff, color="#9FAAAD", ls="--", lw=1)
ax.set_title(f"Break it: ablating the induction heads kills in-context learning\n{MODEL}", fontweight="bold")
ax.set_xlabel("token position"); ax.set_ylabel("loss (nats)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / f"induction_3_ablation_{TAG}.png", dpi=120); print(f"→ results/induction_3_ablation_{TAG}.png")
