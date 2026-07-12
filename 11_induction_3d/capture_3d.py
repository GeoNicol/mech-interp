#!/usr/bin/env python3
"""
capture_3d.py — record the induction circuit firing, for the 3D replay viewer.

Experiment 01 proves the circuit with static charts; this script records everything the
companion viewer.html needs to REPLAY it in 3D, token by token:
  loss/pred : per-position loss and top-1 prediction (clean vs induction-heads-ablated)
  edges     : attention compressed to the top-3 source positions per (layer, head, dest)
  activity  : each head's firing intensity per token (norm of its hook_z output)

Output is results/scene_{TAG}.js — a JS assignment (not raw JSON) so viewer.html can load
it with a plain <script> tag straight from file:// (no CORS, no web server needed).

  python 11_induction_3d/capture_3d.py                  # gpt2-small (default)
  python 11_induction_3d/capture_3d.py Qwen/Qwen3-1.7B  # then open viewer.html?tag=Qwen3-1_7B
"""
import json, sys, torch
from collections import defaultdict
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50          # length of the random block that gets repeated
THRESH = 0.3      # induction-score threshold (same as experiment 01)
TOPK = 3          # attention edges kept per (layer, head, destination)
W_MIN = 0.03      # attention edges below this weight are dropped

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TAG = MODEL.split("/")[-1].replace(".", "_")
DTYPE = torch.float32 if MODEL == "gpt2" else torch.bfloat16

# ── serve (same as experiment 01, incl. TransformerBridge fallback) ───────────
try:
    model = HookedTransformer.from_pretrained(MODEL, device=DEV, dtype=DTYPE)
except Exception:
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(MODEL, device=DEV, dtype=DTYPE)
    model.enable_compatibility_mode()
nL, nH = model.cfg.n_layers, model.cfg.n_heads

# ── stimulus (same as experiment 01) ──────────────────────────────────────────
torch.manual_seed(0)
bos = model.tokenizer.bos_token_id
if bos is None:
    bos = model.tokenizer.eos_token_id
prefix = [[bos]] if bos is not None else [[]]
P = len(prefix[0])
rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))
tokens = torch.cat([torch.tensor(prefix, dtype=torch.long), rand, rand], dim=1).to(DEV)
N = tokens.shape[1]

# only the two hook families the viewer needs get cached (patterns + per-head outputs)
FILT = lambda n: n.endswith("hook_pattern") or n.endswith("hook_z")

# ── clean pass ────────────────────────────────────────────────────────────────
logits_clean, cache_clean = model.run_with_cache(tokens, names_filter=FILT)

# induction score per head from the clean patterns (same stripe as experiment 01)
induction = torch.zeros(nL, nH)
attn_layers = []
for L in range(nL):
    key = f"blocks.{L}.attn.hook_pattern"
    if key not in cache_clean.cache_dict:
        continue
    attn_layers.append(L)
    patt = cache_clean[key]
    stripe = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1)
    induction[L] = stripe.float().mean(-1)[0].cpu()
top = [(L, H, induction[L, H].item()) for L in range(nL) for H in range(nH)
       if induction[L, H] >= THRESH]
top.sort(key=lambda x: -x[2])

# ── ablated pass (cached, so downstream pattern shifts are captured too) ──────
heads_by_layer = defaultdict(list)
for L, H, _ in top:
    heads_by_layer[L].append(H)

def zero_heads(z, hook):                # z: [batch, pos, head, d_head]
    for h in heads_by_layer[hook.layer()]:
        z[:, :, h, :] = 0.0
    return z
hooks = [(f"blocks.{L}.attn.hook_z", zero_heads) for L in heads_by_layer]
with model.hooks(fwd_hooks=hooks):
    logits_abl, cache_abl = model.run_with_cache(tokens, names_filter=FILT)

# ── extract the viewer payload per condition ──────────────────────────────────
def payload(logits, cache):
    lp = torch.log_softmax(logits[0].float(), -1)                    # [N, vocab]
    nxt = tokens[0, 1:]
    loss = (-lp[:-1].gather(-1, nxt[:, None])[:, 0]).cpu()           # loss[i] predicts token i+1
    probs = lp[:-1].exp()
    top_p, top_i = probs.max(-1)

    # attention edges: top-K sources per (layer, head, dest), weight quantized to 0..255
    eL, eH, eD, eS, eW = [], [], [], [], []
    for L in attn_layers:
        patt = cache[f"blocks.{L}.attn.hook_pattern"][0].float().cpu()   # [H, dest, src]
        w, s = patt.topk(min(TOPK, patt.shape[-1]), dim=-1)
        for h, d, k in (w >= W_MIN).nonzero().tolist():
            eL.append(L); eH.append(h); eD.append(d)
            eS.append(int(s[h, d, k])); eW.append(round(float(w[h, d, k]) * 255))

    # firing intensity: ‖z‖ per (layer, pos, head); flattened pos*nH+h, u8 after global norm
    act = torch.zeros(nL, N, nH)
    for L in range(nL):
        key = f"blocks.{L}.attn.hook_z"
        if key in cache.cache_dict:
            act[L] = cache[key][0].float().norm(dim=-1).cpu()        # [pos, head]

    return {
        "loss": [round(v, 4) for v in loss.tolist()],
        "pred": [model.tokenizer.decode([t]) for t in top_i.tolist()],
        "predP": [round(v, 4) for v in top_p.tolist()],
        "predOk": [int(a == b) for a, b in zip(top_i.tolist(), nxt.tolist())],
        "edges": {"L": eL, "H": eH, "D": eD, "S": eS, "W": eW},
    }, act, loss

clean, act_clean, loss_clean = payload(logits_clean, cache_clean)
abl, act_abl, loss_abl = payload(logits_abl, cache_abl)

# normalize both conditions by the CLEAN max so ablated heads visibly dim
scale = 255.0 / max(act_clean.max().item(), 1e-6)
for cond, act in ((clean, act_clean), (abl, act_abl)):
    cond["act"] = [[min(255, round(v * scale)) for v in act[L].flatten().tolist()]
                   for L in range(nL)]

reg = slice(P + SEQ - 1, P + 2 * SEQ - 1)
ind_clean, ind_abl = loss_clean[reg].mean().item(), loss_abl[reg].mean().item()

data = {
    "model": MODEL, "tag": TAG, "nL": nL, "nH": nH, "SEQ": SEQ, "P": P,
    "tokens": [model.tokenizer.decode([t]) for t in tokens[0].tolist()],
    "induction": [[L, H, round(s, 3)] for L, H, s in top],
    "attnLayers": attn_layers,
    "region": [P + SEQ - 1, P + 2 * SEQ - 1],
    "stats": {"indClean": round(ind_clean, 3), "indAbl": round(ind_abl, 3),
              "ratio": round(ind_abl / ind_clean, 1)},
    "clean": clean, "ablated": abl,
}

path = OUT / f"scene_{TAG}.js"
with open(path, "w", encoding="utf-8") as f:
    f.write("window.SCENE_DATA = window.SCENE_DATA || {};\n")
    f.write(f"window.SCENE_DATA[{json.dumps(TAG)}] = ")
    f.write(json.dumps(data, separators=(",", ":")))
    f.write(";\n")

print(f"model: {MODEL}  ({nL} layers x {nH} heads)")
print(f"induction heads (score>={THRESH}): " +
      ", ".join(f"L{L}H{H}={s:.2f}" for L, H, s in top))
print(f"in-context (2nd-copy) loss   clean: {ind_clean:.3f}   ablated: {ind_abl:.3f}"
      f"   -> {ind_abl/ind_clean:.1f}x worse when the induction heads are removed")
print(f"edges: clean {len(clean['edges']['L'])}, ablated {len(abl['edges']['L'])}")
print(f"→ results/scene_{TAG}.js ({path.stat().st_size/1e6:.1f} MB)"
      f" — open 11_induction_3d/viewer.html" + (f"?tag={TAG}" if TAG != "gpt2" else ""))
