#!/usr/bin/env python3
"""
capture_emergence.py — record the induction circuit being BORN, for the 3D viewer.

Experiment 07 proves the phase change with two charts; this script records everything
emergence.html needs to REPLAY it: the full experiment-01 measurement (per-position
loss/top-1 prediction, top attention edges, per-head firing) at each of the 14 published
Pythia-160m training checkpoints, on the SAME repeated random sequence every time.
The viewer then adds a second time axis — scrub through training and watch the heads
wall ignite and the targeting board flip from red to green between step 512 and 1000.

Output is results/emergence_{TAG}.js — a JS assignment (not raw JSON) so emergence.html
can load it with a plain <script> tag straight from file:// (no CORS, no web server).

  python 11_induction_3d/capture_emergence.py                        # pythia-160m (default)
  python 11_induction_3d/capture_emergence.py EleutherAI/pythia-410m # any Pythia size
"""
import json, sys, torch
from pathlib import Path
from transformer_lens import HookedTransformer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = 50          # length of the random block that gets repeated
THRESH = 0.3      # induction-score threshold (same as experiments 01/07)
TOPK = 3          # attention edges kept per (layer, head, destination)
W_MIN = 0.03      # attention edges below this weight are dropped

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-160m"
TAG = MODEL.split("/")[-1].replace(".", "_")

# same grid as experiment 07: dense around the expected phase change (~step 1000-2500)
STEPS = [0, 64, 256, 512, 1000, 2000, 3000, 4000, 5000, 8000, 16000, 32000, 64000, 143000]

# only the two hook families the viewer needs get cached (patterns + per-head outputs)
FILT = lambda n: n.endswith("hook_pattern") or n.endswith("hook_z")

tokens = None   # built from the first checkpoint's tokenizer; identical at every step

def capture(step):
    """Load one checkpoint and extract the full replay payload for the viewer."""
    global tokens
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step,
                                              device=DEV, dtype=torch.float32)
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    if tokens is None:                       # stimulus (same as experiments 01/07)
        torch.manual_seed(0)
        bos = model.tokenizer.bos_token_id
        if bos is None:
            bos = model.tokenizer.eos_token_id
        prefix = [[bos]] if bos is not None else [[]]
        rand = torch.randint(0, model.cfg.d_vocab, (1, SEQ))
        tokens = torch.cat([torch.tensor(prefix, dtype=torch.long), rand, rand], dim=1).to(DEV)
    P = tokens.shape[1] - 2 * SEQ
    N = tokens.shape[1]

    logits, cache = model.run_with_cache(tokens, names_filter=FILT)

    # induction score per head (the exp-01 stripe) — this colours the heads wall
    induction = torch.zeros(nL, nH)
    for L in range(nL):
        patt = cache[f"blocks.{L}.attn.hook_pattern"]
        stripe = patt.diagonal(offset=-(SEQ - 1), dim1=-2, dim2=-1)
        induction[L] = stripe.float().mean(-1)[0].cpu()

    lp = torch.log_softmax(logits[0].float(), -1)                    # [N, vocab]
    nxt = tokens[0, 1:]
    loss = (-lp[:-1].gather(-1, nxt[:, None])[:, 0]).cpu()           # loss[i] predicts token i+1
    top_p, top_i = lp[:-1].exp().max(-1)

    # attention edges: top-K sources per (layer, head, dest), weight quantized to 0..255
    eL, eH, eD, eS, eW = [], [], [], [], []
    for L in range(nL):
        patt = cache[f"blocks.{L}.attn.hook_pattern"][0].float().cpu()   # [H, dest, src]
        w, s = patt.topk(min(TOPK, patt.shape[-1]), dim=-1)
        for h, d, k in (w >= W_MIN).nonzero().tolist():
            eL.append(L); eH.append(h); eD.append(d)
            eS.append(int(s[h, d, k])); eW.append(round(float(w[h, d, k]) * 255))

    # firing intensity: ‖z‖ per (layer, pos, head), u8 after per-checkpoint norm
    # (normalized within the checkpoint — raw norms drift over training; the emergence
    #  story is carried by the induction scores, the glow just shows who is firing NOW)
    act = torch.zeros(nL, N, nH)
    for L in range(nL):
        act[L] = cache[f"blocks.{L}.attn.hook_z"][0].float().norm(dim=-1).cpu()
    scale = 255.0 / max(act.max().item(), 1e-6)

    l1 = loss[slice(P, P + SEQ - 1)].mean().item()
    l2 = loss[slice(P + SEQ - 1, P + 2 * SEQ - 1)].mean().item()
    ckpt = {
        "step": step,
        "stats": {"loss1": round(l1, 3), "loss2": round(l2, 3),
                  "topScore": round(induction.max().item(), 3)},
        "ind": [[round(v, 3) for v in row] for row in induction.tolist()],
        "loss": [round(v, 4) for v in loss.tolist()],
        "pred": [model.tokenizer.decode([t]) for t in top_i.tolist()],
        "predP": [round(v, 4) for v in top_p.tolist()],
        "predOk": [int(a == b) for a, b in zip(top_i.tolist(), nxt.tolist())],
        "edges": {"L": eL, "H": eH, "D": eD, "S": eS, "W": eW},
        "act": [[min(255, round(v * scale)) for v in act[L].flatten().tolist()]
                for L in range(nL)],
    }
    meta = (nL, nH, P, N, induction,
            [model.tokenizer.decode([t]) for t in tokens[0].tolist()])
    del cache, model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return ckpt, meta

# ── sweep the checkpoints ─────────────────────────────────────────────────────
print(f"model: {MODEL}   checkpoints: {len(STEPS)}")
ckpts = []
for step in STEPS:
    ckpt, meta = capture(step)
    ckpts.append(ckpt)
    s = ckpt["stats"]
    print(f"  step {step:>6}: 2nd-copy loss {s['loss2']:6.2f}   top induction score "
          f"{s['topScore']:.2f}   copy-2 hits {sum(ckpt['predOk'][meta[2]+SEQ-1:]):>2}/{SEQ}"
          f"   edges {len(ckpt['edges']['L'])}")

nL, nH, P, N, ind_final, tok_strs = meta   # meta from the LAST (fully trained) checkpoint

# "destined" induction heads: above threshold at the end of training — the viewer rings
# these from step 0 so you can watch the exact spots where the circuit will be born
final = [(L, H, ind_final[L, H].item()) for L in range(nL) for H in range(nH)
         if ind_final[L, H] >= THRESH]
final.sort(key=lambda x: -x[2])

data = {
    "model": MODEL, "tag": TAG, "nL": nL, "nH": nH, "SEQ": SEQ, "P": P,
    "tokens": tok_strs,
    "steps": STEPS,
    "tokensPerStep": 2.1e6,                       # Pythia: 1024 seqs x 2049 tokens per step
    "induction": [[L, H, round(s, 3)] for L, H, s in final],
    "region": [P + SEQ - 1, P + 2 * SEQ - 1],
    "ckpts": ckpts,
}

path = OUT / f"emergence_{TAG}.js"
with open(path, "w", encoding="utf-8") as f:
    f.write("window.EMERGE_DATA = window.EMERGE_DATA || {};\n")
    f.write(f"window.EMERGE_DATA[{json.dumps(TAG)}] = ")
    f.write(json.dumps(data, separators=(",", ":")))
    f.write(";\n")

# shared manifest (all four scenes); file:// pages can't list directories
idx = {"induction": sorted(p.stem[len("scene_"):] for p in OUT.glob("scene_*.js")),
       "refusal":   sorted(p.stem[len("refusal_"):] for p in OUT.glob("refusal_*.js")),
       "unlearning": sorted(p.stem[len("unlearning_"):] for p in OUT.glob("unlearning_*.js")),
       "emergence": sorted(p.stem[len("emergence_"):] for p in OUT.glob("emergence_*.js"))}
with open(OUT / "index.js", "w", encoding="utf-8") as f:
    f.write(f"window.SCENE_INDEX = {json.dumps(idx)};\n")

first, last = ckpts[0]["stats"], ckpts[-1]["stats"]
print(f"phase change: 2nd-copy loss {first['loss2']:.1f} → {last['loss2']:.1f}, "
      f"top induction score {first['topScore']:.2f} → {last['topScore']:.2f}")
print(f"destined induction heads (score>={THRESH} at step {STEPS[-1]}): " +
      ", ".join(f"L{L}H{H}={s:.2f}" for L, H, s in final))
print(f"→ results/emergence_{TAG}.js ({path.stat().st_size/1e6:.1f} MB)"
      f" — open 11_induction_3d/emergence.html" + (f"?tag={TAG}" if TAG != "pythia-160m" else ""))
