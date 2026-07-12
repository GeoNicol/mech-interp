# mech-interp

Mechanistic-interpretability experiments — a numbered sequence of experiment folders
(`01_.../`, `02_.../`), each holding one self-contained script plus a `results/`
subfolder for its charts and dumps. Every script takes an optional model name argument
(GPT-2-small by default), works with any TransformerLens-loadable model, and writes its
outputs to its own `results/` regardless of the cwd.
Convention: scripts stay self-contained (some duplicated boilerplate is deliberate) so
each experiment reads and runs on its own; extract shared helpers only if it gets painful.
Git: one commit per experiment.

## Experiments

- `01_induction_heads/induction_heads.py` — find the induction circuit: load a model with
  hooks, measure the induction cliff on a repeated random sequence, score every attention
  head, zero-ablate the induction heads, and write three charts + a token dump.
- `02_random_control/random_control.py` — specificity control: ablate the same NUMBER of
  randomly chosen non-induction heads (10 seeded trials, drawn from the same
  attention-layer pool) and compare against ablating the induction heads. Writes
  `control_4_random_{TAG}.png` (bar chart: clean / random / induction).
- `03_prev_token_heads/prev_token_heads.py` — composition: find previous-token heads
  (offset −1 diagonal, induction heads excluded), ablate them, and re-score the induction
  heads' stripes inside the ablated model. 2nd CLI arg overrides PREV_THRESH (default 0.5;
  non-default runs get a `_p{..}` file tag). Writes `composition_5_scores_{TAG}.png`
  (clean vs ablated induction score per head) and `composition_6_loss_{TAG}.png`.
- Per-model outputs live in each experiment's `results/`, tagged with the model name
  (`{TAG}` = model basename, dots → `_`):
  - `induction_1_cliff_{TAG}.png` — per-position loss; the cliff at the 2nd copy is
    in-context learning.
  - `induction_2_heatmap_{TAG}.png` — induction score per (layer, head); dots mark
    heads above threshold.
  - `induction_3_ablation_{TAG}.png` — clean vs ablated loss; the cliff disappears
    when the induction heads are removed.
  - `tokens_{TAG}.txt` — the exact stimulus: position, token id, copy1/copy2 role,
    decoded string.
- Results so far (01): gpt2 (36.5× loss increase on ablation), Qwen3-1.7B (2.9× — heavy
  self-repair/redundancy), Qwen3.5-2B (23.3× — hybrid model, induction concentrated
  in its few softmax-attention layers).
- Results (02, specificity): induction-head ablation sits 10σ (gpt2), 11σ (Qwen3-1.7B),
  and 34σ (Qwen3.5-2B) above the mean of 10 random same-size ablations — the damage is
  specific to the circuit, not to losing k heads.
- Results (03, composition): cutting prev-token heads collapses the induction heads' own
  attention without touching them. Qwen3-1.7B: 7 heads at thresh 0.5 → induction score
  0.65→0.23, loss 2.7×. gpt2: needs thresh 0.3 (10 heads) → 0.58→0.18, loss 11.4× (its
  prev-token signal is redundant across many weak heads). Qwen3.5-2B: NO prev-token heads
  in its 6 softmax layers at either threshold — the upstream half of the circuit
  apparently lives in the linear-attention layers (open thread).

## Running

The script runs under WSL (**distro Ubuntu-24.04, not the default docker-desktop**)
with the venv at `/root/.venv`. Seconds on a 12 GB GPU once weights are cached.

```bash
source /root/.venv/bin/activate
python 01_induction_heads/induction_heads.py                  # gpt2-small (default)
python 01_induction_heads/induction_heads.py Qwen/Qwen3-1.7B  # any supported model
python 02_random_control/random_control.py Qwen/Qwen3.5-2B    # newer archs via TransformerBridge
```

From Windows: `wsl -d Ubuntu-24.04 -u root -- sh -c "cd /mnt/c/Users/Geo/Documents/mech-interp && /root/.venv/bin/python 01_induction_heads/induction_heads.py"`

Charts are written to the repo root (matplotlib uses the headless `Agg` backend —
nothing is displayed, only saved).

## How the experiment works

1. **Serve** — `HookedTransformer.from_pretrained(MODEL)` loads the model with
   TransformerLens hooks on every activation. Models missing from HookedTransformer's
   registry (e.g. Qwen3.5) fall back to `TransformerBridge.boot_transformers` +
   `enable_compatibility_mode()`, which re-exposes the classic hook names. Non-gpt2
   models load in bf16 to fit 12 GB.
2. **Measure** — build `[BOS] r0..r49 r0..r49` from random tokens (falls back to EOS
   or no prefix for tokenizers without BOS, e.g. Qwen). Random tokens are unpredictable
   *except* by looking back at the first copy, so low loss on the 2nd copy isolates
   in-context learning. Each head gets an "induction score": its average attention on
   the diagonal stripe at offset `-(SEQ-1)`, i.e. attention from each token back to the
   token that **followed its previous occurrence**. Hybrid architectures (Qwen3.5
   interleaves linear-attention layers with no attention pattern) are handled by scoring
   only the layers that expose `hook_pattern`; the rest stay 0 in the heatmap.
3. **Break** — heads scoring ≥ `THRESH` (0.3) are zero-ablated via forward hooks on
   `blocks.{L}.attn.hook_z` (per-head output, shape `[batch, pos, head, d_head]`).
   Re-measuring shows the 2nd-copy loss jumping several-fold: causal proof the
   circuit drives the behaviour.
4. **Chart** — the three PNGs above.

## Conventions & gotchas

- Inference only: `torch.set_grad_enabled(False)` at the top; keep it that way for
  new experiments unless you're training probes.
- `loss_per_token` gives `N-1` losses for `N` tokens — `loss[i]` is the loss predicting
  token `i+1`. The "induction region" is `slice(P + SEQ - 1, P + 2*SEQ - 1)` where `P`
  is the prefix length (1 with a BOS/EOS prefix, 0 without).
- `SEQ` and `THRESH` at the top of the script are the only knobs; the model comes from
  `sys.argv[1]`.
- Cross-model caution: different tokenizers draw *different random tokens* (vocab sizes
  differ), so compare the clean-vs-ablated **ratio** within a model, not raw nats across
  models.
- Seeded with `torch.manual_seed(0)` so runs (and charts) are reproducible.
