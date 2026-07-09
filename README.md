# Mechanistic Interpretability: Dissecting the Induction Circuit

Hands-on mechanistic interpretability experiments that locate, verify, and dismantle the
**induction circuit** — the attention-head mechanism behind in-context learning — across
three generations of language models: GPT-2-small (2019), Qwen3-1.7B (2025), and the
hybrid-attention Qwen3.5-2B (2026).

Each experiment is a small, self-contained script built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). The sequence
follows the standard interpretability workflow: **observe** a behaviour, **localize** the
components that correlate with it, then **intervene causally** to prove they implement it.

## Key results

| | GPT-2-small | Qwen3-1.7B | Qwen3.5-2B |
|---|---|---|---|
| Architecture | 12L × 12H, all softmax | 28L × 16H, all softmax | 24L × 8H, **6 softmax + 18 linear-attention layers** |
| Induction heads found | 15 | 13 | 14 (all in the 6 softmax layers) |
| Ablating induction heads (2nd-copy loss) | **36.5×** worse | 2.9× worse | **23.3×** worse |
| vs. ablating random heads | 10σ above random | 11σ above random | 34σ above random |
| Cutting upstream prev-token heads | induction scores 0.58 → 0.18 | 0.65 → 0.23 | no prev-token heads exist in softmax layers |

Three findings worth highlighting:

1. **Universality** — all three models, trained years apart by different labs with
   different architectures, grow crisp induction heads at similar relative depth
   (~50–60% through the network).
2. **Self-repair at scale** — Qwen3-1.7B barely degrades (2.9×) when its induction heads
   are removed: larger all-attention models carry redundant backup pathways. GPT-2 has no
   such safety net (36.5×).
3. **Division of labor in hybrid models** — Qwen3.5-2B concentrates induction into its few
   softmax layers (making it *more* ablation-fragile than the older Qwen3, 23.3×), and its
   softmax layers contain **zero** dedicated previous-token heads — the upstream half of
   the circuit apparently lives in the linear-attention layers, which act as natural
   short-range shift registers.

## The experiments

### 01 — Find and break the induction circuit

The canonical first mech-interp experiment. Feed the model a random token block repeated
twice (`[BOS] r0..r49 r0..r49`): random tokens carry no learnable statistics, so the only
way to predict the second copy is to look back at the first — isolating in-context
learning as a visible **loss cliff**. Each head gets an *induction score*: its average
attention on the diagonal where a token attends to the one that **followed its previous
occurrence**. Heads above threshold are then zero-ablated at `hook_z` (per-head output,
before the output projection mixes heads) and the cliff vanishes.

![Induction heatmap, GPT-2](01_induction_heads/results/induction_2_heatmap_gpt2.png)
![Ablation A/B, GPT-2](01_induction_heads/results/induction_3_ablation_gpt2.png)

### 02 — Random-head control (specificity)

Maybe removing *any* k heads hurts that much? Ten seeded trials ablate the same number of
randomly chosen non-induction heads from the same layer pool. They barely move the loss;
the induction ablation sits 10–34σ outside the random distribution. The damage is
specific to the circuit, not to losing k heads.

![Random control, Qwen3.5-2B](02_random_control/results/control_4_random_Qwen3_5-2B.png)

### 03 — Previous-token heads (composition)

Induction heads don't work alone: an earlier **previous-token head** writes "the token
before me was X" into each position, and the induction head reads that signal
(K-composition). This experiment ablates only the upstream prev-token heads and re-scores
the induction heads *inside the broken model*: their attention stripes collapse even
though they were never touched. GPT-2 needs a wider cut (its prev-token signal is spread
across ~10 weak heads — a redundancy dose-response you can reproduce via the threshold
CLI arg), while Qwen3.5-2B has no softmax prev-token heads at all.

![Composition, Qwen3-1.7B](03_prev_token_heads/results/composition_5_scores_Qwen3-1_7B.png)

### 04 — Layer-knockout profile (tracing the circuit into linear attention)

Qwen3.5's linear-attention layers have no attention patterns to score — but they can
still be cut. Knocking out one attention sublayer at a time and re-measuring the
induction heads' stripes *inside the ablated model* yields a layer-by-layer dependency
profile (a lesion study). In GPT-2 the profile validates the method: layer 0 is
foundational, and the prev-token layers dip mildly (redundancy, as found in 03). In
Qwen3.5-2B, the **linear layers immediately preceding the two main induction layers dip
hardest** (L14 feeding L15, L10 feeding L11): the shifted-token signal appears to be
supplied just-in-time by adjacent linear-attention layers rather than by one early
dedicated layer — consistent with linear attention acting as a local shift register.

![Knockout profile, Qwen3.5-2B](04_layer_knockout/results/knockout_7_profile_Qwen3_5-2B.png)

## Reproducing

Requirements: Python 3.12, CUDA GPU (~12 GB; models load in bf16), and:

```bash
pip install torch transformer_lens matplotlib
```

Every script takes an optional model name (default `gpt2`) and writes charts + token
dumps to its own `results/` folder:

```bash
python 01_induction_heads/induction_heads.py
python 01_induction_heads/induction_heads.py Qwen/Qwen3-1.7B
python 02_random_control/random_control.py Qwen/Qwen3.5-2B
python 03_prev_token_heads/prev_token_heads.py gpt2 0.3   # optional prev-token threshold
python 04_layer_knockout/layer_knockout.py Qwen/Qwen3.5-2B
```

Newer architectures absent from `HookedTransformer`'s registry (e.g. Qwen3.5) load
automatically through TransformerLens's `TransformerBridge` with compatibility mode.
Everything is seeded — reruns reproduce the numbers above exactly. Cross-model caveat:
different tokenizers draw different random stimuli, so compare clean-vs-ablated *ratios*
within a model, not raw loss across models.

## References

- Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic, 2021)
- Olsson et al., [*In-context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Anthropic, 2022)
- Nanda et al., [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
