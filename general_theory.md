# Entropy and Cross-Entropy — Information Theory for Classification

A deeper pass: where the formulas come from, why they are what they are, and why they
are the right thing to minimize in classification.

---

## Part I — Quick intuition recap

- Self-information of an event with probability `p`: `I(x) = -log p(x)`. Rare events are surprising.
- Entropy `H(p) = -Σ p[k] log p[k]`: the *expected* surprise of distribution `p`.
- Cross-entropy `H(p,q) = -Σ p[k] log q[k]`: expected surprise when `p`'s events are measured with `q`.
- KL divergence: `D_KL(p‖q) = H(p,q) − H(p) ≥ 0`.

---

## Part II — Why `-log p`? (Shannon, 1948)

Shannon's 1948 paper *A Mathematical Theory of Communication* asked: how much information
does an event carry? He required three axioms, and `-log p` is essentially the **unique**
function (up to a positive constant) satisfying them:

1. **Monotonicity**: less probable events carry more information → `I` decreases as `p` increases.
2. **Continuity**: `I(p)` should change smoothly with `p` (no jumps).
3. **Additivity**: for two *independent* events `x`, `y`, the information of both happening
   is the sum: `I(p(x)·p(y)) = I(p(x)) + I(p(y))`.

Try `I(p) = -c·log p`: then `I(p·q) = -c·log(pq) = -c·log p - c·log q = I(p) + I(q)`. ✓
Any function satisfying additivity + continuity is a logarithm (Cauchy's functional
equation argument). The constant `c` only sets the unit: `log₂` → **bits**, `ln` → **nats**.

So `-log p` is not arbitrary — it is the only additive, continuous measure of surprise,
which is exactly what "information" must be if independent pieces of information add up.

## Part III — Why entropy is the expected code length

**Prefix codes and the Kraft inequality.** In any prefix-free (instantaneous) binary code,
codeword lengths `l₁..l_K` satisfy:

```
Σ_k 2^(−l_k) ≤ 1        (Kraft inequality)
```

**Shannon's source coding theorem (intuition).** The optimal prefix code assigns
`l_k ≈ -log₂ p_k` bits to symbol `k`. Then the *expected* code length is:

```
L = Σ_k p_k · l_k = Σ_k p_k · (−log₂ p_k) = H(p)   (in bits)
```

and no uniquely decodable code can have expected length below `H(p)`. So **entropy is the
fundamental limit of compression**: you cannot compress data from distribution `p` below
`H(p)` bits per symbol on average, and you can get arbitrarily close.

Properties that follow:

- `H(p) ≥ 0`; `H = 0` iff one outcome is certain.
- **Maximum at uniform.** By Jensen's inequality (log is concave):
  `H(p) = E[-log p] ≤ -log E[p] ... ` more directly: `H(p) ≤ log K`, with equality iff
  `p` is uniform over `K` outcomes. Uniform = maximum uncertainty = hardest to compress.
- **Chain rule:** `H(X,Y) = H(X) + H(Y|X)` — joint entropy decomposes; conditioning never
  increases entropy: `H(Y|X) ≤ H(Y)`.

## Part IV — Cross-entropy as "wrong code length"

Suppose the data really comes from `p`, but you built your code for `q`, assigning
`l_k = -log₂ q_k` bits to symbol `k`. Expected length:

```
L = Σ_k p_k · (−log₂ q_k) = H(p, q)     (cross-entropy, in bits)
```

So cross-entropy is literally "how many bits per symbol you waste by using the wrong
distribution's code". The waste over the optimal code is the KL divergence:

```
H(p,q) = H(p) + D_KL(p‖q),   D_KL(p‖q) = Σ_k p_k log(p_k/q_k)
```

**Gibbs' inequality: D_KL ≥ 0.** Proof sketch: `ln x ≤ x − 1` for all `x > 0`, so

```
−D_KL = Σ p_k ln(q_k/p_k) ≤ Σ p_k (q_k/p_k − 1) = Σ q_k − Σ p_k = 1 − 1 = 0
```

Hence `D_KL(p‖q) ≥ 0`, equality iff `p = q`. The true distribution's own code is optimal;
any other code is worse on average. Note KL is **asymmetric** — `D_KL(p‖q) ≠ D_KL(q‖p)`.

(History: the name "entropy" was borrowed from statistical mechanics — Gibbs/Boltzmann,
where `S = −k Σ p ln p`; Shannon needed a measure of uncertainty in communication and took
the same form. KL divergence is from Kullback & Leibler, 1951.)

## Part V — Why cross-entropy is *the* classification loss

### 1. It is maximum likelihood in disguise

Training data: samples `x⁽ⁱ⁾` with true classes `cᵢ`. The model says `P(class = c | x) = q[c]`
(e.g. softmax of the logits). The likelihood of the whole dataset under the model is:

```
L(θ) = Π_i q_θ[c_i | x⁽ⁱ⁾]
```

Maximizing likelihood = maximizing `Σ_i log q[c_i]` = **minimizing `Σ_i −log q[c_i]`** —
exactly the cross-entropy of the one-hot empirical distribution against `q`, averaged.
So "minimize cross-entropy" ≡ "maximum likelihood estimation". That is the deep reason
the formula is the right one: it asks the model to assign high probability to the observed
data, and nothing else.

### 2. One-hot labels reduce it to `-log q[c]`

True label as a distribution: `p = onehot(c)`. Then:

```
H(p, q) = −Σ_k p[k] log q[k] = −log q[c]
```

### 3. Well-behaved optimization with softmax + logits

With `q = softmax(l)`, `l = W_y h + b_y` (the **logits** — raw final-layer scores):

```
ℓ = −log softmax(l)[c] = logsumexp(l) − l[c] = log(Σ_k e^{l_k}) − l[c]
∂ℓ/∂l = softmax(l) − onehot(c) = ŷ − y
```

- Computing from logits via **log-sum-exp** is numerically stable (no overflow from `e^{l}`).
- The gradient is just *prediction minus truth* — clean, no softmax-derivative saturation.
- Loss → ∞ as the correct class's probability → 0: confident mistakes are punished hard.

### 4. KL viewpoint

`H(p,q) = H(p) + D_KL(p‖q)` and `H(p)` is constant (0 for one-hot labels), so

```
argmin_q H(p,q)  =  argmin_q D_KL(p‖q)
```

Minimizing cross-entropy is exactly minimizing the KL divergence from the true label
distribution to the predicted one — pulling `q` onto `p`.

### 5. It generalizes: soft targets

Because the full form `−Σ p[k] log q[k]` doesn't require one-hot `p`, you can use
soft labels (label smoothing, `p = [0.9, 0.05, 0.05]`, mixtures, distillation targets).

## Part VI — Numeric example

Binary problem, true class `c = 0`:

| prediction q | CE loss `−log q[0]` |
|--------------|---------------------|
| (0.9, 0.1)   | 0.105 nats          |
| (0.5, 0.5)   | 0.693 nats          |
| (0.1, 0.9)   | 2.303 nats          |

Loss grows exactly as confidence in the wrong class grows.

## TL;DR

- Information must be additive for independent events → forced to be `-log p`.
- Entropy = expected optimal code length = compression limit; max at uniform (`log K`).
- Cross-entropy = expected length using the wrong code = `H(p) + D_KL(p‖q)`.
- In classification, one-hot labels make CE = `−log q[c]` = negative log-likelihood;
  minimizing it is MLE and equivalently minimizing KL to the true label; with softmax
  logits it gives the stable form `logsumexp(l) − l[c]` and gradient `ŷ − y`.
