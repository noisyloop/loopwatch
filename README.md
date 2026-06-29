# loopwatch

<!-- badges: placeholder -->
![build](https://img.shields.io/badge/build-placeholder-lightgrey)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

A detection-engineering tool for spotting LLM-assisted *engagement bots* on
social platforms — accounts that generate replies, quote-posts, and comments
with a language model in the loop. loopwatch scores accounts from manually
collected post histories using behavioral, stylometric, and temporal signals,
and attempts probabilistic model attribution.

loopwatch's particular focus is the **feedback loop**: how an automated
account's behavior shifts when it detects attention. Bots that optimize for
engagement change measurably once they start getting it — reply cadence
tightens, vocabulary narrows around what worked, posting volume spikes when a
new keyword target lands. loopwatch is built to surface those *changes over
time*, not just to take a static snapshot.

It is a research and triage aid, not an oracle. Read [Known
limitations](#known-limitations) before you trust a score.

---

## What it is, and what it isn't

- **It is** a scoring tool that turns a JSON dump of one account's posts into a
  set of interpretable signals plus a combined suspicion score.
- **It is** pure-stdlib Python. No `pip install`, no service, no API keys, no
  network calls. You bring the data; it does math on it.
- **It is not** a classifier with a calibrated false-positive rate. The
  thresholds are heuristics tuned on a small hand-labeled corpus. Treat outputs
  as leads.
- **It does not** access any platform API or scrape anything. Collection is
  your problem, and deliberately so (see [Input format](#input-format)).
- **It does not** prove automation. Stylometry and timing are circumstantial.
  A high score means "worth a human look," nothing stronger.

---

## Install

No dependencies. Clone and run.

```bash
git clone https://github.com/noisyloop/loopwatch.git
cd loopwatch
python3 -m loopwatch --help
```

Python 3.9+ (standard library only). There is intentionally no `requirements.txt`.

---

## Usage

Score a single account's collected posts:

```bash
python3 -m loopwatch score account.json
```

Emit the full per-signal breakdown as JSON (for piping into your own tooling):

```bash
python3 -m loopwatch score account.json --json
```

Run model attribution alone:

```bash
python3 -m loopwatch attribute account.json
```

Adjust the rolling window (in days) used for drift and growth signals:

```bash
python3 -m loopwatch score account.json --window 7
```

Typical human-readable output:

```
account: @example_handle    posts: 412    span: 2025-11-03 .. 2026-06-21

  BEHAVIORAL
    reply_timing_consistency   0.81   tight inter-reply gaps (low variance)
    length_uniformity          0.74   reply lengths cluster narrowly
    reply_ratio                0.93   93% of posts are replies
    engagement_asymmetry       0.88   emits >> receives
    burst_pattern              0.66   3 bursts > 20 posts/hr

  STYLOMETRIC
    topic_coherence            0.71   unusually on-topic across contexts
    vocabulary_decay           0.69   lexical novelty falling over time
    hedge_absence              0.77   near-zero hedging language
    em_dash_frequency          0.58   em-dashes at 2.1x human baseline
    sentence_entropy           0.64   low structural variation

  TEMPORAL / GROWTH
    volume_spike               0.72   2 spikes aligned with follower gains
    lexical_novelty_decay      0.70   new-token rate down 41% post-spike
    behavioral_drift           0.63   cadence tightened after attention

  ATTRIBUTION (probabilistic — see limitations)
    GPT-4o mini        0.34
    Hermes 3           0.21
    Claude             0.12
    (others < 0.10)

  COMBINED SUSPICION   0.73  /  1.00     [REVIEW]
```

Scores are in `[0, 1]`. Higher means more consistent with an LLM-in-the-loop
account. The `[REVIEW]` / `[LIKELY HUMAN]` / `[FLAG]` band is a convenience
label, not a verdict.

---

## Input format

loopwatch reads a single JSON object describing one account and its posts. You
collect this however you like — manual copy-paste, your own exporter, a
notebook. There is **no built-in collector** on purpose: keeping ingestion out
of scope keeps the tool free of platform Terms-of-Service entanglements and API
keys, and forces you to look at the data you're feeding it.

```json
{
  "handle": "example_handle",
  "collected_at": "2026-06-21T00:00:00Z",
  "account": {
    "followers": 1840,
    "following": 51,
    "created_at": "2025-10-30T00:00:00Z"
  },
  "posts": [
    {
      "id": "p_0001",
      "timestamp": "2026-06-20T14:03:11Z",
      "text": "Great point — the key thing here is signal, not noise.",
      "is_reply": true,
      "in_reply_to": "u_other",
      "likes": 2,
      "reposts": 0,
      "keywords_targeted": ["signal", "noise"]
    }
  ]
}
```

Field notes:

| Field | Required | Used by |
| --- | --- | --- |
| `posts[].timestamp` | yes | all temporal/behavioral signals (ISO 8601, UTC) |
| `posts[].text` | yes | all stylometric signals |
| `posts[].is_reply` | recommended | reply ratio, engagement asymmetry |
| `posts[].likes` / `reposts` | optional | engagement asymmetry, volume/growth alignment |
| `posts[].keywords_targeted` | optional | new-keyword-target drift detection |
| `account.followers` / `created_at` | optional | growth-aligned spike scoring |

Missing optional fields degrade gracefully — the affected signals report `null`
and are dropped from the combined score rather than guessed. More posts and a
longer time span make every signal more reliable; see [Known
limitations](#known-limitations) for the small-sample caveats.

---

## Signals

Each signal returns a normalized `[0, 1]` value plus a short human-readable
note. None is decisive alone; the combined score is a weighted aggregate of the
non-null signals.

| Signal | Family | What it measures | High value suggests |
| --- | --- | --- | --- |
| `reply_timing_consistency` | behavioral | Variance of inter-reply gaps | Machine-paced cadence; low jitter |
| `length_uniformity` | behavioral | Spread of reply character lengths | Template-bounded generation |
| `reply_ratio` | behavioral | Share of posts that are replies | Engagement-farming posture |
| `engagement_asymmetry` | behavioral | Output volume vs. received engagement | Emits far more than it earns |
| `burst_pattern` | behavioral | Clustered high-rate posting windows | Scheduled or triggered batches |
| `topic_coherence` | stylometric | On-topic consistency across contexts | Prompted, narrowly-scoped output |
| `vocabulary_decay` | stylometric | Falling rate of novel tokens over time | Convergence on a working register |
| `hedge_absence` | stylometric | Frequency of hedging/uncertainty markers | Confident, un-hedged generation |
| `em_dash_frequency` | stylometric | Em-dash rate vs. human baseline | A persistent LLM punctuation tell |
| `sentence_entropy` | stylometric | Structural variability of sentences | Low variation; formulaic phrasing |
| `volume_spike` | temporal | Posting-volume spikes vs. baseline | Activity surging with attention |
| `lexical_novelty_decay` | temporal | Drop in new-token rate after spikes | Repeating what got engagement |
| `behavioral_drift` | temporal | Change in behavior across rolling windows | Adaptation to the feedback loop |

### A note on the em-dash signal

The em-dash is a useful *weak* tell, not a strong one. Its baseline is drifting:
the prevalence of LLM-written text online has raised human em-dash usage too,
and style guides vary. loopwatch weights it low and reports the raw ratio so you
can judge it yourself. Do not flag an account on punctuation alone.

### The feedback-loop signals

`volume_spike`, `lexical_novelty_decay`, and `behavioral_drift` exist to catch
the thing static detectors miss: an account that *changes* when it starts
winning. The detector computes each signal over rolling windows and looks for
inflection points aligned with follower gains, engagement jumps, or the
appearance of new `keywords_targeted`. An account that tightens its cadence and
narrows its vocabulary right after a spike is exhibiting the optimization
pattern this tool is named for.

This is also where Goodhart's Law cuts both ways — see [Known
limitations](#known-limitations).

---

## Model attribution

loopwatch produces a **probability distribution** over candidate model families
based on stylometric fingerprints (punctuation distributions, hedge patterns,
list/structure habits, characteristic phrasings, refusal and disclaimer
shapes). Candidates:

- Hermes 3
- Hermes Agent (agent-framework output)
- GPT-4o mini
- GPT-5 series
- Mistral
- Claude
- DeepSeek V4
- OpenClaw
- raw Llama (un-fine-tuned base/instruct output)

**Read this before quoting an attribution number.** Attribution is the weakest
claim loopwatch makes:

- Fingerprints are statistical priors over short text, not signatures. They
  shift with every model release, system prompt, sampling temperature, and
  post-processing step.
- Fine-tuning, paraphrase passes, and human editing all blur the signal toward
  noise. A determined operator can defeat attribution entirely.
- A confident-looking distribution on a handful of posts is not confident.
  Attribution needs volume; with few posts the output approaches the prior.
- Treat attribution as "which family does this *resemble*," never as
  provenance. It is a hint for an analyst, not evidence.

---

## Known limitations

loopwatch is built by and for people who already know detection is adversarial.
The limitations below are not edge cases — they are the operating environment.

### Adversarial evasion

Every signal here is evadable by an operator who knows it exists, and this
README tells them all of them. Inject jitter into reply timing, vary lengths,
sprinkle hedges, throttle bursts, rotate vocabulary, strip em-dashes — each
counter is cheap. loopwatch raises the cost of *low-effort* automation and
gives analysts leads; it does not stop a motivated, informed adversary. Anyone
claiming otherwise is selling something.

### Goodhart's Law

Once a signal becomes a target, it stops being a good measure. Two ways this
bites:

1. **Operators optimize against the metric.** Published detectors train the
   bots. Expect the population you're measuring to drift away from your
   thresholds precisely because the thresholds exist.
2. **Adversarial flooding.** An operator who knows the `behavioral_drift` and
   `volume_spike` logic can deliberately inject human-looking noise — random
   gaps, off-topic posts, manufactured hesitation — to suppress the very
   feedback-loop signals loopwatch keys on. The growth signals are designed
   with this in mind (they look for *inflection*, not absolute levels), but a
   patient adversary can still launder the loop into the baseline. Calibrate
   against your own labeled data and re-calibrate often.

### The stylometric arms race

Stylometry is a moving target. As base models improve and as LLM text saturates
training corpora, the human/machine boundary blurs in both directions: models
read more human, humans (writing alongside models) read more machine. The
em-dash baseline shift is one visible example; there will be more. Any
fingerprint or threshold in this tool has a shelf life. Re-validate before you
rely on it, and assume your gold-standard labels are decaying too.

### Small-sample and collection bias

Signals are unreliable below ~50 posts and meaningfully shaky below ~150. Manual
collection introduces selection bias — you tend to collect the accounts that
already look suspicious, which inflates apparent precision. loopwatch cannot
correct for how you sampled.

### What a high score does and doesn't mean

A high combined score means the account's *observable behavior is consistent
with* an LLM-in-the-loop engagement pattern. It does not establish automation,
intent, identity, or coordination. It is a reason to look closer, full stop.

---

## Research

The signal design draws on the social-bot detection literature. These are the
core references; the behavioral and feature-based framing in particular owes a
lot to this body of work.

- Ferrara, E., Varol, O., Davis, C., Menczer, F., & Flammini, A. (2016). *The
  Rise of Social Bots.* Communications of the ACM, 59(7), 96–104.
- Varol, O., Ferrara, E., Davis, C. A., Menczer, F., & Flammini, A. (2017).
  *Online Human-Bot Interactions: Detection, Estimation, and Characterization.*
  Proceedings of ICWSM 2017.
- Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017).
  *The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the
  Arms Race.* Proceedings of WWW 2017 Companion.
- Kudugunta, S., & Ferrara, E. (2018). *Deep Neural Networks for Bot Detection.*
  Information Sciences, 467, 312–322.

loopwatch is a stdlib, feature-transparent take on these ideas, updated for the
era of cheap LLM-generated engagement and the feedback-loop dynamics that come
with it.

---

## Contributing

Issues and PRs welcome. Useful contributions, in rough priority order:

- **Labeled data and calibration.** The thresholds need adversarial,
  re-validated ground truth more than they need new signals.
- **New or refined signals**, with a clear statement of what they measure and
  how they're evaded.
- **Attribution fingerprints** for current model releases, with the corpus and
  method used to derive them.

Keep the zero-dependency, pure-stdlib constraint. If a change needs a third
party package, it probably belongs in a separate tool.

### Upstream: agent-native port

The eventual goal is an agent-native port that lets an analysis agent run
loopwatch's signals as tools rather than as a CLI batch job. That work is
tracked upstream against the Hermes Agent framework:
**NousResearch/hermes-agent issue #496** is the target for the port. If you're
interested in the agent integration, start there.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
