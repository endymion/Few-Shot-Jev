# Preregistration: Jev in-context examples on AG News

This protocol is frozen before live Jev calls. Any later exploratory work must be
separated from this result and must not overwrite this file.

## Question and primary outcome

Does supplying one labeled AG News training example per class improve Jev's
agreement with ground-truth labels on a previously untouched AG News test
scoreboard? The primary estimand is **4-shot minus zero-shot accuracy**. A
hierarchical 95% percentile bootstrap resamples targets and the five
demonstration draws, both with replacement, for 2,000 resamples using seed
`20260925`.

## Data and isolation

- Dataset: `fancyzhx/ag_news` at revision
  `eb185aade064a813bc0b7f42de02595523103ca4`.
- Scoreboard: exactly 500 official test examples per class, sampled with
  `random.Random(20260925)` independently within the fixed class order World,
  Sports, Business, Sci/Tech; then ordered by class and upstream row index.
- Demonstrations: five four-per-class samples from the train split, using seeds
  0–4. A train row whose normalized text exactly matches a scoreboard row is
  excluded. No test label or remaining test item enters example selection.
- The repository ships only source IDs, labels, and normalized-text hashes in
  the manifest. It does not redistribute AG News article text.

## Conditions and prompt

All conditions use one `choice` question with the same instructions and choice
criteria `World`, `Sports`, `Business`, and `Sci/Tech`. Every state contains
`labeled_examples` and `target`; zero-shot uses an empty example list.

For each draw, demonstrations are nested: 4-shot has one example per class,
8-shot has two, and 16-shot has four. Their display order is a deterministic
draw-specific shuffle. Zero-shot runs once per target. Each few-shot level runs
once per target per draw: 32,000 calls total.

## Secondary reporting

Report accuracy, macro-F1, per-class recall, multiclass log loss, multiclass
Brier score, ten-bin top-label ECE/reliability data, latency, request count,
and returned token usage. 8- and 16-shot results are secondary dose-response
analyses; no result changes the primary comparison or its metric.

## Execution gate

`preflight` is run and recorded before live execution. The live command demands
`--approve` and a request ceiling of at least 32,000. Raw response artifacts
contain no article text, examples, or credentials.
