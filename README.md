# Jev zero-shot vs few-shot on AG News

A small, preregistered benchmark asking one narrow question: do labeled
in-context examples improve Jev's agreement with AG News topic labels? It
compares zero-shot with 4, 8, and 16 class-balanced training examples while
keeping the question, labels, target scoreboard, and request shape fixed.

The primary result is the paired accuracy difference between zero-shot and
4-shot. The other two conditions are secondary dose-response measurements.
Read the frozen [protocol](protocol/PREREGISTRATION.md) before interpreting a
live result.

## Design

- **Dataset:** [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news),
  pinned to a specific upstream revision. The repository downloads it on demand
  and does not ship article text.
- **Scoreboard:** 2,000 official test examples: 500 each of World, Sports,
  Business, and Sci/Tech. Demonstrations come only from the official train split.
- **Few-shot state:** Jev receives `{labeled_examples, target}` and is instructed
  to classify only `target.text`. The zero-shot condition uses the identical
  state shape with `labeled_examples: []`.
- **Examples:** five deterministic, nested, class-balanced draws. Exact
  normalized train/test text matches are excluded.
- **Artifacts:** manifests and live response cache rows contain IDs, labels,
  hashes, model responses, usage, and fingerprints—not article text or secrets.

Jev accepts structured JSON state, including related examples and target
content: [TypeSafe State documentation](https://docs.typesafe.ai/concepts/state).

## Run

```bash
make install
make test
make preflight                 # downloads AG News; sends no Jev requests
.venv/bin/jev-agnews prepare-manifest
git add manifests/scoreboard.jsonl  # freeze this text-free selection in the release
```

`make preflight` prints the 32,000-request live matrix: 2,000 zero-shot calls
plus 30,000 few-shot calls. It never constructs a Jev client or reads a key.

After reviewing the preflight output and setting `TYPESAFE_API_KEY` in a
gitignored `.env`, explicitly choose the request ceiling:

```bash
MAX_REQUESTS=32000 make run
make report
```

Live responses go to ignored `results/responses.jsonl`; `make report` writes
ignored `results/report.json`. They are resumable by a hash of the complete
state and question, so an interrupted run does not repeat completed requests.
Returned token usage is reported; this repository does not invent a dollar rate.

## License and data

Repository code is [MIT](LICENSE). AG News is downloaded from Hugging Face and
remains subject to its upstream terms; no dataset license is implied by this
repository's MIT license.
