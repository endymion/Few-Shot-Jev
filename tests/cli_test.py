from jev_fewshot.cli import request_plan
from jev_fewshot.data import Article


def test_a_preflight_plan_has_exactly_32000_requests_at_the_registered_size():
    labels = ("World", "Sports", "Business", "Sci/Tech")
    scoreboard = [Article("test", i, f"test {i}", labels[i % 4]) for i in range(2000)]
    draws = {seed: [Article("train", seed * 100 + i, f"train {seed} {i}", labels[i % 4])
                    for i in range(16)] for seed in range(5)}
    assert len(list(request_plan(scoreboard, draws))) == 32000
