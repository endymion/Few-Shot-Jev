from .data import Article, balanced_scoreboard, demonstration_draws, examples_for_shots, text_hash


def _article(split, index, label, text=None):
    return Article(split, index, text or f"{split} {label} {index}", label)


def test_a_scoreboard_is_balanced_and_stable():
    labels = ("World", "Sports", "Business", "Sci/Tech")
    test = [_article("test", i * 10 + j, label) for i, label in enumerate(labels) for j in range(3)]
    first = balanced_scoreboard(test, per_class=2, seed=4)
    assert first == balanced_scoreboard(test, per_class=2, seed=4)
    assert [row.label for row in first].count("World") == 2


def test_a_demo_draw_drops_exact_normalized_test_text_and_is_nested():
    labels = ("World", "Sports", "Business", "Sci/Tech")
    scoreboard = [_article("test", i, label, f"shared {label}") for i, label in enumerate(labels)]
    train = [_article("train", i * 10 + j, label, f"shared {label}" if j == 0 else f"train {label} {j}")
             for i, label in enumerate(labels) for j in range(6)]
    draw = demonstration_draws(train, scoreboard, seeds=[0])[0]
    assert not {text_hash(row.text) for row in draw} & {text_hash(row.text) for row in scoreboard}
    assert {row.manifest_id for row in examples_for_shots(draw, 4)} <= {row.manifest_id for row in examples_for_shots(draw, 8)}
    assert {row.manifest_id for row in examples_for_shots(draw, 8)} <= {row.manifest_id for row in examples_for_shots(draw, 16)}
    assert [row.label for row in examples_for_shots(draw, 4)] == list(labels)
