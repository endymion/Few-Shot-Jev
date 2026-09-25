from .metrics import hierarchical_accuracy_delta, summarize


def test_multiclass_summary_is_perfect_for_perfect_probability_rows():
    rows = [{"actual": "World", "prediction": "World", "probabilities": {"World": 1.0}},
            {"actual": "Sports", "prediction": "Sports", "probabilities": {"Sports": 1.0}},
            {"actual": "Business", "prediction": "Business", "probabilities": {"Business": 1.0}},
            {"actual": "Sci/Tech", "prediction": "Sci/Tech", "probabilities": {"Sci/Tech": 1.0}}]
    result = summarize(rows)
    assert result["accuracy"] == result["macro_f1"] == 1.0
    assert result["log_loss"] == result["brier"] == result["ece"] == 0.0


def test_the_hierarchical_bootstrap_is_deterministic():
    zero = [{"target_id": "a", "actual": "World", "prediction": "Sports"},
            {"target_id": "b", "actual": "World", "prediction": "World"}]
    few = [{"target_id": target, "actual": "World", "prediction": "World", "draw_seed": draw}
           for draw in (0, 1) for target in ("a", "b")]
    assert hierarchical_accuracy_delta(zero, few, resamples=100, seed=3) == hierarchical_accuracy_delta(zero, few, resamples=100, seed=3)
