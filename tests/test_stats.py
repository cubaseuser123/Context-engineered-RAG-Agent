from evals.stats import (
    bootstrap_mean_ci,
    cohens_kappa,
    mcnemar_exact,
    paired_bootstrap_delta,
    wilson_interval,
)


def test_wilson_handles_the_extremes():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.65 < low < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_brackets_the_point_estimate():
    low, high = wilson_interval(7, 10)
    assert low < 0.7 < high


def test_bootstrap_is_deterministic_for_a_seed():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    first = bootstrap_mean_ci(values, resamples=200, seed=3)
    second = bootstrap_mean_ci(values, resamples=200, seed=3)
    assert (first.low, first.high) == (second.low, second.high)
    assert first.n == 5


def test_paired_delta_points_the_right_way():
    pairs = [(0.0, 1.0)] * 8
    result = paired_bootstrap_delta(pairs, resamples=200)
    assert result["delta"] == 1.0
    assert result["p_positive"] == 1.0


def test_mcnemar_exact_matches_the_textbook_example():
    # 10 discordant pairs, all one way: p = 2 * 0.5^10
    result = mcnemar_exact(10, 0)
    assert abs(result["p_value"] - 2 * 0.5**10) < 1e-12


def test_kappa_is_one_for_perfect_agreement():
    result = cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0])
    assert result["kappa"] == 1.0


def test_kappa_is_negative_when_raters_are_worse_than_chance():
    # balanced but opposite labels: observed 0, expected 0.5 -> kappa -1
    result = cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1])
    assert result["kappa"] == -1.0
