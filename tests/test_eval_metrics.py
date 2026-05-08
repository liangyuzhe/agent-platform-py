"""Unit tests for retrieval evaluation metrics."""

from agents.eval.metrics import (
    accuracy_at_k,
    aggregate_metrics,
    evaluate_single,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_no_hits(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_k_limits_scope(self):
        # Only top-1 considered, "b" is at position 2
        assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 3) == 1.0


class TestPrecisionAtK:
    def test_precision_counts_hits_in_top_k(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_precision_empty_retrieval(self):
        assert precision_at_k([], {"a"}, 3) == 0.0


class TestAccuracyAtK:
    def test_accuracy_requires_all_relevant_docs(self):
        assert accuracy_at_k(["a", "b", "x"], {"a", "b"}, 2) == 1.0
        assert accuracy_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.0

    def test_accuracy_empty_relevant_is_perfect(self):
        assert accuracy_at_k(["x"], set(), 1) == 1.0


class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_rank(self):
        assert mrr(["x", "a", "b"], {"a"}) == 0.5

    def test_third_rank(self):
        assert mrr(["x", "y", "a"], {"a"}) == 1 / 3

    def test_not_found(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_imperfect_ranking(self):
        # Only "a" is relevant; "b" at pos 1 is irrelevant, "a" at pos 2
        val = ndcg_at_k(["b", "a"], {"a"}, 2)
        assert 0.0 < val < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], set(), 2) == 1.0

    def test_single_relevant_at_top(self):
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == 1.0

    def test_single_relevant_not_at_top(self):
        val = ndcg_at_k(["x", "a", "y"], {"a"}, 3)
        assert 0.0 < val < 1.0


class TestEvaluateSingle:
    def test_returns_all_keys(self):
        result = evaluate_single(["a", "b"], {"a"}, k_values=[1, 3])
        assert "mrr" in result
        assert "accuracy@1" in result
        assert "precision@1" in result
        assert "recall@1" in result
        assert "recall@3" in result
        assert "ndcg@1" in result
        assert "ndcg@3" in result


class TestAggregateMetrics:
    def test_averages(self):
        results = [
            {"recall@5": 1.0, "mrr": 0.5},
            {"recall@5": 0.5, "mrr": 0.0},
        ]
        agg = aggregate_metrics(results)
        assert agg["recall@5"] == 0.75
        assert agg["mrr"] == 0.25

    def test_empty(self):
        assert aggregate_metrics([]) == {}
