from willrec.eval.metrics import dcg_at_k, ndcg_at_k, mrr_at_k

def test_metrics_basic():
    rels = [1, 0, 0]
    assert dcg_at_k(rels, 3) > 0
    assert 0 <= ndcg_at_k(rels, 3) <= 1
    assert mrr_at_k(rels, 3) == 1.0
