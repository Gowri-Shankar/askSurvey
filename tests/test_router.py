from router import normalize_route_label


def test_normalize_route_label_pandas():
    assert normalize_route_label("pandas_metric") == "pandas_metric"


def test_normalize_route_label_rerank():
    assert normalize_route_label("rerank_insight") == "rerank_insight"


def test_normalize_route_label_unknown_defaults_to_rerank():
    assert normalize_route_label("something_else") == "rerank_insight"
