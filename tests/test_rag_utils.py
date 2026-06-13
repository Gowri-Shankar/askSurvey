from unittest.mock import MagicMock
from rag import build_context, rerank_documents


def _make_doc(text):
    doc = MagicMock()
    doc.page_content = text
    return doc


def test_build_context_joins_docs():
    docs = [_make_doc("Review A"), _make_doc("Review B")]
    context = build_context(docs)
    assert "Review A" in context
    assert "Review B" in context


def test_build_context_respects_max_docs():
    docs = [_make_doc(f"Review {i}") for i in range(20)]
    context = build_context(docs, max_docs=3)
    assert "Review 0" in context
    assert "Review 2" in context
    assert "Review 3" not in context


def test_rerank_documents_returns_top_k():
    docs = [_make_doc(f"doc {i}") for i in range(10)]

    mock_reranker = MagicMock()
    # Simulate reranker returning results in reverse order
    mock_results = []
    for i in range(10):
        r = MagicMock()
        r.doc_id = 9 - i
        mock_results.append(r)
    mock_reranker.rank.return_value = mock_results

    result = rerank_documents("question", docs, mock_reranker, top_k=3)
    assert len(result) == 3
    # First result should be doc 9 (highest ranked)
    assert result[0].page_content == "doc 9"
