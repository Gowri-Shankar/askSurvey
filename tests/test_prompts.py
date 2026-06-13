from prompts import build_classification_prompt, TOPIC_LABELS, SENTIMENT_LABELS


def test_prompt_contains_review_text():
    prompt = build_classification_prompt("Great product, fast delivery")
    assert "Great product, fast delivery" in prompt


def test_prompt_contains_all_topic_labels():
    prompt = build_classification_prompt("test")
    for label in TOPIC_LABELS:
        assert label in prompt


def test_prompt_contains_format_instruction():
    prompt = build_classification_prompt("test")
    assert "Topic--Sentiment" in prompt


def test_sentiment_labels_are_correct():
    assert set(SENTIMENT_LABELS) == {"Positive", "Neutral", "Negative"}
