import pandas as pd
from classification import parse_classification_response


def test_parse_valid_with_spaces():
    topic, sentiment = parse_classification_response("  Shipping speed -- Negative  ")
    assert topic == "Shipping speed"
    assert sentiment == "Negative"


def test_parse_multiline_takes_last_line():
    response = "some preamble\nProduct Durability--Positive"
    topic, sentiment = parse_classification_response(response)
    assert topic == "Product Durability"
    assert sentiment == "Positive"


def test_parse_invalid_sentiment_returns_error():
    topic, sentiment = parse_classification_response("Product Durability--Maybe")
    assert topic == "Product Durability"
    assert sentiment == "Error"


def test_parse_neutral_sentiment():
    topic, sentiment = parse_classification_response("Value for money--Neutral")
    assert topic == "Value for money"
    assert sentiment == "Neutral"
