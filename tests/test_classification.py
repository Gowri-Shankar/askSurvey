from classification import parse_classification_response


def test_parse_classification_response_valid():
    topic, sentiment = parse_classification_response("Product Durability--Negative")
    assert topic == "Product Durability"
    assert sentiment == "Negative"


def test_parse_classification_response_with_markdown():
    topic, sentiment = parse_classification_response("**Product Durability--Positive**")
    assert topic == "Product Durability"
    assert sentiment == "Positive"


def test_parse_classification_response_invalid():
    topic, sentiment = parse_classification_response("unclear output")
    assert topic == "Error"
    assert sentiment == "Error"
