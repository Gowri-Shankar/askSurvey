from preprocessing import clean_text


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_strips_newlines():
    assert clean_text("\n\nhello\n\n") == "hello"
