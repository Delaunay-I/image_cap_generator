from text_preprocessor import text_clean


def test_text_clean():
    raw_text = "I was s over there, a once came 3 people that.!?- a k"
    out_text = text_clean(raw_text)
    assert out_text == "I was over there once came people that"