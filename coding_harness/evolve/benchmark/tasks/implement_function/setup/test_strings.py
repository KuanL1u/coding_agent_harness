from strings import reverse_words


def test_basic():
    assert reverse_words("hello world") == "world hello"


def test_three_words():
    assert reverse_words("a b c") == "c b a"


def test_collapses_whitespace():
    assert reverse_words("  the   quick  brown ") == "brown quick the"


def test_single_word():
    assert reverse_words("solo") == "solo"


def test_empty():
    assert reverse_words("") == ""
