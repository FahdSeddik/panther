from panther import add


def test_add_positive_numbers():
    assert add(1, 2) == 3


def test_add_negative_numbers():
    assert add(-1, -2) == -3


def test_add_positive_and_negative():
    assert add(1, -2) == -1


def test_add_zero():
    assert add(0, 0) == 0


def test_add_large_numbers():
    assert add(1000000, 2000000) == 3000000


def test_add_with_zero():
    assert add(5, 0) == 5
    assert add(0, 5) == 5
