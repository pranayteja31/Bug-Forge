from cart import calculate_total

def test_total_A01():
    result = calculate_total("A01", 10.0)
    assert result == 50.0, f"Expected 50.0, got {result}"

def test_total_B02():
    result = calculate_total("B02", 20.0)
    assert result == 60.0, f"Expected 60.0, got {result}"

def test_total_unknown():
    result = calculate_total("Z99", 10.0)
    assert result == 0.0, f"Expected 0.0, got {result}"

if __name__ == "__main__":
    test_total_A01()
    test_total_B02()
    test_total_unknown()
    print("ALL_TESTS_PASSED")