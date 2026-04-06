from utils import calculate_discount

def test_discount_10_percent():
    result = calculate_discount(100, 10)
    assert result == 10.0, f"Expected 10.0, got {result}"

def test_discount_50_percent():
    result = calculate_discount(200, 50)
    assert result == 100.0, f"Expected 100.0, got {result}"

def test_discount_zero():
    result = calculate_discount(100, 0)
    assert result == 0.0, f"Expected 0.0, got {result}"

if __name__ == "__main__":
    test_discount_10_percent()
    test_discount_50_percent()
    test_discount_zero()
    print("ALL_TESTS_PASSED")