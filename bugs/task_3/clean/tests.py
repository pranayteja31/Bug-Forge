from cart import apply_coupon

def test_flat50():
    result = apply_coupon(200, "FLAT50")
    assert result == 150, f"Expected 150, got {result}"

def test_half():
    result = apply_coupon(200, "HALF")
    assert result == 100.0, f"Expected 100.0, got {result}"

def test_no_coupon():
    result = apply_coupon(200, "NONE")
    assert result == 200, f"Expected 200, got {result}"

if __name__ == "__main__":
    test_flat50()
    test_half()
    test_no_coupon()
    print("ALL_TESTS_PASSED")