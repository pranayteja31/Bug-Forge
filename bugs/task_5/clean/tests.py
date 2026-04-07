from cart import estimate_shipping


def test_west_zone_rate():
    result = estimate_shipping("94105", 2.0)
    assert result == 9.0, f"Expected 9.0, got {result}"


def test_standard_zone_rate():
    result = estimate_shipping("10001", 2.0)
    assert result == 6.0, f"Expected 6.0, got {result}"


def test_west_zone_prefix_8():
    result = estimate_shipping("80202", 4.0)
    assert result == 10.0, f"Expected 10.0, got {result}"


if __name__ == "__main__":
    test_west_zone_rate()
    test_standard_zone_rate()
    test_west_zone_prefix_8()
    print("ALL_TESTS_PASSED")
