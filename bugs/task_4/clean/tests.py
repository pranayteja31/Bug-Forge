from utils import normalize_username, build_handle


def test_normalize_trims_spaces():
    result = normalize_username("  Alice  ")
    assert result == "alice", f"Expected 'alice', got {result!r}"


def test_normalize_handles_case():
    result = normalize_username("BoB")
    assert result == "bob", f"Expected 'bob', got {result!r}"


def test_build_handle():
    result = build_handle("  Carol ")
    assert result == "@carol", f"Expected '@carol', got {result!r}"


if __name__ == "__main__":
    test_normalize_trims_spaces()
    test_normalize_handles_case()
    test_build_handle()
    print("ALL_TESTS_PASSED")
