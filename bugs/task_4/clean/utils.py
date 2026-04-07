def normalize_username(username):
    return username.strip().lower()


def build_handle(username):
    normalized = normalize_username(username)
    return f"@{normalized}"
