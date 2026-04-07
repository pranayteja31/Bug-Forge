from utils import build_handle


def welcome_tag(username):
    return f"welcome:{build_handle(username)}"
