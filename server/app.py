from openenv.core.env_server import create_app
from bugforge.models import BugforgeAction, BugforgeObservation
from bugforge.server.bugforge_environment import BugforgeEnvironment

app = create_app(
    BugforgeEnvironment,
    BugforgeAction,
    BugforgeObservation,
    env_name="bugforge",
)


def main():
    import uvicorn
    uvicorn.run(
        "bugforge.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()