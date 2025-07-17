import functools
import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/preprocessing#citation"


def citation_reminder(func):
    """
    Decorator to remind users to cite brainles-preprocessing.

    The reminder is shown when the environment variable
    `BRAINLES_PREPROCESSING_CITATION_REMINDER` is set to "true" (default).
    To disable the reminder, set the environment variable to "false".

    Environment variable used:
    - BRAINLES_PREPROCESSING_CITATION_REMINDER: Controls whether the reminder is shown.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if (
            os.environ.get("BRAINLES_PREPROCESSING_CITATION_REMINDER", "true").lower()
            == "true"
        ):
            console = Console()
            console.rule("Thank you for using [bold]brainles-preprocessing[/bold]")
            console.print(
                "Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
        return func(*args, **kwargs)

    return wrapper
