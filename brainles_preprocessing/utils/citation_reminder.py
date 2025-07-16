import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/preprocessing#citation"


def citation_reminder(func):
    """Decorator to remind users to cite brainles-preprocessing."""

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
            os.environ["BRAINLES_PREPROCESSING_CITATION_REMINDER"] = (
                "false"  # Show only once
            )
        return func(*args, **kwargs)

    return wrapper
