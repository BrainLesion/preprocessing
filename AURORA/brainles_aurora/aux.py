from path import Path
import os


def turbo_path(the_path):
    turbo_path = Path(
        os.path.normpath(
            os.path.abspath(
                the_path,
            )
        )
    )
    return turbo_path
