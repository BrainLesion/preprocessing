from path import Path
import os


def turbopath(ipath):
    """_summary_

    Args:
        ipath (_type_): _description_

    Returns:
        _type_: _description_
    """
    turbop = Path(
        os.path.normpath(
            os.path.abspath(
                ipath,
            )
        )
    )
    return turbop


def name_extractor(input_path):
    input_path = turbopath(input_path)
    file_name = input_path.name
    parts = file_name.split(".")
    return parts[0]
