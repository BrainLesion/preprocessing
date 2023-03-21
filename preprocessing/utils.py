from path import Path
import os
from collections import namedtuple


class Modality:
    def __init__(self, modality_name, input_path, output_path, bet) -> None:
        self.modality_name = modality_name
        self.input_path = turbopath(input_path)
        self.output_path = turbopath(output_path)
        self.bet = bet


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
