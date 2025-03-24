def check_and_add_suffix(filename: str, suffix: str) -> str:
    """
    Adds a suffix to the filename if it doesn't already have it.

    Parameters:
        filename (str): The filename to check and potentially modify.
        suffix (str): The suffix to add to the filename.

    Returns:
        str: The filename with the suffix added if needed.
    """
    filename_copy = filename
    if not filename_copy.endswith(suffix):
        filename_copy += suffix
    return filename_copy
