import os
import shutil


def check_path(path: str) -> str:
    """Create a repository if it does not exists.

    Args:
        path (str): Target path

    Returns:
        str: Target path, with existence assured.
    """
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def rm_dir(path: str) -> None:
    """Delete existing folder, even if it's not empty.

    Args:
        path (str): Target path
    """
    shutil.rmtree(path)