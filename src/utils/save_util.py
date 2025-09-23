from pathlib import Path
import torch
import os
from datetime import datetime

def find_repo_root(start_path: Path = None) -> Path:
    """
    Find the root directory of the git repository by looking for a .git folder.
    Starts from start_path (or current file's directory if None) and moves up the directory tree.
    Returns the Path to the repository root, or current working directory if not found.
    """
    if start_path is None:
        try:
            start = Path(__file__).resolve()
        except NameError:
            start = Path.cwd()
    else:
        start = Path(start_path).resolve()

    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return Path.cwd()

def make_model_save_path(filename: str = "best_model_wts.pth",
                         repo_root: Path = None,
                         subfolder: str = "logs/model_state_dict_logs",
                         suffix_ts: bool = False) -> Path:
    """
    Create a full path for saving model weights, ensuring the directory exists.
    Optionally appends a timestamp to the filename.
    """
    repo_root = Path(repo_root) if repo_root is not None else find_repo_root()
    logs_dir = repo_root / subfolder
    logs_dir.mkdir(parents=True, exist_ok=True)

    if suffix_ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{ts}{ext}"

    return (logs_dir / filename).resolve()