from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


def check_exist_mkdir(path: Path) -> None:
    if "." in str(path):
        path = path.parent
    if not path.exists():
        path.resolve().mkdir(parents=True, exist_ok=True)
        print(f"makedir {path.resolve()}")
