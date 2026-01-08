GEON_FORMAT_VERSION = 1
GEON_FORMAT_NAME = "geon"

from importlib.metadata import PackageNotFoundError, version

def get_version() -> str:
    try:
        return version("geon")  
    except PackageNotFoundError:
        return "0.0.0+dev"
