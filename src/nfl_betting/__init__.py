"""Core package for nfl-betting tooling."""

from importlib import metadata

try:
    __version__ = metadata.version("nfl-betting")
except metadata.PackageNotFoundError:  # pragma: no cover - during editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
